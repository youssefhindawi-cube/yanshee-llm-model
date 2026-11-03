"""Cloud speech-to-text adapters for the Egyptian comparison. Each takes 16 kHz mono int16 audio and returns text.

Credentials come from environment variables only (never from code or chat):
  openai      OPENAI_API_KEY                       model gpt-4o-transcribe, language ar
  elevenlabs  ELEVENLABS_API_KEY                   model scribe_v2 (falls back to scribe_v1), language ara
  deepgram    DEEPGRAM_API_KEY                     model nova-3, language ar
  azure       AZURE_SPEECH_KEY + AZURE_SPEECH_ENDPOINT (or AZURE_SPEECH_REGION)   locale ar-EG (Egyptian), profanity unmasked
  google      gcloud application-default login + GOOGLE_CLOUD_PROJECT   Speech-to-Text v2, chirp_2, locale ar-EG
  munsit      MUNSIT_API_KEY                       Munsit (CNTXT AI) model munsit, Arabic, dialect auto-detected
  gemini      GEMINI_API_KEY (+ optional GEMINI_MODEL)   Google Gemini via AI Studio, verbatim-transcription prompt
  speechmatics SPEECHMATICS_API_KEY                 batch API v2, language ar, operating point enhanced
"""
import io
import json
import os
import subprocess
import urllib.error
import urllib.request
import uuid
import wave

RATE = 16000


def wav_bytes(audio_int16):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _multipart(fields, files):
    """Minimal multipart/form-data encoder (no third-party SDKs needed)."""
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    for k, (fname, data, ctype) in files.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n" % (b, k, fname, ctype)).encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(("--%s--\r\n" % b).encode())
    return body.getvalue(), "multipart/form-data; boundary=" + b


def _post(url, data, headers, timeout=120):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %d from %s: %s" % (e.code, url.split("?")[0], e.read().decode("utf-8", "replace")[:300]))


def _load_dotenv():
    """Read KEY=value lines from a .env file in the project root (never printed, never committed: .env is git-ignored)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), os.path.dirname(here)):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.replace("export ", "").strip()
                    os.environ.setdefault(k, v.strip().strip("\'\""))


def _env(name):
    v = os.environ.get(name)
    if not v:
        _load_dotenv()
        v = os.environ.get(name)
    if not v:
        raise RuntimeError("set %s in the environment first (export it, or put %s=... in the project's .env file)" % (name, name))
    return v


def openai_transcribe(audio, model="gpt-4o-transcribe"):
    body, ctype = _multipart({"model": model, "language": "ar", "response_format": "json"},
                             {"file": ("clip.wav", wav_bytes(audio), "audio/wav")})
    r = _post("https://api.openai.com/v1/audio/transcriptions", body,
              {"Authorization": "Bearer " + _env("OPENAI_API_KEY"), "Content-Type": ctype})
    return r.get("text", "")


def elevenlabs_transcribe(audio, model="scribe_v2"):
    for m in (model, "scribe_v1"):
        body, ctype = _multipart({"model_id": m, "language_code": "ara", "tag_audio_events": "false", "diarize": "false"},
                                 {"file": ("clip.wav", wav_bytes(audio), "audio/wav")})
        try:
            r = _post("https://api.elevenlabs.io/v1/speech-to-text", body,
                      {"xi-api-key": _env("ELEVENLABS_API_KEY"), "Content-Type": ctype})
            return r.get("text", "")
        except RuntimeError as e:
            if m != "scribe_v1" and ("model" in str(e).lower()):
                continue
            raise


def deepgram_transcribe(audio, model="nova-3"):
    r = _post("https://api.deepgram.com/v1/listen?model=%s&language=ar&smart_format=false&punctuate=true" % model, wav_bytes(audio),
              {"Authorization": "Token " + _env("DEEPGRAM_API_KEY"), "Content-Type": "audio/wav"})
    try:
        return r["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        return ""


def azure_transcribe(audio, language="ar-EG"):
    """Azure AI Speech, REST API for short audio (<= 60 s per clip). Credentials in .env or the environment:
    AZURE_SPEECH_KEY plus either AZURE_SPEECH_ENDPOINT (https://<resource>.cognitiveservices.azure.com, the form new
    resources get) or AZURE_SPEECH_REGION (e.g. eastus, the older regional endpoint). profanity=raw keeps every word
    unmasked, otherwise the service replaces some words with asterisks and the error rate would be wrong."""
    key = _env("AZURE_SPEECH_KEY")
    endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT") or (_load_dotenv() or os.environ.get("AZURE_SPEECH_ENDPOINT"))
    if endpoint:
        base = endpoint.rstrip("/") + "/stt/speech/recognition/conversation/cognitiveservices/v1"
    else:
        base = "https://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1" % _env("AZURE_SPEECH_REGION")
    url = base + "?language=%s&format=simple&profanity=raw" % language
    r = _post(url, wav_bytes(audio), {"Ocp-Apim-Subscription-Key": key,
                                     "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000", "Accept": "application/json"})
    status = r.get("RecognitionStatus")
    if status == "Success":
        return r.get("DisplayText", "")
    if status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout"):
        return ""                       # the service heard no usable speech: score as empty, do not retry
    raise RuntimeError("Azure RecognitionStatus=%s" % status)


_gtoken = {}


def _google_token():
    import time
    if _gtoken.get("exp", 0) > time.time():
        return _gtoken["tok"]
    tok = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    _gtoken.update(tok=tok, exp=time.time() + 50 * 60)
    return tok


def google_transcribe(audio, model="chirp_2", location="us-central1"):
    import base64
    project = _env("GOOGLE_CLOUD_PROJECT")
    url = "https://%s-speech.googleapis.com/v2/projects/%s/locations/%s/recognizers/_:recognize" % (location, project, location)
    body = json.dumps({"config": {"autoDecodingConfig": {}, "languageCodes": ["ar-EG"], "model": model,
                                  "features": {"enableAutomaticPunctuation": True}},
                       "content": base64.b64encode(wav_bytes(audio)).decode()}).encode()
    r = _post(url, body, {"Authorization": "Bearer " + _google_token(), "Content-Type": "application/json",
                          "x-goog-user-project": project})
    return " ".join(res["alternatives"][0]["transcript"] for res in r.get("results", []) if res.get("alternatives")).strip()


def munsit_transcribe(audio, model="munsit"):
    """Munsit (CNTXT AI), Arabic specialist; dialect is detected automatically. Key: MUNSIT_API_KEY (x-api-key header).
    model 'munsit' = Arabic only (default), 'munsit-en-ar' = mixed Arabic/English. 100 credits per minute of audio;
    the free plan has 10,000 credits. Timestamps are switched off to keep responses small (they are free anyway)."""
    body, ctype = _multipart({"model": model, "return_timestamps": "false"},
                             {"file": ("clip.wav", wav_bytes(audio), "audio/wav")})
    r = _post("https://api.munsit.com/api/v1/audio/transcribe", body,
              {"x-api-key": _env("MUNSIT_API_KEY"), "Content-Type": ctype})
    data = r.get("data") or {}
    if "transcription" not in data:
        raise RuntimeError("Munsit response without transcription: %s" % str(r)[:200])
    return data.get("transcription", "")


_gemini_model = {}


def gemini_transcribe(audio, model=None):
    """Google Gemini (AI Studio key, GEMINI_API_KEY): a multimodal LLM asked to transcribe verbatim, not an ASR engine.
    Model from GEMINI_MODEL (default gemini-3.5-flash with thinking off; 3.8-flash was overloaded on 2026-11-03; a retired name is replaced by the one Google's error suggests).
    Free tier: rate-limited per minute and per day; the harness caches answers, so a rerun resumes where it stopped."""
    import base64, time
    key = _env("GEMINI_API_KEY")
    model = model or os.environ.get("GEMINI_MODEL") or _gemini_model.get("name") or "gemini-3.5-flash"
    prompt = ("Transcribe this Egyptian Arabic audio verbatim, exactly as spoken, in Arabic script. "
              "Keep dialect words as pronounced. Do not translate, summarise, add punctuation beyond what is natural, "
              "or add any commentary. Output only the transcript. If there is no speech, output nothing.")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes(audio)).decode()}}]}],
                       "generationConfig": {"temperature": 0, "maxOutputTokens": 512, "thinkingConfig": {"thinkingBudget": 0}}}).encode()   # thinking off: transcription needs none and it avoids time-outs
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    for attempt in range(6):
        try:
            r = _post(url, body, {"Content-Type": "application/json", "x-goog-api-key": key}, timeout=120)
            break
        except RuntimeError as e:
            msg = str(e)
            if "HTTP 404" in msg and not _gemini_model.get("name"):   # retired model: Google's error names the replacement
                import re as _re
                m = _re.search(r"use models/([A-Za-z0-9.\-]+)", msg)
                if not m:
                    raise RuntimeError("Gemini model %s not available and no replacement suggested: %s" % (model, msg[:200]))
                _gemini_model["name"] = m.group(1)
                url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % _gemini_model["name"]
                continue
            if "HTTP 429" in msg or "HTTP 503" in msg:        # free-tier rate limit: wait and retry
                time.sleep(15 * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError("Gemini: rate-limited repeatedly; rerun later, cached clips are kept")
    try:
        parts = r["candidates"][0]["content"]["parts"]
        return " ".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        return ""


def speechmatics_transcribe(audio, language="ar", operating_point="enhanced"):
    """Speechmatics batch API v2 (SPEECHMATICS_API_KEY, Bearer): submit a job, poll until done, fetch the plain-text transcript.
    Arabic is one language code ('ar'); there is no Egyptian locale. Free plan: 240 min of batch audio per month."""
    import time
    key = _env("SPEECHMATICS_API_KEY")
    base = os.environ.get("SPEECHMATICS_URL", "https://asr.api.speechmatics.com/v2")
    config = json.dumps({"type": "transcription", "transcription_config": {"language": language, "operating_point": operating_point}})
    body, ctype = _multipart({"config": config}, {"data_file": ("clip.wav", wav_bytes(audio), "audio/wav")})
    job = _post(base + "/jobs", body, {"Authorization": "Bearer " + key, "Content-Type": ctype})
    jid = job.get("id")
    if not jid:
        raise RuntimeError("Speechmatics did not return a job id: %s" % str(job)[:200])
    for _ in range(120):                                   # poll up to ~4 min
        time.sleep(2)
        req = urllib.request.Request("%s/jobs/%s" % (base, jid), headers={"Authorization": "Bearer " + key})
        st = json.loads(urllib.request.urlopen(req, timeout=60).read()).get("job", {})
        if st.get("status") == "done":
            req = urllib.request.Request("%s/jobs/%s/transcript?format=txt" % (base, jid), headers={"Authorization": "Bearer " + key})
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8").strip()
        if st.get("status") in ("rejected", "deleted", "expired"):
            raise RuntimeError("Speechmatics job %s %s: %s" % (jid, st.get("status"), str(st.get("errors"))[:200]))
    raise RuntimeError("Speechmatics job %s still running after 4 min" % jid)


PROVIDERS = {"openai": openai_transcribe, "elevenlabs": elevenlabs_transcribe, "deepgram": deepgram_transcribe, "munsit": munsit_transcribe,
             "gemini": gemini_transcribe, "speechmatics": speechmatics_transcribe,
             "azure": azure_transcribe, "google": google_transcribe,
             "elevenlabs-v3": lambda audio: elevenlabs_transcribe(audio, model="scribe_v3"),   # newer Scribe, accepted by the API (Nov 2026)
             "elevenlabs-v1": lambda audio: elevenlabs_transcribe(audio, model="scribe_v1")}
