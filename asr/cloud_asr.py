"""Cloud speech-to-text adapters for the Egyptian comparison. Each takes 16 kHz mono int16 audio and returns text.

Credentials come from environment variables only (never from code or chat):
  openai      OPENAI_API_KEY                       model gpt-4o-transcribe, language ar
  elevenlabs  ELEVENLABS_API_KEY                   model scribe_v2 (falls back to scribe_v1), language ara
  deepgram    DEEPGRAM_API_KEY                     model nova-3, language ar
  azure       AZURE_SPEECH_KEY + AZURE_SPEECH_REGION   locale ar-EG (Egyptian)
  google      gcloud application-default login + GOOGLE_CLOUD_PROJECT   Speech-to-Text v2, chirp_2, locale ar-EG
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


def azure_transcribe(audio):
    region = _env("AZURE_SPEECH_REGION")
    url = "https://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=ar-EG&format=simple" % region
    r = _post(url, wav_bytes(audio), {"Ocp-Apim-Subscription-Key": _env("AZURE_SPEECH_KEY"),
                                     "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000", "Accept": "application/json"})
    return r.get("DisplayText", "") if r.get("RecognitionStatus") == "Success" else ""


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


PROVIDERS = {"openai": openai_transcribe, "elevenlabs": elevenlabs_transcribe, "deepgram": deepgram_transcribe,
             "azure": azure_transcribe, "google": google_transcribe,
             "elevenlabs-v3": lambda audio: elevenlabs_transcribe(audio, model="scribe_v3"),   # newer Scribe, accepted by the API (Nov 2026)
             "elevenlabs-v1": lambda audio: elevenlabs_transcribe(audio, model="scribe_v1")}
