"""Text-to-speech engines for Q's Egyptian Arabic replies, behind one tiny interface.

    tts = make_tts("voicetut:Ahmed")         # local, Egyptian (VoiceTut-TTS in its own venv, GPU or CPU)
    tts = make_tts("munsit:Fares")           # cloud, Munsit Faseeh (no Egyptian voices yet: Fusha/Gulf only)
    tts = make_tts("piper:ar_JO-kareem-medium")   # local, robotic baseline
    for sr, pcm in tts.synth("أهلاً بيك في كيوب"):   # chunks of int16 samples, sentence by sentence when possible
        play(sr, pcm)

Engine specs are "engine" or "engine:voice". Keys (Munsit) come from the environment or llm_model/.env.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cloud_asr import _env  # noqa: E402  (reads llm_model/.env too)

VOICETUT_PYTHON = os.path.expanduser("~/.venvs/voicetut/bin/python")
VOICETUT_MODEL = os.environ.get("VOICETUT_MODEL", "mohammedaly22/VoiceTut-TTS")


def wav_bytes(sr, pcm):
    """int16 samples -> WAV file bytes (mono)."""
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return b.getvalue()


def read_wav(data_or_path):
    """WAV bytes or path -> (sr, int16 mono samples)."""
    src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
    with wave.open(src, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("expected 16-bit WAV, got %d-byte samples" % sw)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return sr, pcm


class PiperTTS:
    """Offline baseline (robotic): any .onnx voice in llm_model/voices."""
    def __init__(self, voice="ar_JO-kareem-medium"):
        from piper import PiperVoice
        path = os.path.join(ROOT, "voices", voice + ".onnx")
        if not os.path.exists(path):
            raise RuntimeError("no Piper voice " + path)
        self.voice, self.name, self.sr = PiperVoice.load(path), "piper:" + voice, None

    def synth(self, text):
        pcm, sr = [], None
        for chunk in self.voice.synthesize(text):
            sr = chunk.sample_rate
            pcm.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if pcm:
            self.sr = sr
            yield sr, np.concatenate(pcm)


class MunsitTTS:
    """Munsit Faseeh (cloud): POST text, get a 24 kHz WAV back. Voice = name or voice_id from GET /voices/public."""
    BASE = "https://api.munsit.com/api/v1"
    MODEL = "faseeh-v1-preview"

    def __init__(self, voice="Fares", stability=0.5, speed=1.0):
        self.key = _env("MUNSIT_API_KEY")
        if not self.key:
            raise RuntimeError("MUNSIT_API_KEY is not set (put it in llm_model/.env)")
        self.stability, self.speed = stability, speed
        self.voices = self.list_voices()
        hit = [v for v in self.voices if v.get("voice_id") == voice] or [v for v in self.voices if (v.get("name") or "").lower() == voice.lower()]
        if not hit:
            raise RuntimeError("Munsit voice %r not found; names: %s" % (voice, ", ".join(sorted(v.get("name", "?") for v in self.voices))[:600]))
        self.voice = hit[0]
        self.name = "munsit:%s (%s)" % (self.voice.get("name"), "/".join(self.voice.get("dialect") or ["?"]))
        self.sr = 24000

    def list_voices(self):
        req = urllib.request.Request(self.BASE + "/voices/public", headers={"x-api-key": self.key})
        data = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return data if isinstance(data, list) else (data.get("voices") or data.get("data") or [])

    def synth(self, text):
        body = json.dumps({"voice_id": self.voice["voice_id"], "text": text, "stability": self.stability, "speed": self.speed, "streaming": False}).encode()
        req = urllib.request.Request(self.BASE + "/text-to-speech/" + self.MODEL, data=body, method="POST",
                                     headers={"x-api-key": self.key, "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                r = urllib.request.urlopen(req, timeout=120)
                ct, data = r.headers.get("Content-Type", ""), r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise RuntimeError("Munsit TTS HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:300]))
        if "json" in ct:
            raise RuntimeError("Munsit TTS returned JSON instead of audio: %s" % data[:300].decode(errors="replace"))
        sr, pcm = read_wav(data)
        self.sr = sr
        yield sr, pcm


class VoiceTutTTS:
    """VoiceTut-TTS (local, Egyptian) through asr/voicetut_worker.py running in the ~/.venvs/voicetut Python."""
    def __init__(self, speaker="Ahmed", device=None, steps=32, python=None):
        py = python or os.environ.get("VOICETUT_PYTHON") or VOICETUT_PYTHON
        if not os.path.exists(py):
            raise RuntimeError("VoiceTut environment not found at %s (run scripts/setup_voicetut.sh)" % py)
        self.tmp = tempfile.mkdtemp(prefix="voicetut_")
        cmd = [py, os.path.join(HERE, "voicetut_worker.py"), "--model", VOICETUT_MODEL, "--out", self.tmp, "--steps", str(steps), "--speaker", speaker]
        if device:
            cmd += ["--device", device]
        env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("VoiceTut worker died while loading (see the messages above)")
        self.info = json.loads(line)
        self.speakers = self.info.get("speakers") or []          # [{"id": "spk_01_male", "name": "Ahmed", "gender": "male"}, ...]
        known = {x["id"] for x in self.speakers} | {x["name"] for x in self.speakers}
        if known and speaker not in known:
            raise RuntimeError("VoiceTut speaker %r unknown; choose one of: %s" % (speaker, ", ".join("%s (%s)" % (x["name"], x["gender"]) for x in self.speakers)))
        self.speaker, self.sr, self.lock, self.n = speaker, self.info["sr"], threading.Lock(), 0
        self.name = "voicetut:%s (%s %s, %d steps)" % (speaker, self.info["device"], self.info["dtype"], self.info["steps"])
        self.last = {}

    def synth(self, text):
        with self.lock:
            self.n += 1
            rid = self.n
            self.p.stdin.write(json.dumps({"id": rid, "text": text, "speaker": self.speaker}, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("VoiceTut worker exited")
                msg = json.loads(line)
                if msg.get("id") != rid:
                    continue
                if msg.get("done"):
                    self.last = msg
                    if msg.get("error"):
                        raise RuntimeError("VoiceTut: " + msg["error"])
                    return
                sr, pcm = read_wav(msg["chunk"])
                os.remove(msg["chunk"])
                yield sr, pcm

    def close(self):
        try:
            self.p.stdin.write('{"cmd": "quit"}\n')
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


ENGINES = {"voicetut": VoiceTutTTS, "munsit": MunsitTTS, "piper": PiperTTS}
DEFAULT_VOICE = {"voicetut": "Ahmed", "munsit": "Fares", "piper": "ar_JO-kareem-medium"}


def make_tts(spec, device=None, steps=32):
    """'voicetut', 'voicetut:Hassan', 'munsit:Fares', 'piper:ar_JO-kareem-medium' -> engine with .name, .sr, .synth(text)."""
    engine, _, voice = spec.partition(":")
    engine = engine.strip().lower()
    if engine not in ENGINES:
        raise ValueError("unknown TTS engine %r (choose from %s)" % (engine, ", ".join(ENGINES)))
    voice = voice.strip() or DEFAULT_VOICE[engine]
    if engine == "voicetut":
        return VoiceTutTTS(voice, device=device, steps=steps)
    return ENGINES[engine](voice)


if __name__ == "__main__":  # python3 asr/tts_backends.py munsit:Fares "نص" out.wav
    spec, text, out = sys.argv[1], sys.argv[2], sys.argv[3]
    tts = make_tts(spec)
    t = time.time()
    chunks = list(tts.synth(text))
    pcm = np.concatenate([c[1] for c in chunks])
    open(out, "wb").write(wav_bytes(chunks[0][0], pcm))
    print("%s: %.1f s of audio in %.2f s -> %s" % (tts.name, len(pcm) / chunks[0][0], time.time() - t, out))
