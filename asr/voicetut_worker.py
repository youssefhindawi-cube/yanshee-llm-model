#!/usr/bin/env python3
"""VoiceTut-TTS worker: Egyptian Arabic text-to-speech in its own Python.

VoiceTut (OmniVoice backbone) needs transformers 5, which conflicts with the rest of this project
(transformers 4.57 / huggingface_hub 0.35 in the user site, pinned for lerobot and Cohere). So it runs in a
separate virtual environment (scripts/setup_voicetut.sh -> ~/.venvs/voicetut) and this worker talks to the
main process over stdin/stdout, one JSON object per line:

  in : {"id": 1, "text": "...", "speaker": "Hassan"}         -> synthesise, sentence by sentence
  out: {"id": 1, "chunk": "/path/x.wav", "audio_s": 2.1, "gen_s": 0.9}   (one per sentence, as soon as it is ready)
       {"id": 1, "done": true, "audio_s": 6.3, "gen_s": 2.7}
  in : {"cmd": "quit"}

Standalone test:  ~/.venvs/voicetut/bin/python asr/voicetut_worker.py --text "ازيك عامل ايه؟" --out /tmp/x --speaker Hassan
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mohammedaly22/VoiceTut-TTS", help="HF repo id or local checkpoint folder")
    ap.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    ap.add_argument("--dtype", default=None, help="float16 (default on cuda) or float32 (default on cpu)")
    ap.add_argument("--steps", type=int, default=32, help="diffusion steps: fewer = faster, more = cleaner (model default 32)")
    ap.add_argument("--speaker", default="Ahmed", help="built-in speaker name or id (Ahmed, Ali, Hassan, ... Hadeer, Hanan, ...)")
    ap.add_argument("--out", required=True, help="folder for the WAV chunks")
    ap.add_argument("--text", help="one-shot test instead of the stdin loop")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    proto = os.fdopen(os.dup(1), "w", buffering=1)       # the protocol stream: everything else goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import numpy as np
    import soundfile as sf
    import torch
    from voicetut_tts import GenerationParams, VoiceTutTTS

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = a.dtype or ("float16" if device == "cuda" else "float32")
    t = time.time()
    tts = VoiceTutTTS.from_pretrained(a.model, device=device, dtype=dtype)
    params = GenerationParams(num_step=a.steps)
    speakers = [{"id": s.speaker_id, "name": s.speaker_name, "gender": s.gender} for s in tts.list_speakers()]
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    proto.write(json.dumps({"ready": True, "device": device, "dtype": dtype, "steps": a.steps, "load_s": round(time.time() - t, 1),
                            "vram_gb": round(vram, 2), "sr": tts.sampling_rate, "speakers": speakers}) + "\n")

    def run(req):
        rid, text, speaker = req.get("id"), req.get("text", ""), req.get("speaker") or a.speaker
        t0, total, n = time.time(), 0.0, 0
        for sr, chunk in tts.stream(text, speaker=speaker, params=params):
            n += 1
            path = os.path.join(a.out, "%s_%d.wav" % (rid, n))
            pcm = np.clip(chunk, -1, 1)
            sf.write(path, pcm, sr, subtype="PCM_16")
            total += len(pcm) / sr
            proto.write(json.dumps({"id": rid, "chunk": path, "sr": sr, "audio_s": round(len(pcm) / sr, 2), "gen_s": round(time.time() - t0, 2)}) + "\n")
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        proto.write(json.dumps({"id": rid, "done": True, "chunks": n, "audio_s": round(total, 2), "gen_s": round(time.time() - t0, 2), "peak_gb": round(peak, 2)}) + "\n")

    if a.text:
        run({"id": "test", "text": a.text, "speaker": a.speaker})
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("cmd") == "quit":
            break
        try:
            run(req)
        except Exception as e:  # keep serving: report the failure for this request only
            proto.write(json.dumps({"id": req.get("id"), "done": True, "error": "%s: %s" % (type(e).__name__, e)}) + "\n")


if __name__ == "__main__":
    main()
