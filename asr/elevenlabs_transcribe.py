#!/usr/bin/env python3
"""Send one audio file to ElevenLabs speech-to-text and print the text.

  python3 asr/elevenlabs_transcribe.py recording.m4a          # any format ffmpeg can read
  python3 asr/elevenlabs_transcribe.py clip1.wav clip2.wav    # several files

The key comes from ELEVENLABS_API_KEY in the environment, or from a line
ELEVENLABS_API_KEY=... in the project's .env file (git-ignored). Each file is one request against
your ElevenLabs quota, billed by audio length.
"""
import io
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_asr import elevenlabs_transcribe  # noqa: E402


def load_16k_mono(path):
    """Decode anything with ffmpeg into 16 kHz mono int16."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    for f in files:
        audio = load_16k_mono(f)
        print("%s  (%.1f s)" % (f, len(audio) / 16000.0))
        try:
            print("   " + elevenlabs_transcribe(audio))
        except Exception as e:
            print("   ERROR: %s" % e)


if __name__ == "__main__":
    main()
