#!/usr/bin/env bash
# VoiceTut-TTS (Egyptian Arabic TTS, OmniVoice backbone) needs transformers 5, which would break this project's
# pinned user-site packages (transformers 4.57 / huggingface_hub 0.35 for lerobot + Cohere). It therefore lives in
# its own virtual environment; asr/voicetut_worker.py runs inside it and the main scripts talk to it over a pipe.
set -e
V="${VOICETUT_VENV:-$HOME/.venvs/voicetut}"
python3 -m venv "$V"
"$V/bin/pip" install --upgrade pip
"$V/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu128      # CUDA 12.8 build for the RTX 4060
"$V/bin/pip" install "git+https://github.com/k2-fsa/OmniVoice.git" voicetut-tts soundfile
# model weights (2.5 GB; the optimizer.bin training file is skipped): cached under ~/.cache/huggingface
"$V/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('mohammedaly22/VoiceTut-TTS', ignore_patterns=['optimizer.bin']))"
echo "VoiceTut ready in $V  ->  $V/bin/python asr/voicetut_worker.py --text 'ازيك عامل ايه؟' --out /tmp/vt --speaker Hassan"
