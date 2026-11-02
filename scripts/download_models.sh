#!/usr/bin/env bash
# Downloads the speech models that are not kept in git (about 570 MB). Run from anywhere.
set -e
cd "$(dirname "$0")/.."
echo "== Piper voices -> voices/"
python3 -m piper.download_voices en_US-lessac-medium en_US-ryan-high en_GB-jenny_dioco-medium ar_JO-kareem-medium --data-dir voices
echo "== Vosk models -> stt/"
mkdir -p stt && cd stt
for m in vosk-model-small-en-us-0.15 vosk-model-en-us-0.22-lgraph; do
  if [ ! -d "$m" ]; then
    curl -sSL -o "$m.zip" "https://alphacephei.com/vosk/models/$m.zip"
    python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$m.zip"
    rm "$m.zip"
  fi
done
echo "Done. Whisper large-v3 and the Arabic adapter (about 4 GB) download themselves from Hugging Face on first use."
