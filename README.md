# Q on Yanshee — brain, voice and speech tests

Experiments for putting "Q", Cube Consultants' company AI, on a UBTech Yanshee robot, plus the laptop tooling
built around it. Private: `Q_Brain_Brief_Yanshee.md` / `.pdf` contain an internal section (Part 2).

## What is here

| Path | What it does |
|---|---|
| `q_yanshee.py` | The Q "brain": keyword matching over the speech-ready table in `Q_Brain_Brief_Yanshee.md`. No LLM, no network. Runs on the robot (robot TTS/ASR via YanAPI) or on a laptop (`--text`, `--mic`, Piper voice). |
| `Q_Brain_Brief_Yanshee.md` | The knowledge file Q answers from (public Part 1, internal Part 2 behind `--internal`). |
| `asr/live_arabic_web.py` | Live Egyptian-Arabic speech-to-text: the browser captures the mic, Whisper large-v3 + an Egyptian LoRA adapter transcribe on the GPU, results stream to a right-to-left web page. |
| `asr/live_arabic.py` | Same pipeline in the terminal (`--mic` or `--wav file`). Shares Silero VAD, the denoiser and the transcriber with the web version. |
| `asr/denoise.py` | High-pass + spectral gate used before Whisper. |
| `asr/test_arabic_whisper.py` | Batch comparison of Whisper large-v3 with and without the Arabic adapter on `asr/clips/`. |
| `asr/clips/` | Short synthetic Arabic/English test sentences with reference text. |
| `scripts/download_models.sh` | Fetches the Piper voices and Vosk models that are not in git. |

## Setup (laptop)

```bash
pip install --user -r requirements.txt      # torch first, from pytorch.org, for your CUDA
scripts/download_models.sh                  # Piper voices + Vosk models (~570 MB)
sudo apt install espeak                     # optional fallback voice
```

## Run

```bash
python3 q_yanshee.py --selftest             # check the keyword brain
python3 q_yanshee.py --text                 # type questions, Q answers with the Piper voice
python3 q_yanshee.py --mic                  # laptop microphone (offline Vosk) + Piper voice
python3 q_yanshee.py --ip 192.168.x.x       # drive a Yanshee over the network (needs YanAPI.py next to the script)

python3 asr/live_arabic_web.py              # then open http://127.0.0.1:8765 and press Start
python3 asr/live_arabic.py --wav clip.m4a   # transcribe a recording through the live pipeline
python3 asr/test_arabic_whisper.py asr/clips cuda
```

## Notes

* Whisper large-v3 needs about 3 GB of VRAM in fp16 (5 GB peak while merging the adapter). The adapter
  (`MohamedAhmedAE/Whisper-large-V3-Arabic`) is a light fine-tune; on standard-Arabic test clips its output
  equals the base model.
* On the development laptop (HP Victus, Linux) the internal mic is only usable through the browser's processing,
  which is why the web version captures audio in the browser. A headset gives far better results than the built-in mic.
* Robot side, still open: `q_yanshee.py` calls the Yanshee's `sync_do_voice_asr` (a chatbot endpoint) and should
  use the dictation functions (`sync_do_voice_iat` / `get_voice_iat`); on the robot the SDK import is
  `from YansheeRobot import YanAPI`. Both Yanshee speech services are Chinese cloud services.
