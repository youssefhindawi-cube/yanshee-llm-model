# Q on Yanshee — brain, voice and speech tests

Experiments for putting "Q", Cube Consultants' company AI, on a UBTech Yanshee robot, plus the laptop tooling
built around it. Private: `Q_Brain_Brief_Yanshee.md` / `.pdf` contain an internal section (Part 2).

## What is here

| Path | What it does |
|---|---|
| `q_yanshee.py` | The Q "brain": keyword matching over the speech-ready table in `Q_Brain_Brief_Yanshee.md`. No LLM, no network. Runs on the robot (robot TTS/ASR via YanAPI) or on a laptop (`--text`, `--mic`, Piper voice). |
| `Q_Brain_Brief_Yanshee.md` | The knowledge file Q answers from (public Part 1, internal Part 2 behind `--internal`). |
| `Q_Brain_Brief_Yanshee_ar.md` | The same brain in Arabic: Egyptian keywords and spoken Egyptian answers; `q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md` (Arabic matching: hamza/ة/ى/diacritics/digits/ال-و prefixes normalised). |
| `asr/live_arabic_web.py` | Live Egyptian-Arabic speech-to-text in the browser. `--model whisper|cohere|munsit` picks the recogniser; `--brain Q_Brain_Brief_Yanshee_ar.md` turns it into Q: each spoken question is answered in Egyptian Arabic as text; `--speak voicetut:Hassan` (local) or `--speak munsit:Fares` (cloud) also speaks the answer in the browser, muting the microphone while Q talks. |
| `asr/backends.py` | The three recognisers behind one interface (Whisper + adapter, Cohere + Sygma in bfloat16, Munsit cloud). |
| `asr/tts_backends.py` | The voices behind one interface: `voicetut[:Speaker]` (VoiceTut-TTS, local Egyptian, 19 speakers: 12 male, 7 female), `munsit[:Voice]` (Munsit Faseeh cloud TTS, Fusha/Gulf voices only), `piper[:voice]` (offline baseline). `python3 asr/tts_backends.py voicetut:Hassan "النص" out.wav` makes a file. |
| `asr/voicetut_worker.py` | VoiceTut runs in its own virtual environment (it needs transformers 5); this worker is started by `tts_backends.py` and streams sentences back over a pipe. |
| `scripts/setup_voicetut.sh` | Creates `~/.venvs/voicetut` (OmniVoice + voicetut-tts, sharing the laptop's CUDA PyTorch) and fetches the 2.5 GB model. |
| `asr/tts_samples/` | The same Q greeting spoken by every voice tested (listen before choosing). |
| `asr/live_arabic.py` | Same pipeline in the terminal (`--mic` or `--wav file`). Shares Silero VAD, the denoiser and the transcriber with the web version. |
| `asr/denoise.py` | High-pass + spectral gate used before Whisper. |
| `asr/test_arabic_whisper.py` | Batch comparison of Whisper large-v3 with and without the Arabic adapter on `asr/clips/`. |
| `asr/clips/` | Short synthetic Arabic/English test sentences with reference text. |
| `asr/eval_casablanca_egypt.py` | Egyptian-dialect benchmark: scores ASR systems on the Casablanca Egypt test set (846 clips, 1.02 h) with the Open Universal Arabic ASR Leaderboard normalisation. Systems: `whisper`, `whisper-adapter`, `cohere-greedy`, `cohere-beam8`, `cohere-sygma`, `cloud:<provider>`. |
| `asr/sygma/` | Vendored Sygma decoding (Apache-2.0) and Cohere's model code for `CohereLabs/cohere-transcribe-arabic-07-2026` (gated: accept its terms on Hugging Face, then log in). |
| `asr/cloud_asr.py` | Cloud speech-to-text adapters (OpenAI, ElevenLabs, Deepgram, Azure, Google) with keys from the environment or a git-ignored `.env`. |
| `asr/elevenlabs_transcribe.py` | `python3 asr/elevenlabs_transcribe.py file.m4a` prints the ElevenLabs transcript of any audio file. |
| `asr/results/casablanca_egypt/` | Per-clip predictions and summaries from the benchmark runs below. |
| `asr/clips_fabhaus30/`, `asr/results/fabhaus30/` | Second Egyptian test set (30 clips from `fabhaus/masri_audio_transcription`, MIT) and its results; `--dataset fabhaus30`. |
| `asr/compare_sets.py` | Prints every system on both 30-clip sets side by side. |
| `scripts/download_models.sh` | Fetches the Piper voices and Vosk models that are not in git. |

## Setup (laptop)

```bash
pip install --user -r requirements.txt      # torch first, from pytorch.org, for your CUDA
scripts/download_models.sh                  # Piper voices + Vosk models (~570 MB)
sudo apt install espeak                     # optional fallback voice
scripts/setup_voicetut.sh                   # optional: VoiceTut Egyptian voice in its own venv (~/.venvs/voicetut) + 2.5 GB model
```

## Run

```bash
python3 q_yanshee.py --selftest             # check the keyword brain
python3 q_yanshee.py --text                 # type questions, Q answers with the Piper voice
python3 q_yanshee.py --mic                  # laptop microphone (offline Vosk) + Piper voice
python3 q_yanshee.py --ip 192.168.x.x       # drive a Yanshee over the network (needs YanAPI.py next to the script)

python3 asr/live_arabic_web.py              # then open http://127.0.0.1:8765 and press Start
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md   # speak Egyptian, Q replies in Arabic text
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan   # ... and in an Egyptian voice (local)
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak munsit:Fares    # ... cloud voice (Fusha; needs MUNSIT_API_KEY)
python3 q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md --text --voice voicetut:Hassan   # keyboard in, Egyptian voice out through the speakers
python3 asr/live_arabic.py --wav clip.m4a   # transcribe a recording through the live pipeline
python3 asr/test_arabic_whisper.py asr/clips cuda
```

## Egyptian-dialect benchmark (Casablanca / Egypt / test, 846 clips, 1.02 h)

| System | WER | CER | Time on an RTX 4060 | Breakdowns |
|---|---|---|---|---|
| Cohere transcribe-arabic + Sygma decoding | **26.95%** | **11.99%** | 120 s | 0 |
| Cohere transcribe-arabic, plain 8-beam | 27.43% | 13.00% | 119 s | 1 |
| Cohere transcribe-arabic, greedy | 28.32% | 13.22% | 77 s | 8 |
| Whisper-large-v3 + Egyptian LoRA adapter | 45.91% | 21.94% | 557 s | 3 |
| Whisper-large-v3 | 46.05% | 22.37% | 535 s | 4 |

ElevenLabs Scribe on the first 3 clips only (free-plan quota): WER 25.8%, CER 14.6%, versus 35.5% / 30.0% for
Cohere + Sygma on the same 3 clips. Run the Cohere variants in separate processes: two loaded models exceed 8 GB.

```bash
python3 asr/eval_casablanca_egypt.py --systems whisper
python3 asr/eval_casablanca_egypt.py --systems cohere-sygma
python3 asr/eval_casablanca_egypt.py --systems cloud:elevenlabs --limit 3   # cloud runs are cached per clip
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
