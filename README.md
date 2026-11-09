# Q on Yanshee — brain, voice and speech tests

Experiments for putting "Q", Cube Consultants' company AI, on a UBTech Yanshee robot, plus the laptop tooling
built around it. Private: `Q_Brain_Brief_Yanshee.md` / `.pdf` contain an internal section (Part 2).

## What is here

| Path | What it does |
|---|---|
| `q_yanshee.py` | The Q "brain": keyword matching over the speech-ready table in `Q_Brain_Brief_Yanshee.md`. No LLM, no network. Runs on the robot (robot TTS/ASR via YanAPI) or on a laptop (`--text`, `--mic`, Piper voice). |
| `Q_Brain_Brief_Yanshee.md` | The knowledge file Q answers from (public Part 1, internal Part 2 behind `--internal`). |
| `Q_Brain_Brief_Yanshee_ar.md` | The same brain in Arabic: Egyptian keywords and spoken Egyptian answers; `q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md` (Arabic matching: hamza/ة/ى/diacritics/digits/ال-و prefixes normalised). |
| `asr/live_arabic_web.py` | Live Egyptian-Arabic speech-to-text in the browser. `--model whisper|cohere|munsit` picks the recogniser; `--brain Q_Brain_Brief_Yanshee_ar.md` turns it into Q: a question addressed with the wake word "يا كيو" (or a follow-up within 15 s) is answered in Egyptian Arabic as text, the rest is shown as "not for Q" (`--no-wake` answers everything, `--wake-window` changes the 15 s); `--speak voicetut:Hassan` (local) or `--speak munsit:Fares` (cloud) also speaks the answer in the browser, muting the microphone while Q talks. |
| `asr/backends.py` | The three recognisers behind one interface (Whisper + adapter, Cohere + Sygma in bfloat16, Munsit cloud). |
| `asr/tts_backends.py` | The voices behind one interface: `voicetut[:Speaker]` (VoiceTut-TTS, local Egyptian, 19 speakers: 12 male, 7 female), `munsit[:Voice]` (Munsit Faseeh cloud TTS, Fusha/Gulf voices only), `piper[:voice]` (offline baseline). `python3 asr/tts_backends.py voicetut:Hassan "النص" out.wav` makes a file. |
| `asr/robot_audio.py` | The Yanshee's microphone and speakers over the network for the live pipeline (`live_arabic_web.py --robot IP`): ssh to the robot, `arecord` streams its mic to the PC, a small player on the Pi sounds Q's replies, mic gated while Q talks. Also `--record`, `--play`, `--latency`, `--volume` on their own. |
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
| `q_tv.py` | The TV layer: finds a TV on the network by itself at start and casts a reply row's video or image to it over DLNA while Q speaks. Standard library only, Python 3.5+. Used by `q_yanshee.py` and `live_arabic_web.py`; on its own: `python3 q_tv.py --scan`. See [TV layer](#tv-layer). |
| `tv_media/` | The files the TV column of the brain can show (`welcome.mp4` is the greeting's welcome animation, `welcome.png` its still). |
| `scripts/build_tv_media.py` | Turns an image (AVIF, PNG, JPG, ...) into a TV-ready 1920x1080 H.264 video with a fade in, slow zoom and fade out. |
| `scripts/download_models.sh` | Fetches the Piper voices and Vosk models that are not in git. |

## Setup (laptop)

```bash
pip install --user -r requirements.txt      # torch first, from pytorch.org, for your CUDA
scripts/download_models.sh                  # Piper voices + Vosk models (~570 MB)
sudo apt install espeak                     # optional fallback voice
sudo apt install sshpass                    # for --robot (ssh to the Yanshee's Raspberry Pi)
scripts/setup_voicetut.sh                   # optional: VoiceTut Egyptian voice in its own venv (~/.venvs/voicetut) + 2.5 GB model
sudo apt install ffmpeg                     # only to make TV videos (scripts/build_tv_media.py, Pillow: AVIF needs a recent one, 11.3 here); the TV layer itself needs nothing
```

## Run

```bash
python3 q_yanshee.py --selftest             # check the keyword brain
python3 q_yanshee.py --text                 # type questions, Q answers with the Piper voice
python3 q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md --text --voice none   # silent typed test: replies printed only (TV still found and used)
python3 q_yanshee.py --mic                  # laptop microphone (offline Vosk) + Piper voice
python3 q_yanshee.py --ip 192.168.x.x       # drive a Yanshee over the network (needs YanAPI.py next to the script)

python3 asr/live_arabic_web.py              # then open http://127.0.0.1:8765 and press Start
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md   # say "يا كيو ..." in Egyptian, Q replies in Arabic text (--no-wake: answer everything)
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan   # ... and in an Egyptian voice (local)
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak munsit:Fares    # ... cloud voice (Fusha; needs MUNSIT_API_KEY)
python3 q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md --text --voice voicetut:Hassan   # keyboard in, Egyptian voice out through the speakers
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan --robot 192.168.89.99   # the robot's own mic and speakers (volume slider on the page); this PC runs the models
#   ^ the show configuration. With --brain the TV layer is on by default: a TV on the network is found at start and the greeting plays the welcome video on it
python3 asr/live_arabic.py --wav clip.m4a   # transcribe a recording through the live pipeline
python3 asr/test_arabic_whisper.py asr/clips cuda
```

## TV layer

When Q answers with a row that has a TV action, the video or image shows on a TV in the room at the same time as the
voice. Example: say "هاي كيو" or "hi" and the welcome animation (`tv_media/welcome.mp4`) plays on the TV while Q greets you.
It is on by default whenever a brain is loaded (`q_yanshee.py`, `live_arabic_web.py --brain ...`); nothing to type.

### What happens at start

1. The TV search runs in the background, while the speech model loads, so it never delays Q.
2. It tries the TV used last time first (saved in `~/.cache/q_tv/last_tv.json`): reconnects in under a second.
3. Otherwise it scans every real network interface (Wi-Fi, Ethernet; docker, VPN and loopback are skipped) three ways:
   SSDP multicast search for DLNA renderers, an SSDP search sent to every host of the subnet (for networks that drop
   multicast between clients), and a probe of Samsung's renderer port 9197 on every host (some Samsung TVs, such as the
   Q80BA, do not answer SSDP at all). About 7 s on a /24 network. Every device that can play a URL (UPnP AVTransport)
   counts; TVs come first.
4. Found: `TV connected: Samsung Q80BA 55 TV ... at 192.168.1.6`, and the live page shows "📺 TV: ...".
5. Not found: one `WARNING: could not find a TV ...` (the page shows ⚠️), a new search every 30 s, and Q keeps
   listening, answering and speaking as normal. A TV switched on later is picked up; a TV that stops answering is
   dropped, with a warning, and searched for again.

Casting uses the method proven in `yanshee_sim/tv_cast`: this machine serves the file over HTTP (port 8809) and tells
the TV's renderer to play that URL (UPnP `SetAVTransportURI` + `Play`). Asking for the same file again within 20 s
does not restart it.

### Adding a TV action to an answer

The quick-answer table in `Q_Brain_Brief_Yanshee_ar.md` has a fourth column, **التلفزيون (TV)**. Put one of these in a row:

| Cell | What the TV does when Q answers with that row |
|---|---|
| `welcome.mp4` | plays the file once |
| `welcome.mp4 loop` | plays it and starts it again whenever it ends, until another row's action replaces it |
| `stop` | stops whatever is playing |
| (empty) | nothing; the TV stays as it is |

```markdown
| الكلمات المفتاحية | السؤال | الرد (يُقال كما هو) | التلفزيون (TV) |
|---|---|---|---|
| اهلا, ازيك, ..., هاي, مرحبا, hi, hello | Greeting · تحية | أهلاً وسهلاً! أنا كيو ... | welcome.mp4 |
| كيوب ايه, ايه هي كيوب, ... | What is Cube? · ما هي كيوب | كيوب للاستشارات شركة ... | cube_intro.mp4 |
| شكرا, مع السلامه, باي, ... | Closing · ختام | العفو! ... | stop |
```

Files are looked up in `tv_media/` next to the brain file (or `--tv-media DIR`). Only the file name goes in the cell.
Tables without the fourth column (the English brain) work as before. The wake word on its own ("يا كيو") has no TV
action; a wake word that is a greeting ("هاي كيو", "hello Q") counts as the greeting row.

### Making media for the TV

* **Video is the dependable path**: H.264 MP4, 1920x1080 (what the Samsung TVs play reliably over DLNA).
  Images (`.jpg`, `.png`) work too, but the TV's photo viewer can get stuck after repeated pushes.
* From an image (any format Pillow reads, AVIF included):

  ```bash
  python3 scripts/build_tv_media.py picture.avif --out tv_media/welcome.mp4    # 12 s: fade in, slow zoom, fade out
  python3 scripts/build_tv_media.py logo.png --out tv_media/logo.mp4 --secs 8 --bg white --fill 0.6
  ```

  It also writes a 1920x1080 still next to the video (same name, `.png`). Needs Pillow and ffmpeg (laptop).
* Your own clips: re-encode anything unusual first, e.g.
  `ffmpeg -i in.mov -vf scale=1920:1080 -c:v libx264 -pix_fmt yuv420p -movflags +faststart tv_media/clip.mp4`.

### Options (both `q_yanshee.py` and `live_arabic_web.py`)

| Option | Meaning |
|---|---|
| `--tv auto` | default: the last TV used, else the first TV found |
| `--tv 192.168.1.6` | that TV only |
| `--tv Q80` | the TV whose name contains this text (when several are found; the start-up line lists the others) |
| `--tv off` | no TV layer |
| `--tv-media DIR` | folder with the TV files (default `tv_media/` next to the brain file) |
| `--tv-port 8809` | port this machine serves the files on |
| `--tv-rescan 30` | seconds between searches while no TV is found |

### On its own

```bash
python3 q_tv.py --scan                          # list the TVs (DLNA renderers) on the network
python3 q_tv.py tv_media/welcome.mp4            # find a TV and play the file once
python3 q_tv.py --tv 192.168.1.6 stop           # stop playback on that TV
```

### If the TV shows nothing

* **First cast from a new laptop to a Samsung**: the TV may ask on screen whether to allow this device; press
  **Allow** on the remote (asked once per device). Until then it answers with UPnP error 705 "Transport is locked";
  Q stops the TV, retries once, and then logs that hint.
* **No TV found**: the TV must be on (not in deep standby) and on the same network as this machine. `q_tv.py --scan`
  shows what is visible. The warning lists any renderers it did see, in case the name given with `--tv` does not match.
* **The TV fetches nothing**: the TV downloads the file from this machine on port 8809, so a firewall must allow that
  port from the local network (`--tv-port` to use another one).
* **Stuck photo viewer**: change the TV's source or press Home, then greet Q again. Prefer video files.
* Miracast / screen mirroring and the TV's built-in browser do not work for this; see `yanshee_sim/tv_cast/README.md`.

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
