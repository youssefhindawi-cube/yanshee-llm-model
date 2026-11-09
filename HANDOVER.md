# Handover: llm_model (Q on Yanshee)

What this repository is, what it needs, how to run each part, what the measurements say, and what is still open.
Written 9 Nov 2026. The robot-control side (simulator, GUI, walking, get-up) is the sibling repository
`yanshee_sim` with its own `HANDOVER.md`.

---

## 1. What this is

Experiments for putting **"Q"**, Cube Consultants' company AI, on the UBTECH Yanshee robot, for the exhibition on
**Monday 7 December 2026** (milestone M2, 27 Nov: Q answers spoken or typed questions through the robot).

Three parts, all working on the development laptop:

1. **Q's brain** (`q_yanshee.py`): keyword matching over a speech-ready knowledge table written in Markdown
   (`Q_Brain_Brief_Yanshee.md`, Arabic version `Q_Brain_Brief_Yanshee_ar.md`). **No LLM and no network**: despite the
   folder name there is no language model in the loop yet. It runs on old Python (3.5+) so it can run on the robot.
2. **Egyptian-Arabic speech-to-text** (`asr/`): a live browser pipeline (`live_arabic_web.py`) with three recognisers
   behind one interface (Whisper large-v3, Cohere transcribe-arabic with Sygma decoding, Munsit cloud), plus a
   benchmark harness that produced the numbers in section 5.
3. **Voices** (`asr/tts_backends.py`): VoiceTut-TTS (local Egyptian, 19 speakers), Munsit Faseeh (cloud, Fusha and
   Gulf only), Piper (offline baseline). And the robot's own microphone and speakers as the pipeline's audio over ssh
   (`asr/robot_audio.py`, `--robot IP`).

The intended show path: the robot hears a question starting with the wake word "يا كيو" (or "Hey Q" in English),
the laptop recognises it (Cohere), the brain picks a row, the answer is spoken through the robot's speaker in an
Egyptian voice (VoiceTut Hassan), the microphone muted while Q talks.

`Q_Brain_Brief_Yanshee.md` Part 2 is **internal**: it is only answered with `--internal`, and the repository is private.

---

## 2. Requirements

### Laptop (development machine: HP Victus, Ubuntu, RTX 4060 Laptop 8 GB, Python 3.10.12)

| Need | Details |
|---|---|
| GPU | A CUDA GPU with 8 GB is enough for one model at a time. Whisper large-v3 fp16 takes 3.1 GB (4.9 GB peak with the adapter merge), Cohere bf16 4.2 GB. **Never load two models in one process.** |
| PyTorch | Install first from pytorch.org for your CUDA (laptop has `torch 2.10.0+cu128` with matching `torchaudio`). |
| Packages | `pip install --user -r requirements.txt`: numpy 2.x (<2.3), scipy, transformers 4.x (<5), peft, huggingface_hub (<0.36), soundfile, piper-tts, vosk, silero-vad, librosa, numba, sentencepiece, coverage, pyarrow. |
| Models not in git | `scripts/download_models.sh` fetches Piper voices (`voices/`) and Vosk English models (`stt/`), about 570 MB. Whisper, the adapter and Cohere download from Hugging Face on first use. |
| Cohere model | Gated: accept the terms of `CohereLabs/cohere-transcribe-arabic-07-2026` on Hugging Face, then `huggingface-cli login`. |
| VoiceTut voice | `scripts/setup_voicetut.sh` builds `~/.venvs/voicetut` (it needs transformers 5, so it lives in its own venv and is driven through `asr/voicetut_worker.py`) and fetches a 2.5 GB model. |
| System packages | `sudo apt install sshpass` (robot audio over ssh), `espeak` (optional fallback voice). |
| Cloud keys | A git-ignored `.env` in the repo root with `ELEVENLABS_API_KEY`, `DEEPGRAM_API_KEY`, `MUNSIT_API_KEY`, `GEMINI_API_KEY`, `SPEECHMATICS_API_KEY`. Get your own keys; the previous owner's are not handed over. Only `MUNSIT_API_KEY` is needed for a show setup that uses the Munsit voice or recogniser; the rest were for the benchmark. |
| Microphone | A headset or close microphone for any serious live test. The laptop's internal mic is only usable through the browser's processing, which is why the web version captures audio in the browser. |

**Pins that must hold on the development laptop** (shared user site with lerobot, open3d and others):
`huggingface_hub<0.36`, `transformers<5`, `numpy==2.2.6`, torch and torchaudio at the same version. Run
`pip install --dry-run` before adding anything and refuse whatever touches numpy, torch, transformers or
huggingface_hub. Do not install DeepFilterNet (it downgraded numpy and did not import). Details in `FINDINGS.md` §10.

### The robot (for `--robot IP` and `q_yanshee.py --ip`)

Raspberry Pi 3B, Raspbian 9, Python 3.5, one sound card (1 microphone, 2 speakers, ALSA dmix/dsnoop at 48 kHz),
ssh `pi` / `raspberry` with passwordless sudo, REST on `:9090`. Its Wi-Fi IP changes per network; find it with
`yanshee_sim/tools/find_robot.py`. The robot's own speech services are Chinese cloud services, which is why
recognition and voice run on the laptop. The official SDK on the robot imports as `from YansheeRobot import YanAPI`.

---

## 3. How to run it

From the repository root.

```bash
# Brain only
python3 q_yanshee.py --selftest                    # checks of the keyword matcher (12 questions; one known miss, see §6)
python3 q_yanshee.py --text                        # type questions, Q answers with the Piper voice
python3 q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md --text --voice voicetut:Hassan   # Arabic brain, Egyptian voice
python3 q_yanshee.py --mic                         # laptop mic (offline Vosk, English) + Piper voice
python3 q_yanshee.py --ip 192.168.x.x              # robot TTS/ASR through YanAPI.py placed next to the script (Chinese ASR: not for the show)

# Live Egyptian-Arabic page: open http://127.0.0.1:8765 and press Start
python3 asr/live_arabic_web.py                                       # Whisper, transcription only
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md                       # Q answers as text
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan   # ... spoken, local voice
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak munsit:Fares      # ... cloud voice
python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan --robot ROBOT_IP
#   ^ the show configuration: the robot's mic and speakers, the laptop runs the models, page becomes a monitor with a volume slider
#   --no-wake answers everything; --wake-window 15 is the follow-up window after a wake word
#   TV layer (on by default with --brain): at start Q finds a TV on the network by itself (last TV used, else a scan:
#   SSDP multicast + unicast and Samsung's port 9197), while the models load. A row's "التلفزيون (TV)" column (e.g. the
#   greeting -> tv_media/welcome.mp4) is cast to it over DLNA while Q speaks. No TV: one WARNING, rescans every 30 s,
#   speech and voice run as normal. --tv off | IP | part of the name. The page shows the TV state. (q_tv.py)

# TV layer on its own, and making a TV video from an image (AVIF/PNG/JPG -> 1080p H.264 with fade and slow zoom)
python3 q_tv.py --scan                                       # list the TVs on the network
python3 q_tv.py tv_media/welcome.mp4                         # find a TV, play once   (q_tv.py --tv 192.168.1.6 stop)
python3 scripts/build_tv_media.py IMAGE --out tv_media/NAME.mp4
python3 q_yanshee.py --md Q_Brain_Brief_Yanshee_ar.md --text --voice none   # typed test, silent laptop, TV found automatically
#   A Samsung asks once per new device whether to allow it (UPnP error 705 "Transport is locked" until someone presses Allow).

# Robot audio pieces on their own
python3 asr/robot_audio.py ROBOT_IP --record 5 out.wav | --play file.wav | --latency | --volume 70

# Voices
python3 asr/tts_backends.py voicetut:Hassan "النص" out.wav     # make a file; samples of every voice in asr/tts_samples/

# Offline tools
python3 asr/live_arabic.py --wav clip.m4a          # the live pipeline on a recording, in the terminal
python3 asr/elevenlabs_transcribe.py file.m4a      # cloud transcript of any audio file
python3 asr/eval_casablanca_egypt.py --systems cohere-sygma      # the benchmark (one system per process)
python3 asr/eval_casablanca_egypt.py --systems cloud:elevenlabs --limit 3 --dataset fabhaus30
python3 asr/compare_sets.py                        # every system on both 30-clip sets side by side
```

Shutdown of the robot pipeline: Ctrl-C (closes the ssh link in about 3 s; leftover `arecord` or `q_player` on the Pi
are killed at the next start).

---

## 4. Where to look

1. `README.md`: the file table, setup, run lines, the benchmark table.
2. `FINDINGS.md` (44 KB, the real knowledge of the project): timeline, what is true about the laptop microphone, why
   the live transcriber looks the way it does, Whisper and the adapter, the leaderboard research, the Egyptian
   benchmark and its final ranking, cloud providers, datasets and licences, **§10 environment facts that cost time**,
   §10b-10f the Arabic brain, the voices, the robot audio, sound localisation research, the wake word, and **§11
   open items**.
3. `q_yanshee.py`: `norm()` (Arabic and English normalisation), the table parser, the matcher, the robot and laptop
   I/O. Read the docstring first.
4. `asr/live_arabic_web.py` and `asr/backends.py`, `asr/tts_backends.py`, `asr/robot_audio.py`: the pipeline.
5. `asr/sygma/`: vendored Sygma decoding (Apache-2.0) and Cohere's model code; `ORIGIN.md` says where it came from.
6. `asr/results/`: per-clip predictions and summaries of every benchmark run, cloud results cached per clip.

---

## 5. What the measurements say

Egyptian dialect, Casablanca Egypt test set (846 clips, 1.02 h), word and character error rates:

| System | WER | CER | Time on the RTX 4060 |
|---|---|---|---|
| Cohere transcribe-arabic + Sygma decoding (chosen) | 26.95 % | 11.99 % | 120 s |
| Cohere, plain 8-beam | 27.43 % | 13.00 % | 119 s |
| Cohere, greedy | 28.32 % | 13.22 % | 77 s |
| Whisper large-v3 + Egyptian LoRA adapter | 45.91 % | 21.94 % | 557 s |
| Whisper large-v3 | 46.05 % | 22.37 % | 535 s |

Cloud providers (ElevenLabs Scribe v1/v2/v3, Deepgram, Munsit, Speechmatics, Gemini) were compared on 30-clip
subsets of two Egyptian sets; the final ranking is in `FINDINGS.md` §7. The Whisper adapter adds nothing and should be
dropped. Robot audio: far-field pickup works, Q's own voice does not echo back as a transcript, VoiceTut's first
sound comes 1.5 to 2.5 s after the text reply.

---

## 6. Things that bite

- **Two models in one process exceed 8 GB.** Run each benchmark system in its own process.
- **The robot pipeline runs as root on the Pi** on purpose: ALSA's shared objects belong to whoever opened them
  first and the robot's own services are root. One multiplexed ssh connection (`ControlPath=/tmp/qy-ssh-%C`); a second
  full handshake costs the Pi 4 to 5 s of CPU and starves audio. Anything else talking to the robot during a session
  should reuse that socket.
- **Echo gating**: the robot hears its speakers at about -42 dBFS, so the mic is gated from send until sound plus
  `--robot-tail` (0.6 s), using a measured 0.85 s lag. Changing the volume over REST plays a tip sound; gated too.
- **Without the wake word Q answers everything** it hears in the room. The wake word is on by default now; watch the
  "not for Q" lines for unexpected spellings of "يا كيو".
- **`q_yanshee.py` on the robot** still calls the chatbot endpoint `sync_do_voice_asr`; it should use the dictation
  functions (`sync_do_voice_iat` / `get_voice_iat`). Both are Chinese cloud services, so this path is not the show
  path anyway.
- **Known self-test miss**: "who founded" is not in the founder row's keywords (11/12).
- **Licences**: Casablanca's licence does not permit derivatives (no fine-tuning on it); `asr/sygma` is Apache-2.0;
  the fabhaus30 clips are MIT. Cohere's model is gated.
- Keep `.env` out of git (already ignored) and rotate any key that was ever exposed.
- The robot cannot tell where a voice comes from (one microphone). A USB microphone array (ReSpeaker XVF3000 or
  similar) or face detection through the camera would be needed (`FINDINGS.md` §10e).

---

## 7. Open work

From `FINDINGS.md` §11 and the exhibition plan:

1. **Test with your own voice in front of the robot** in the show configuration; settle `--robot-volume`; compare the
   three recognisers on real speech through the page.
2. **Gestures while Q talks** (head and arm motions from `yanshee_sim` while the reply plays): not started. This is
   the integration point between the two repositories.
3. **A real language model behind Q** if the keyword table is not enough for the show. Nothing is built; the brain's
   `answer()` is the place to put it, and the knowledge file is already speech-ready text. Latency and the no-internet
   show network (see `yanshee_sim/HANDOVER.md` §5) decide whether it runs locally.
4. `q_yanshee.py` fixes: founder keyword, dictation functions, `from YansheeRobot import YanAPI`, English dictation
   test on the robot.
5. Benchmark extensions if budget allows (ElevenLabs on the full hour, Google or Azure `ar-EG`, Audar or Jais locally,
   Sygma's beta tuned on validation data).
6. Ask Munsit whether an Egyptian Faseeh voice is coming; check the Munsit credit balance on app.munsit.com (the API
   does not report it).

---

## 8. Git

- Remote: `git@github-cube:youssefhindawi-cube/yanshee-llm-model.git` (GitHub `youssefhindawi-cube/yanshee-llm-model`,
  private). Ask the owner for access and clone with a plain `git@github.com:` URL.
- Last commit at handover: `7a09f6d` (wake word). Working tree clean.
- Git-ignored and fetched by scripts: `voices/`, `stt/`, `.env`, runtime logs and transcripts.
- Findings go into `FINDINGS.md` with the date; results files into `asr/results/<dataset>/`.
