# Findings and work log — Q on Yanshee, laptop voice stack, Egyptian-Arabic speech-to-text

Last updated: 2026-11-02. Repo: `youssefhindawi-cube/yanshee-llm-model` (private). Work was done on an
HP Victus 16 laptop (Ubuntu, kernel 6.8, RTX 4060 Laptop 8 GB, 31 GB RAM, Python 3.10 user-site packages).

This file is the hand-over: what exists, what was measured, which data and models were used, what is still open.

---

## 1. What the project is

"Q" is Cube Consultants' company AI persona, meant to run on a UBTech Yanshee humanoid as a receptionist.
`q_yanshee.py` is a deliberately simple brain: keyword matching over the speech-ready answer table in
`Q_Brain_Brief_Yanshee.md` (public Part 1; internal Part 2 only with `--internal`). No LLM, no network,
answers capped at 60 words. The Markdown brief was produced by the user from `Q_Brain_Brief_Yanshee.pdf`.

The rest of the repo is laptop tooling built around it, and an investigation into Egyptian-dialect speech
recognition, which is the hard part of putting Q in front of Egyptian visitors.

## 2. Timeline of what was done

### 2026-10-29
- Read and tested `q_yanshee.py`. Self-test passes 11/12: "who founded the company" ties between the
  "What is Cube?" row (keyword `company`) and the founder row (keyword `founded`); the first row wins.
  Fix not applied yet: add the two-word keyword `who founded` to the founder row in the Markdown.
- Found two robot-side problems in the script (not fixed, needs the robot to test):
  - `listen()` calls `sync_do_voice_asr`, which in this Yanshee SDK is the Tuling chatbot endpoint, not
    dictation. The dictation functions are `sync_do_voice_iat` / `start_voice_iat` / `get_voice_iat` and the
    script never tries them.
  - On the robot the SDK import is `from YansheeRobot import YanAPI`; the script only tries `import YanAPI`
    (fine on a PC with `YanAPI.py` next to it; the SDK copy is in `../yanshee robot/yanshee/yanshee python library/YanAPI.py`).
  - Both Yanshee speech services are Chinese cloud services; English dictation is unverified.
- Laptop voice: installed espeak (robotic) then Piper neural TTS. Voices in `voices/`:
  `en_US-lessac-medium` (default, the user's choice), `en_US-ryan-high`, `en_GB-jenny_dioco-medium`,
  `ar_JO-kareem-medium` (Arabic, used to synthesise test clips). `--voice NAME` or `--voice espeak`.
- Laptop microphone for Q: Vosk offline recogniser (`stt/vosk-model-small-en-us-0.15` default,
  `stt/vosk-model-en-us-0.22-lgraph` larger). `--mic`, `--mic --ip X` (laptop mic + robot voice), `--stt NAME`.
  Piper-spoken English questions fed to Vosk route 9/9 correctly (after fixing "for cares" → "four cares",
  spoken years → digits, "he q" as wake phrase, and ignoring lone noise words).

### 2026-11-02
- Tested `MohamedAhmedAE/Whisper-large-V3-Arabic` (a LoRA adapter, see §5). Built `asr/test_arabic_whisper.py`.
- Built the live transcriber (`asr/live_arabic.py`, `asr/live_arabic_web.py`), see §4.
- Diagnosed the laptop microphone (§3) and the noise / wrong-words problems (§4).
- Researched the Open Universal Arabic ASR Leaderboard (§6), ran the Egyptian benchmark (§7),
  surveyed cloud services (§8), connected ElevenLabs (§8), checked the Claude API (§8).
- Created the git repo and pushed everything (§10).

## 3. The laptop microphone: what is true

- Raw capture of the internal digital mic (PulseAudio source `...hw_sofhdadsp_6__source`, ALSA `hw:1,6`
  and `hw:1,7`) gives a weak signal buried in 48–80 Hz hum. A loudspeaker-loopback test looked flat, which
  led to a wrong early conclusion that the mic was dead.
- Through Edge/Chrome `getUserMedia` with echo cancellation, noise suppression and auto gain on, the very
  same PulseAudio source yields usable speech. The browser and `arecord` open the same device; only the
  processing differs. This is why `live_arabic_web.py` captures audio in the browser.
- The headset-jack input (`Mic2`) is silent when nothing is plugged in. The HP webcam has no USB audio.
- Mic gain was raised from 32% to 100% and left there. No kernel fix was found for raw capture
  (closest public report: HP OMEN 16, same Ubuntu/kernel, unresolved).
- Browser auto-gain makes pauses almost as loud as speech, which breaks energy-based voice detection.
  Silero VAD handles it.

## 4. Live Egyptian-Arabic transcriber (what exists and why it looks like this)

Files: `asr/live_arabic.py` (terminal), `asr/live_arabic_web.py` (browser capture + RTL web page at
http://127.0.0.1:8765), `asr/denoise.py`. Shared parts: `Transcriber` (Whisper large-v3 + adapter),
`SpeechDetector` (Silero VAD, energy fallback), `Highpass` + `SpectralGate`.

Design decisions, each backed by a measurement on the user's own recordings:
- **Browser captures the mic** (see §3). The page streams 16 kHz PCM to the Python server; results stream back
  over server-sent events; Arabic renders right-to-left properly (terminal rendering of Arabic was poor).
- **Silero VAD** instead of energy thresholds: 512-sample frames, start at probability ≥ 0.6 (`--vad`),
  quiet below 0.35, phrase ends after 800 ms quiet (`--silence`) or 1.2 s without clear voice.
- **Shorter phrases**: soft cut at the first ≥250 ms pause after 8 s, firm cut after 12 s, hard cap 15 s.
  Long (19–28 s) noisy segments made Whisper drift and loop ("أقرأ لكوشو كمية" repeated).
- **Greedy final decoding**: beam search (4) looked better in theory but produced repetition loops on the
  user's real phrases; greedy was the most stable. `--beams-final` exists if needed.
- **Repeat collapsing** of Whisper loops, **blacklist** of Whisper's stock noise outputs
  ("اشتركوا في القناة", lone "شكرا", "موسيقى", ...), **confidence flag**: mean token log-prob below −0.8
  is shown grey with "؟" and "(unclear)". On real audio this flagged exactly the garbage lines.
- **High-pass 100 Hz + spectral gate** before Whisper: measured effect small (±1 WER point; helps at 0 dB
  SNR), kept on because it is free and helps the hum-laden terminal path.
- What the noise actually was: through the browser, the user's captures had ~30 dB between speech and quiet
  and almost nothing below 100 Hz. Silero rated 82% of a "quiet" capture as voice, and one 19 s segment was
  unintelligible to every system: speech-like background (people talking, or speaking away from the laptop).
  No filter fixes that; a headset or close microphone does.
- Verified: 6/6 phrases on a clean 40 s synthetic recording and on a noise+hum+auto-gain simulated copy, via
  both the terminal and the browser path; live Egyptian speech from the user came through
  ("أنت بتهم لي إيه؟", "هل انت شغال كويس؟").
- Debug aid: `--dump DIR` saves 10 s of ambient audio after Start and every phrase (raw and cleaned) as WAV.

Hand-over command: `python3 asr/live_arabic_web.py`, then open http://127.0.0.1:8765 and press Start.

## 5. Whisper and the "Egyptian" adapter

- `MohamedAhmedAE/Whisper-large-V3-Arabic` is not a model but a 923 MB LoRA adapter (rank 128, alpha 32,
  targets q/k/v/out/fc1/fc2) on `openai/whisper-large-v3`, trained for one epoch of 50 steps at batch 2 on
  `MightyStudent/Egyptian-ASR-MGB-3` (reported WER 38% on its own eval).
- On the RTX 4060 in fp16: 3.1 GB VRAM, 4.9 GB peak while merging, 0.2–0.6 s per short clip.
- Result: on 7 synthetic clips the adapter's output was byte-identical to the base model; on the real
  Egyptian test set (§7) it improves WER by 0.14 points. **Worthless in practice.**

## 6. Leaderboard research (Open Universal Arabic ASR Leaderboard, elmresearchcenter)

- The leaderboard has **no Egyptian column**. Test sets: SADA (Saudi broadcast), Common Voice 18 (read MSA),
  MASC clean/noisy (multi-dialect YouTube), MGB-2 (Al Jazeera), Casablanca (8 dialects incl. Egyptian).
  Casablanca is the only dialect-speech proxy. Results are hard-coded in the Space's `app.py`.
- Casablanca WER, Oct 2026: Sygma-ASR-Arabic 44.7 (a decoding recipe on Cohere's model), Cohere
  transcribe-arabic-07-2026 45.7, Jais-ASR 46.6, Audar-ASR-V1-Turbo 47.0; whisper-large-v3 71.8 (rank 25/39).
- The project's paper (arXiv 2412.13788, Dec 2024) has the only true per-Egyptian numbers (SADA Egyptian
  subset, older models): Nvidia Arabic Conformer-CTC 41.0, whisper-large-v2 48.4, whisper-large-v3 59.3.
- Model facts: Cohere 2.1B params bf16 (4.1 GB, gated with auto-approval, Apache-2.0); Audar Turbo 2.3B
  (licence "other"); Jais-ASR 5.7B (11.5 GB, would need 8-bit on an 8 GB GPU).

## 7. Egyptian-dialect benchmark (the main result)

Harness: `asr/eval_casablanca_egypt.py`. Scoring: the leaderboard's `normalize_arabic_text` (copied verbatim
in `asr/sygma/sygma_asr.py`: strip punctuation and diacritics, unify hamza/alef forms, Eastern→Western digits,
attach و) with word and character Levenshtein error rates (NeMo not installed; same arithmetic).

| System | WER | CER | Time for 1.02 h | Breakdowns |
|---|---|---|---|---|
| Cohere transcribe-arabic-07-2026 + Sygma decoding | **26.95%** | **11.99%** | 120 s | 0 |
| Cohere, plain 8-beam (top hypothesis) | 27.43% | 13.00% | 119 s | 1 |
| Cohere, greedy | 28.32% | 13.22% | 77 s | 8 |
| Whisper-large-v3 + Egyptian LoRA adapter | 45.91% | 21.94% | 557 s | 3 |
| Whisper-large-v3 | 46.05% | 22.37% | 535 s | 4 |
| ElevenLabs Scribe, first 3 clips only | 25.8% (3 clips) | 14.6% (3 clips) | 2 s | 0 |

Ten-clip comparison (2026-11-03; ElevenLabs sent 10 clips, ~40 s of audio, cached in
`asr/results/casablanca_egypt/cache_elevenlabs.jsonl`). Clip `22_1` is excluded from the 9-clip columns: its
reference is the label "[Music]" while all systems correctly transcribed the speech in it.

| System | WER, 9 real clips | CER, 9 real clips | WER, all 10 |
|---|---|---|---|
| Cohere + Sygma | 40.0% | 26.6% | 49.2% |
| Cohere greedy | 41.7% | 26.3% | 50.8% |
| ElevenLabs Scribe | 50.0% | 29.2% | 63.9% |
| Whisper-large-v3 | 81.7% | 47.4% | 90.2% |

These ten are harder than the set average (Cohere+Sygma 27% on all 846); the ordering Cohere > ElevenLabs >
Whisper held on both samples, but ranking ElevenLabs against Cohere with confidence needs the full hour.

Observations:
- Cohere cuts Whisper's word errors by 40% and character errors by nearly half, 4–7× faster.
- Sygma (8-best beam search with a per-word reward of β = 1.0, length penalty 0, plus a breakdown filter)
  adds ~1.4 WER points over greedy and removes decoding breakdowns entirely. Its authors tuned β on
  non-test Saudi audio only.
- Egyptian is the easier end of dialect speech: Whisper scores 46% here vs 72% on the leaderboard's
  Casablanca average (which includes Maghrebi dialects). Compare systems against each other, not against
  the leaderboard column.
- ElevenLabs on 3 clips is in Cohere's league (Cohere+Sygma scores 35.5 / 30.0 on those same 3). Three clips
  cannot rank them; the free plan's quota stopped a full run (1 hour of audio).
- Per-clip predictions for every system: `asr/results/casablanca_egypt/*.jsonl` (reference + hypothesis).

Running Cohere locally:
- Loaded through the vendored Sygma code (`asr/sygma/`, Apache-2.0; `cohere_asr/` is Cohere's own model code
  with a `GenerationMixin` fix). Pinned revision `c3e911b42149bf7a1e53d5cef9878aee87515a23`.
- Loaded in **bfloat16** (Sygma's float32 load needs 8.4 GB). Each variant must run in its **own process**:
  two loaded models exceed 8 GB.
- Needs: `librosa`, `numba>=0.61` (NumPy-2 build; the system numba 0.55 crashes under NumPy 2),
  `sentencepiece`, `coverage>=7` (numba imports `coverage.types`). All in `requirements.txt`.
- Access: accept the model's terms on Hugging Face while logged in (`huggingface-cli whoami` shows the account),
  then it downloads (4.1 GB). The download resumes poorly after an interruption (it restarted from zero once).

## 8. Cloud speech-to-text, ElevenLabs, Claude

- Services with an explicit Egyptian setting: Google Cloud Speech-to-Text (`ar-EG`), Azure AI Speech
  (`ar-EG`), Deepgram Nova-3 (claims ar-EG, vendor-reported). Arabic specialist: Munsit by CNTXT AI
  (claims ~24% WER across dialects; sales-channel access). General strong models without a dialect setting:
  OpenAI gpt-4o-transcribe, ElevenLabs Scribe. No independent Egyptian head-to-head exists.
- `asr/cloud_asr.py`: adapters for OpenAI, ElevenLabs, Deepgram, Azure, Google (plain HTTPS, no SDKs).
  Keys from environment variables or the git-ignored `.env`. The harness caches each clip's answer so a
  rerun never bills twice; a run where every clip fails is reported as failed, not scored.
- ElevenLabs was connected with `npx -y @elevenlabs/cli@latest onboard init / connect / check`: the key
  (ending 0352) lives in `.env`; never print it, never commit it. The user's plan is free with very limited
  credits: send at most a few clips. `asr/elevenlabs_transcribe.py FILE` prints the transcript of any audio
  file. The ElevenLabs Python SDK was deliberately not installed.
- Google: `gcloud` is authenticated as the user's Starpilot work account; using it would bill a work project,
  so it was not used. No other provider keys exist on the machine.
- **Claude API / Console have no speech-to-text**: verified against the current docs on 2026-11-02; the
  Messages API accepts text, images, PDF/plain-text documents and sandbox uploads only.

## 9. Datasets and models used

| Item | Source | Size / notes | Where it lives |
|---|---|---|---|
| Casablanca, Egypt config, **test** split | `UBC-NLP/Casablanca` on Hugging Face (not gated), **licence CC BY-NC-ND 4.0** (non-commercial, no derivatives — fine for evaluation, not for training or redistribution) | 846 clips, 1.02 h, mean 4.4 s, max 39 s, 44.1 kHz stereo WAV in parquet; fields seg_id, transcription, gender, duration | `~/.cache/huggingface/hub/datasets--UBC-NLP--Casablanca` (0.56 GB). A **validation** split of 846 clips exists and is unused |
| Synthetic test clips | Piper TTS (`ar_JO-kareem-medium`, `en_US-lessac-medium`) reading 6 Arabic + 1 English sentences about Cube | 1 MB, with reference text | `asr/clips/` (in repo) |
| User's real room recordings | captured through the browser during tests | ambient + 2 phrases + a 52 s stream | session scratchpad only (not kept) |
| Leaderboard test sets (SADA, CV18, MASC, MGB-2, Casablanca all dialects) | referenced for context only | — | not downloaded |
| `openai/whisper-large-v3` | Hugging Face, Apache-2.0 | 3.09 GB fp16 | HF cache |
| `MohamedAhmedAE/Whisper-large-V3-Arabic` | Hugging Face, Apache-2.0, LoRA on MGB-3 Egyptian | 0.92 GB | HF cache |
| `CohereLabs/cohere-transcribe-arabic-07-2026` | Hugging Face, gated (auto), Apache-2.0 | 4.1 GB bf16, 2.1B params, rev `c3e911b4…` | HF cache |
| Sygma ASR Arabic | github.com/abedbanna/Sygma-ASR-Arabic, Apache-2.0 | decoding code only | `asr/sygma/` (vendored) |
| Silero VAD | `silero-vad` 6.2.3 (pip) | bundled model | user site |
| Vosk models | alphacephei.com: small-en-us-0.15, en-us-0.22-lgraph | 68 MB, 205 MB | `stt/` (git-ignored) |
| Piper voices | rhasspy piper-voices | 4 × ~60–120 MB | `voices/` (git-ignored) |
| ElevenLabs Scribe | cloud, `scribe_v2` with `scribe_v1` fallback | 3 clips sent | — |

`scripts/download_models.sh` restores `voices/` and `stt/`; the Hugging Face models download on first use.

## 10. Environment facts that cost time (read before installing anything)

- All ML packages live in the Python 3.10 **user site** (`~/.local`), shared with lerobot 0.4.4, open3d,
  depthai and others. Pins that must hold: `huggingface_hub<0.36` (lerobot), `transformers<5` (4.57.6; v5
  broke on the system Pillow 9.0.1 and on huggingface_hub 2.x's HTTP decoder with the system brotli),
  `numpy==2.2.6` (several packages pin it), `torch 2.10.0+cu128` with matching `torchaudio 2.10.0`.
- Installed this way during the work: Pillow 11 (user site, shadows the old system one), torchaudio 2.10
  (the old 2.5.1 was already broken), numexpr/bottleneck NumPy-2 builds (silence the NumPy-1 warnings),
  numba 0.68 + llvmlite, librosa 0.11, sentencepiece, coverage 7, piper-tts 1.8, vosk 0.3.45, silero-vad 6.2.3.
- **Do not install DeepFilterNet** here: it silently downgraded numpy to 1.26 and packaging to 23.2 and could
  not import anyway (needs the removed `torchaudio.backend`). It was fully reverted. Rule: run
  `pip install --dry-run` first and refuse anything that touches numpy, torch, transformers or huggingface_hub.
- GPU memory: Whisper large-v3 fp16 3.1 GB (4.9 GB peak with adapter merge); Cohere bf16 4.2 GB; a process
  that merged the adapter keeps ~5 GB reserved. Never load two models in one process on this card.
- Git/GitHub: "cube repo" = the user's second GitHub account `youssefhindawi-cube`, SSH alias `github-cube`
  (key `~/.ssh/github_yanshee`), commit identity `youssefhindawi-cube <yousef.hendawe@cubeconsult.org>`.
  `gh` has both this account and the Nodogoro work account logged in (`gh auth switch`).

## 11. Open items / suggested next steps

1. Wire Cohere + Sygma into `live_arabic_web.py` as the local backend (`--model cohere`), keep Whisper as the
   alternative; compare on the user's own voice. Expect large gains on dialect speech.
2. Use a headset or close microphone for any serious live test; the built-in mic picks up room speech.
3. `q_yanshee.py`: add `who founded` to the founder row's keywords (self-test 12/12); fix `listen()` to use
   the Yanshee dictation functions; try `from YansheeRobot import YanAPI`; test English dictation on the robot.
4. Egyptian benchmark extensions, if budget allows: ElevenLabs on the full hour; Google (`ar-EG`) or Azure
   (`ar-EG`); Audar-ASR-V1-Turbo and Jais-ASR (8-bit) locally; the unused Casablanca validation split as a
   second test set; Sygma's β tuned on Egyptian validation data (keep test untouched).
5. The 50-step LoRA adapter should be dropped; if fine-tuning is ever wanted, start from Cohere's model,
   not Whisper, and use data whose licence permits derivatives (Casablanca's does not).
6. Keep `.env` out of git (already ignored) and rotate the ElevenLabs key if it is ever exposed.
