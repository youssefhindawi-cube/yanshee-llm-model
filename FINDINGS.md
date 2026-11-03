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

Thirty-clip comparison (2026-11-03; ElevenLabs Scribe v1, v2 and v3 each sent the first 30 clips, ~2 min of audio
per version, cached in `asr/results/casablanca_egypt/cache_elevenlabs*.jsonl`). Three clips whose reference is a
label rather than speech (`22_1` "[Music]", `37_3` "[screaming]", `42_4` "[noise]") are excluded from the
real-speech columns: every system transcribed the speech actually present in them. 27 real clips, 219 reference words.

| System | WER, 27 real clips | CER, 27 real clips | WER, all 30 |
|---|---|---|---|
| Cohere, plain 8-beam (local) | 33.0% | 13.9% | 36.2% |
| Cohere + Sygma (local) | 33.5% | 14.5% | 37.6% |
| Cohere greedy (local) | 34.9% | 14.6% | 38.1% |
| Munsit, CNTXT AI, model `munsit` (cloud, Arabic specialist, dialect auto-detected) | 37.2% | **12.9%** | 41.3% |
| ElevenLabs Scribe v3 (cloud, undocumented model id `scribe_v3`) | 43.3% | 18.5% | 48.6% |
| ElevenLabs Scribe v1 (cloud) | 46.0% | 19.5% | 50.9% |
| ElevenLabs Scribe v2 (cloud) | 47.0% | 19.7% | 51.8% |
| Deepgram Nova-3, `language=ar` (cloud) | 57.2% | 26.1% | 57.8% |
| Whisper-large-v3 (local) | 62.8% | 30.2% | 65.6% |
| Whisper + adapter (local) | 64.7% | 30.5% | 67.4% |

Per-clip wins (lowest WER, ties shared): Cohere + Sygma 19, Scribe v3 13, Scribe v1 10, Scribe v2 9, Whisper 4.
Deepgram (added 2026-11-03, 30 clips, $200 free credit, no card): 57.2% WER on the 27 real clips, 5 of 30 clips returned
empty, no punctuation, and it writes ه for ة (e.g. حاجه), which the leaderboard normalisation counts as errors; unifying
ة/ه only moves it a few points, so it remains below ElevenLabs and well below Cohere. Per-clip wins with Deepgram in the
mix: Cohere 21, Scribe v3 14, Deepgram 5, Whisper 4. Full-set run deliberately not done.

Munsit (added 2026-11-03, 30 clips, free plan 10,000 credits, no card, ~200 credits used): 37.2% WER / 12.9% CER on
the 27 real clips, second only to Cohere on WER and the best CER of every system. Its errors are mostly character-level
(split or merged words such as "اه بل", "كدهفضل"), never an empty output. Per-clip wins with Munsit in: Cohere 19,
Scribe v3 12, Munsit 11, Deepgram 4, Whisper 3. Adapter: `cloud:munsit` (x-api-key, `MUNSIT_API_KEY` in .env).
Clips with zero word errors: Cohere 4, Scribe v1 4, Scribe v2 4, Scribe v3 3, Whisper 1.

Reading: Cohere stays ~10 WER points ahead of the best ElevenLabs model; Scribe v3 is the best ElevenLabs version
(3–4 points better than v1/v2, which are indistinguishable); the three Cohere decodings are within noise on 27
clips (on all 846, Sygma leads). ElevenLabs' characteristic failure is inventing words on utterances under ~2 s.
`scribe_v3` is accepted by the API but absent from ElevenLabs' reference and changelog as of 2026-11-03: treat as a
preview. Harness providers: `cloud:elevenlabs` (v2), `cloud:elevenlabs-v3`, `cloud:elevenlabs-v1`.

### Second Egyptian set: fabhaus30 (2026-11-03), and the old-vs-new comparison

Motivation: a concern that Cohere might have been trained on Casablanca. Second set: 30 clips drawn (seed 42, 2–15 s,
>= 3 words) from the test split of `fabhaus/masri_audio_transcription` (Hugging Face, MIT, Nov 2024; natural Egyptian
conversation, human-written transcripts, provenance undocumented). Stored in `asr/clips_fabhaus30/` (3.8 MB, 119 s,
309 words) with `refs.jsonl`; run with `--dataset fabhaus30`; results in `asr/results/fabhaus30/`;
`asr/compare_sets.py` prints old vs new. Candidates rejected: `Yousef1258852/arabic-asr-egyptian-eval` (no audio),
`ehabnegm/masri-podcast-300h` (ASR-made transcripts, Levantine samples), `NightPrince/MasriSpeech-Test` (no text),
`ismaeeelxd/Egyptian-Arabic-Lectures` (30 s clips, 303/305 with English, Gemini-made transcripts). Note: the
inspection downloaded two full test splits (~550 MB) before the user asked to limit traffic; avoid that next time.

WER on 27 (Casablanca) / 30 (fabhaus30) real-speech clips. "Unified" additionally maps ى→ي and ة→ه, because Cohere
writes word-final ى on the new set ("فى", "اللى") and Deepgram writes ة as ه, which the leaderboard normalisation
counts as errors although they are spelling conventions.

| System | Casablanca 30, strict | Casablanca 30, unified | fabhaus30, strict | fabhaus30, unified |
|---|---|---|---|---|
| Munsit | 37.2 | 36.3 | **25.4** | **25.4** |
| Speechmatics (batch v2, `ar`, enhanced; free 240 min/month) | 40.9 | 37.7 | 29.2 | 26.4 |
| ElevenLabs v2 | 47.0 | 44.7 | 29.2 | 29.2 |
| ElevenLabs v1 | 46.0 | 43.7 | 29.5 | 29.5 |
| Cohere + Sygma | 33.5 | 31.2 | 35.9 | 30.5 |
| Cohere greedy | 34.9 | 32.6 | 36.9 | 31.5 |
| Cohere beam8 | **33.0** | **30.7** | 38.3 | 32.9 |
| Deepgram Nova-3 | 57.2 | 51.6 | 36.9 | 35.9 |
| Whisper-large-v3 | 62.8 | 61.4 | 36.3 | 36.3 |
| Whisper + adapter | 64.7 | 63.3 | 36.6 | 36.6 |
| ElevenLabs v3 | 43.3 | 40.5 | 45.1 | 45.1 |

Per-clip wins (strict): Casablanca — Cohere+Sygma 16, Cohere beam8 15, ElevenLabs v3 12, Munsit 9; fabhaus30 — Munsit 22,
ElevenLabs v2 21, ElevenLabs v1 19, Whisper 17, Cohere+Sygma 11.

Added 2026-11-03: **Speechmatics** (second-best cloud after Munsit on both sets; drops fillers and repeats and adds
punctuation, so verbatim WER understates it; CER 11.4 on fabhaus30). **Gemini** (`gemini-3.5-flash`, thinking off,
verbatim prompt; `gemini-2.5-flash` is retired for new keys and `gemini-3.8-flash` was overloaded): the AI Studio
free tier allows only **20 generate-content requests per day**, so 21 Casablanca clips were answered (18 real-speech:
36.4% WER / 17.3% CER, between Cohere and ElevenLabs on that set; first clip near-perfect) and the rest, 9 + 30 clips,
need two more days of quota or billing. Table rows are added once both sets are complete. Harness now retries clips on
transient network errors. A full-846 Speechmatics run was started by mistake and stopped after ~11 extra clips.

Reading:
- **Cohere's accuracy is the same on both sets** (unified 31.2 → 30.5). A model trained on Casablanca would show an
  advantage there that vanishes on unseen data; it does not. This is evidence against contamination, not proof.
- **Every other system improves massively on the new set** (Whisper 61→36, ElevenLabs 44→29, Deepgram 52→36, Munsit
  36→25): fabhaus30 is cleaner, calmer speech (interviews/podcasts) than Casablanca's noisy TV drama. Cohere's edge is
  robustness on hard, noisy dialect audio, which is the closer match to a receptionist robot in a lobby.
- **Munsit is the most consistent and the best on the clean set**; ElevenLabs v1/v2 are strong on clean speech; v3 got
  worse on the new set (45%), so it is not a safe upgrade. Whisper-large-v3 is competitive on clean speech only.
- Recommendation unchanged for the robot (noisy, far-field): Cohere locally, Munsit as the cloud alternative. For clean
  recorded audio, Munsit first.

### Final ranking across both 30-clip sets (2026-11-03)

Average over the two sets (Casablanca 27 real-speech clips, fabhaus30 30 clips). "Unified" maps ى→ي and ة→ه; "strict"
is the leaderboard's own normalisation. Sorted by average strict WER, the leaderboard convention:

| Rank | System | Avg WER strict | Avg WER unified | Avg CER | Casablanca unif. | fabhaus30 unif. | Coverage |
|---|---|---|---|---|---|---|---|
| 1 | Munsit (cloud, Arabic specialist) | **31.3** | 30.9 | **11.6** | 36.3 | 25.4 | full |
| 2 | Cohere + Sygma (local) | 34.7 | **30.8** | 13.7 | 31.2 | 30.5 | full |
| 3 | Speechmatics (cloud) | 35.0 | 32.1 | 14.5 | 37.7 | 26.4 | full |
| 4 | Cohere beam8 (local) | 35.7 | 31.8 | 15.2 | 30.7 | 32.9 | full |
| 5 | Cohere greedy (local) | 35.9 | 32.0 | 14.2 | 32.6 | 31.5 | full |
| 6 | ElevenLabs v1 (cloud) | 37.8 | 36.6 | 15.2 | 43.7 | 29.5 | full |
| 7 | ElevenLabs v2 (cloud) | 38.1 | 36.9 | 15.5 | 44.7 | 29.2 | full |
| 8 | ElevenLabs v3 (cloud, undocumented) | 44.2 | 42.8 | 21.0 | 40.5 | 45.1 | full |
| 9 | Deepgram Nova-3 (cloud) | 47.1 | 43.8 | 22.6 | 51.6 | 35.9 | full |
| 10 | Whisper-large-v3 (local) | 49.5 | 48.8 | 22.7 | 61.4 | 36.3 | full |
| 11 | Whisper + Egyptian LoRA adapter (local) | 50.6 | 49.9 | 22.9 | 63.3 | 36.6 | full |
| – | Gemini 3.5 Flash (cloud) | n/a | n/a | n/a | 34.9 (18 of 27 clips) | not run | **incomplete**: free tier is 20 requests/day |

Munsit and Cohere + Sygma are joint first: a tenth of a point apart on unified WER, Munsit ahead on strict WER (Cohere's
word-final ى costs it ~3 points) and on CER, Cohere ahead by 5 points on the noisy Casablanca clips, Munsit by 5 on the
clean fabhaus30 clips. Choice by conditions: **Cohere (local) for noisy, far-field use such as the robot in a lobby;
Munsit for clean audio or as the cloud option; Speechmatics next (largest free allowance, 240 min/month).** Everything
from ElevenLabs v3 down is 10+ points behind on at least one set. Gemini sat in the top group on the 18 clips it answered
and needs two more free-tier days (9 + 30 clips) to be ranked.

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
