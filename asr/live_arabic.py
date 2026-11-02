#!/usr/bin/env python3
"""Live Arabic speech-to-text in the terminal: Whisper large-v3 + the Egyptian-dialect LoRA adapter.

Standalone test tool. It has nothing to do with q_yanshee.py or the Q brain file.

  python3 asr/live_arabic.py                    # listen on the default microphone, print text as you speak
  python3 asr/live_arabic.py --list-sources     # show the microphones PulseAudio knows about
  python3 asr/live_arabic.py --source <name>    # use a specific microphone (name from --list-sources)
  python3 asr/live_arabic.py --no-adapter       # plain whisper-large-v3, to compare against the adapter
  python3 asr/live_arabic.py --language auto    # let Whisper guess the language (default forces Arabic)
  python3 asr/live_arabic.py --log out.txt      # also append every final line to a file
  python3 asr/live_arabic.py --wav file.wav     # feed a recording through the pipeline instead of the mic

For a browser display with proper Arabic rendering, and to use the browser's own microphone processing,
run asr/live_arabic_web.py instead.

How it works: audio is captured continuously at 16 kHz. Silero VAD (a small neural voice detector that
ignores hum, noise and gain changes) decides when you start and stop talking. While you talk, the growing
phrase is re-transcribed every --partial-every seconds and shown dimmed on one line; when you pause for
--silence ms it is transcribed once more and printed as a final line. Whisper is not a streaming model, so
this is near-real-time (about half a second behind), not word-by-word. Without the silero-vad package an
energy detector is used instead (--threshold only matters then).
"""
import argparse
import collections
import io
import os
import queue
import subprocess
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise import Highpass, SpectralGate  # noqa: E402

RATE = 16000
FRAME = 512                        # samples per frame = what Silero VAD expects at 16 kHz (32 ms)
FRAME_MS = FRAME * 1000 // RATE
MAX_PHRASE_S = 15                  # hard cap: long segments make Whisper drift and loop
SOFT_CUT_S = 8                     # after this, end the phrase at the first ~250 ms pause
FIRM_CUT_S = 12                    # after this, end it at the first ~100 ms without voice
MIN_VOICED_MS = 350                # a phrase needs this much detected speech, or it is treated as noise
BASE = "openai/whisper-large-v3"
ADAPTER = "MohamedAhmedAE/Whisper-large-V3-Arabic"
DIM, RESET, CLEAR = "\x1b[2m", "\x1b[0m", "\x1b[K"


def list_sources():
    out = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True).stdout
    default = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True).stdout.strip()
    print("Microphones (PulseAudio sources), * = default:")
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and ".monitor" not in parts[1]:
            print("  %s %s" % ("*" if parts[1] == default else " ", parts[1]))
    print("Pass --source <name> to use one that is not the default.")


class MicReader(threading.Thread):
    """Reads 16 kHz mono frames from the default (or chosen) PulseAudio microphone into a queue."""
    def __init__(self, source=None):
        super().__init__(daemon=True)
        env = dict(os.environ)
        if source:
            env["PULSE_SOURCE"] = source
        self.proc = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.q = queue.Queue()

    def run(self):
        while True:
            data = self.proc.stdout.read(FRAME * 2)
            if len(data) < FRAME * 2:
                self.q.put(None)
                return
            self.q.put(np.frombuffer(data, dtype=np.int16))

    def close(self):
        self.proc.kill()


class WavReader(threading.Thread):
    """Feeds a recording into the pipeline at real-time pace, for testing without a microphone."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        ok = path.lower().endswith(".wav")
        if ok:
            with wave.open(path) as w:
                ok = w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        if not ok:  # anything else: let ffmpeg convert it
            tmp = os.path.join(os.path.dirname(os.path.abspath(path)), ".live_arabic_16k.wav")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(RATE), tmp], check=True)
            path = tmp
        with wave.open(path) as w:
            self.audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    def run(self):
        t0 = time.time()
        for i in range(0, len(self.audio) - FRAME + 1, FRAME):
            self.q.put(self.audio[i:i + FRAME])
            time.sleep(max(0.0, t0 + (i + FRAME) / RATE - time.time()))
        self.q.put(None)

    def close(self):
        pass


HALLUCINATIONS = ("اشتركوا في القناة", "اشترك في القناة", "لا تنسوا الاشتراك في القناة", "شكرا للمشاهدة", "شكرا لكم",
                  "شكرا", "موسيقى", "ترجمة نانسي قنقر", "نانسي قنقر", "سبحان الله", "اشتركوا بالقناة")


def is_hallucination(text):
    """Whisper's stock outputs on noise or near-silence (learnt from subtitled videos)."""
    import re
    t = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    return t in HALLUCINATIONS or (len(t.split()) <= 5 and any(h in t for h in HALLUCINATIONS if len(h.split()) >= 2))


def collapse_repeats(text, max_n=8):
    """Remove Whisper's repetition loops: an n-gram repeated 3+ times in a row (or 2+ times for n >= 3) is kept once."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, min(max_n, len(words) // 2) + 1):
            i = 0
            while i + 2 * n <= len(words):
                k = 1
                while i + (k + 1) * n <= len(words) and words[i:i + n] == words[i + k * n:i + (k + 1) * n]:
                    k += 1
                if k >= 3 or (k >= 2 and n >= 3):
                    del words[i + n:i + k * n]
                    changed = True
                i += 1
    return " ".join(words)


class Transcriber:
    def __init__(self, device, use_adapter, beams):
        self.last_conf = 0.0
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging
        hf_logging.set_verbosity_error()
        self.torch, self.device, self.beams = torch, device, beams
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        t = time.time()
        self.processor = WhisperProcessor.from_pretrained(BASE)
        model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=self.dtype, low_cpu_mem_usage=True).to(device).eval()
        if use_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
        self.model = model
        print("model ready in %.1f s on %s: whisper-large-v3%s" % (time.time() - t, device, " + Egyptian adapter" if use_adapter else ""))

    def __call__(self, audio_int16, language, beams=None):
        """Returns the text; self.last_conf = mean log-probability of the chosen tokens (0 = sure, below -0.8 = doubtful)."""
        audio = audio_int16.astype(np.float32) / 32768.0
        feats = self.processor(audio, sampling_rate=RATE, return_tensors="pt").input_features.to(self.device, self.dtype)
        n_beams = beams or self.beams
        kw = dict(task="transcribe", max_new_tokens=180, num_beams=n_beams, return_dict_in_generate=True, output_scores=True)
        if language != "auto":
            kw["language"] = language
        with self.torch.inference_mode():
            out = self.model.generate(input_features=feats, **kw)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        try:
            if n_beams > 1 and getattr(out, "sequences_scores", None) is not None:
                self.last_conf = float(out.sequences_scores[0])
            else:
                lps = [float(self.torch.log_softmax(s[0].float(), -1).max()) for s in out.scores]
                self.last_conf = float(np.mean(lps)) if lps else 0.0
        except Exception:
            self.last_conf = 0.0
        return collapse_repeats(text)


class NoiseFloor:
    """Energy threshold that follows the room: drops fast when it is quiet, creeps up slowly otherwise."""
    def __init__(self, fixed=0.0):
        self.fixed, self.floor, self.n = fixed, None, 0

    def update(self, level, talking=False):
        self.n += 1
        if self.fixed:
            return
        if self.floor is None:
            self.floor = level
        elif level < self.floor:
            self.floor += 0.2 * (level - self.floor)
        elif not talking and (level < self.thr or self.n < 10):
            self.floor += 0.02 * (level - self.floor)

    @property
    def thr(self):
        return self.fixed or max(3.0 * (self.floor or 0.0), 60.0)

    @property
    def ready(self):
        return bool(self.fixed) or self.n >= 30


def level_of(frame):
    """Loudness of a frame after a pre-emphasis filter, so mains hum and fan rumble count for little."""
    x = frame.astype(np.float32)
    x = x[1:] - 0.95 * x[:-1]
    return float(np.sqrt(np.mean(x * x) + 1e-9))


class SpeechDetector:
    """Silero VAD when the silero-vad package is installed, else the energy detector.
    update(frame) -> (speech, quiet, meter): speech = this frame contains voice, quiet = clearly no voice,
    meter = 0-100 for display (speech probability, or level relative to threshold)."""
    def __init__(self, fixed_threshold=0.0, start=0.6):
        self.vad, self.nf, self.start = None, None, start
        try:
            import torch
            from silero_vad import load_silero_vad
            self.torch = torch
            self.vad = load_silero_vad()
            self.vad(torch.zeros(FRAME), RATE)   # warm up the JIT so the first real frame is fast
        except Exception:
            self.nf = NoiseFloor(fixed_threshold)

    @property
    def name(self):
        return "Silero VAD" if self.vad is not None else "energy detector (pip install silero-vad for a better one)"

    @property
    def ready(self):
        return self.vad is not None or self.nf.ready

    def update(self, frame, talking=False):
        if self.vad is not None:
            p = self.vad(self.torch.from_numpy(np.array(frame, dtype=np.float32) / 32768.0), RATE).item()
            return p >= self.start, p < 0.35, p * 100.0
        level = level_of(frame)
        self.nf.update(level, talking)
        thr = self.nf.thr
        return (self.nf.ready and level > thr), level < thr, min(100.0, level / thr * 50.0)

    def reset(self):
        """Call between phrases."""
        if self.vad is not None:
            self.vad.reset_states()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="PulseAudio source name (see --list-sources); default = system default mic")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--wav", help="feed this recording instead of the microphone")
    ap.add_argument("--no-adapter", action="store_true", help="plain whisper-large-v3 without the Egyptian adapter")
    ap.add_argument("--language", default="ar", help="ar (default), en, ... or auto")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--partial-every", type=float, default=1.5, help="seconds between live updates while you talk")
    ap.add_argument("--silence", type=int, default=800, help="ms of quiet that ends a phrase")
    ap.add_argument("--threshold", type=float, default=0, help="energy threshold for the fallback detector (0 = automatic)")
    ap.add_argument("--beams", type=int, default=1, help="beam size for the live partial updates (1 = fastest)")
    ap.add_argument("--beams-final", type=int, default=1, help="beam size for the final line (1 = greedy; beam search tends to loop on long noisy phrases)")
    ap.add_argument("--vad", type=float, default=0.6, help="voice probability needed to start a phrase; raise towards 0.8 in a noisy room")
    ap.add_argument("--min-conf", type=float, default=-0.8, help="finals whose mean token log-prob is below this are marked (unclear)")
    ap.add_argument("--highpass", type=float, default=100, help="high-pass cutoff in Hz against hum and rumble (0 = off)")
    ap.add_argument("--no-denoise", action="store_true", help="skip the spectral noise gate before Whisper")
    ap.add_argument("--log", help="append final lines to this file")
    ap.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = until Ctrl-C)")
    a = ap.parse_args()
    if a.list_sources:
        list_sources()
        return

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and a.partial_every < 4:
        print("No GPU: live updates slowed to every 4 s (final lines are unaffected)")
        a.partial_every = 4
    asr = Transcriber(device, not a.no_adapter, a.beams)
    det = SpeechDetector(a.threshold, a.vad)
    hp = Highpass(a.highpass) if a.highpass > 0 else None
    clean = (lambda x: x) if a.no_denoise else SpectralGate()
    print("noise handling: %s%s" % ("high-pass %.0f Hz" % a.highpass if hp else "no high-pass", ", spectral gate" if not a.no_denoise else ", no gate"))
    reader = WavReader(a.wav) if a.wav else MicReader(a.source)
    reader.start()
    log = io.open(a.log, "a", encoding="utf-8") if a.log else None

    pre = collections.deque(maxlen=300 // FRAME_MS)   # ~300 ms kept from before speech was detected
    talking, buf, above, below, last_partial, voiced = False, [], 0, 0, 0.0, 0
    status_t, t_begin = 0.0, time.time()
    print("Listening with the %s. Speak Egyptian Arabic. Ctrl-C to stop." % det.name)
    try:
        while True:
            if a.duration and time.time() - t_begin > a.duration:
                break
            frame = reader.q.get()
            if frame is None:
                break
            if hp is not None:
                frame = hp(frame)
            speech, quiet, meter = det.update(frame, talking)
            if not det.ready:
                continue
            if not talking:
                pre.append(frame)
                above = above + 1 if speech else 0
                if above >= 3:                       # ~100 ms of voice: a phrase starts
                    talking, buf, below, unsure, last_partial, voiced = True, list(pre), 0, 0, 0.0, above
                elif time.time() - status_t > 0.15:
                    status_t = time.time()
                    sys.stdout.write("\r%s[voice %3.0f%% %-16s] listening...%s%s" % (DIM, meter, "|" * int(meter / 6), CLEAR, RESET))
                    sys.stdout.flush()
                continue
            buf.append(frame)
            voiced += 1 if speech else 0
            below = below + 1 if quiet else 0              # frames clearly without voice
            unsure = unsure + 1 if not speech else 0        # frames without clear voice (noise can sit here)
            dur = len(buf) * FRAME_MS / 1000.0
            ended = (below * FRAME_MS >= a.silence or unsure * FRAME_MS >= 1.5 * a.silence or dur >= MAX_PHRASE_S
                     or (dur >= SOFT_CUT_S and below * FRAME_MS >= 250) or (dur >= FIRM_CUT_S and unsure * FRAME_MS >= 100))
            real = voiced * FRAME_MS >= MIN_VOICED_MS
            if real and not ended and dur >= 1.0 and dur - last_partial >= a.partial_every:
                last_partial = dur
                text = asr(clean(np.concatenate(buf)), a.language)
                sys.stdout.write("\r%s... %s%s%s" % (DIM, text, CLEAR, RESET))
                sys.stdout.flush()
            if ended:
                talking, above = False, 0
                det.reset()
                keep = len(buf) - max(0, below - 200 // FRAME_MS)   # drop the pause, keep ~200 ms so the last word is not clipped
                audio = np.concatenate(buf[:keep])
                if real and len(audio) / RATE >= 0.4:
                    t = time.time()
                    text = asr(clean(audio), a.language, beams=a.beams_final)
                    dt = time.time() - t
                    stamp = time.strftime("%H:%M:%S")
                    tag = "" if asr.last_conf >= a.min_conf else "  (unclear, conf %.2f)" % asr.last_conf
                    if is_hallucination(text):
                        sys.stdout.write("\r%s%s(dropped, noise: %s)%s\n" % (CLEAR, DIM, text, RESET))
                        sys.stdout.flush()
                        buf = []
                        continue
                    sys.stdout.write("\r%s[%s] %s   %s(%.1f s audio, %.2f s)%s%s\n" % (CLEAR, stamp, text, DIM, len(audio) / RATE, dt, tag, RESET))
                    sys.stdout.flush()
                    if log:
                        log.write("[%s] %s\n" % (stamp, text))
                        log.flush()
                buf = []
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\r" + CLEAR + "\n")
        reader.close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
