# -*- coding: utf-8 -*-
"""Q on Yanshee - answers ONLY from Q_Brain_Brief_Yanshee.md (no server, no AI model).

Works on old Python 3 (3.5+), no extra packages needed except what the Yanshee already has.

Usage (on the Yanshee, or on a PC with --ip <robot-ip>):
  python3 q_yanshee.py --text            # type questions on the keyboard (test the brain, no audio)
  python3 q_yanshee.py --selftest        # run built-in checks of the keyword matcher
  python3 q_yanshee.py --probe           # list the voice/TTS/ASR functions your YanAPI really has
  python3 q_yanshee.py                   # live: listen with the robot's ASR, answer with its voice
  python3 q_yanshee.py --wake            # live, but only answer after "Hey Q"
  python3 q_yanshee.py --internal        # also answer Part 2 (INTERNAL) - private tests only
  python3 q_yanshee.py --ip 192.168.x.x  # control a Yanshee over the network from a PC
  python3 q_yanshee.py --text --voice en_GB-jenny_dioco-medium   # laptop voice: any .onnx in ./voices, or "espeak"
  python3 q_yanshee.py --mic             # laptop only: listen with the laptop microphone (offline), speak with the laptop voice
  python3 q_yanshee.py --mic --ip 192.168.x.x   # listen with the laptop microphone, speak through the robot

Laptop voice: if the Piper package is installed (pip install piper-tts) and a voice model sits in ./voices,
answers are spoken with that natural voice; otherwise espeak is used; otherwise answers are only printed.
On the robot (or with --ip) the robot's own TTS is used and this is ignored.

Laptop microphone: --mic needs the Vosk package (pip install vosk) and an English model folder in ./stt
(default vosk-model-small-en-us-0.15, from alphacephei.com/vosk/models). Recognition runs offline on the laptop.
"""
from __future__ import print_function

import argparse
import io
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORDS = 60  # about 25-30 seconds of speech


# ---------------------------------------------------------------- knowledge (from the md)
AR = re.compile(r"[\u0600-\u06ff]")


def norm(text):
    """Make keywords and recognised speech comparable: lower-case, strip punctuation, drop an English plural 's',
    and for Arabic: remove diacritics and tatweel, unify hamza/alef forms, ة->ه, ى->ي, Eastern->Western digits,
    and strip a leading و/ف and the article ال from longer words ("والمكتب" -> "مكتب")."""
    text = text.lower()
    text = re.sub(r"[\u064b-\u0652\u0640]", "", text)
    text = re.sub("[أإآ]", "ا", text).replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    for e, w in zip("٠١٢٣٤٥٦٧٨٩", "0123456789"):
        text = text.replace(e, w)
    text = re.sub(r"[^a-z0-9\u0600-\u06ff ]+", " ", text)
    words = []
    for w in text.split():
        if AR.match(w):
            if len(w) > 3 and w[0] in "وف":
                w = w[1:]
            if len(w) > 4 and w.startswith("ال"):
                w = w[2:]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " " + " ".join(words) + " "


def trim_words(text, limit=MAX_WORDS):
    words = text.split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 40 else cut.rstrip(",;:") + "."


def load_brain(md_path, internal=False):
    """Parse the speech-ready table (public) and, optionally, the Part 2 bullets (internal)."""
    with io.open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rules, fallback = [], None
    in_table = False
    for line in lines:
        if line.startswith("### Speech-ready quick answers"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            in_table = False
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[0].lower() == "keywords" or set(cells[0]) <= set("-: "):
                continue
            keys, title, answer = cells[0], cells[1], cells[2]
            if keys.startswith("(anything else)"):
                fallback = answer
                continue
            kws = [norm(k) for k in keys.split(",") if k.strip()]
            rules.append({"title": title, "kws": kws, "answer": answer, "guard": title.startswith("Projects") or "المشاريع" in title})

    if internal:
        part2 = False
        for line in lines:
            if line.startswith("## PART 2"):
                part2 = True
                continue
            if part2 and line.startswith("## "):
                break
            m = re.match(r"- \*\*(.+?)(?: \((?:employee|admin)\))?:\*\*\s*(.+)", line)
            if part2 and m:
                label, body = m.group(1), m.group(2)
                body = re.sub(r"\*+", "", body)
                label_words = [w for w in re.split(r"[^a-z\u0600-\u06ff]+", label.lower()) if len(w) > 3 or (AR.match(w) and len(w) > 2)]
                rules.append({"title": "INTERNAL: " + label, "kws": [norm(w) for w in label_words],
                              "answer": body, "guard": False, "internal": True})
    if not fallback:
        fallback = "I don't have that information. I can tell you about our history, vision and offices, or find a colleague who can help."
    return rules, fallback


def kw_in(k, q):
    """Keyword k (normalised, space-padded) occurs in question q: as whole words, or, for keywords of 5+ letters,
    as the start of a word, so Arabic suffixes do not hide it ("هتتكلفك" still matches "هتتكلف")."""
    if k in q:
        return True
    stem = k.strip()
    return len(stem) >= 5 and " " not in stem and any(w.startswith(stem) for w in q.split())


def best_answer(question, rules, fallback):
    q = norm(question)
    # safety first: price / safety / project questions always go to the human-hand-off answer
    for r in rules:
        if r["guard"] and any(kw_in(k, q) for k in r["kws"] if len(k.strip()) > 3 or AR.search(k)):   # Arabic words are short
            return r["title"], trim_words(r["answer"])
    best, best_score = None, 0
    for r in rules:
        score = 0
        for k in r["kws"]:
            if kw_in(k, q):
                score += len(k.strip().split())  # longer phrases count more
        if score > best_score:
            best, best_score = r, score
    if best:
        return best["title"], trim_words(best["answer"])
    return "(fallback)", fallback


# ---------------------------------------------------------------- Yanshee voice I/O
def load_yanapi(ip=None):
    try:
        import YanAPI  # provided by the Yanshee SDK
    except ImportError:
        return None
    if ip and hasattr(YanAPI, "yan_api_init"):
        YanAPI.yan_api_init(ip)
    return YanAPI


# ---------------------------------------------------------------- laptop voice (not used on the robot)
VOICES_DIR = os.path.join(HERE, "voices")
DEFAULT_VOICE = "en_US-lessac-medium"
_piper = {}  # the loaded Piper voice, kept in memory so each answer starts in about 0.3 s


def load_pc_voice(name):
    """Load Piper voice ./voices/<name>.onnx, or an engine from asr/tts_backends.py ("voicetut:Hassan", "munsit:Fares").
    Returns the voice name, or None if it is missing (the reason is kept in _piper["why"])."""
    if name.split(":")[0] in ("voicetut", "munsit"):
        sys.path.insert(0, os.path.join(HERE, "asr"))
        try:
            from tts_backends import make_tts
            _piper["engine"] = make_tts(name)
            return _piper["engine"].name
        except Exception as e:
            _piper["why"] = "%s: %s" % (name, e)
            return None
    model = os.path.join(VOICES_DIR, name + ".onnx")
    if not os.path.exists(model):
        _piper["why"] = "no file " + model
        return None
    try:
        from piper import PiperVoice  # pip install piper-tts
    except ImportError:
        _piper["why"] = "Piper is not installed for this Python: pip install piper-tts"
        return None
    _piper["voice"] = PiperVoice.load(model)
    return name


def pc_speak(text):
    """Speak on the laptop: Piper if loaded, else espeak, else just the printed text."""
    voice, engine = _piper.get("voice"), _piper.get("engine")
    if voice is not None or engine is not None:
        import subprocess
        player = None
        try:
            if engine is not None:             # VoiceTut / Munsit: (sample_rate, int16 samples) per sentence
                chunks = ((pcm.tobytes(), sr) for sr, pcm in engine.synth(text))
            else:
                chunks = ((c.audio_int16_bytes, c.sample_rate) for c in voice.synthesize(text))
            for data, sr in chunks:            # sentence by sentence: playback starts while the rest renders
                if player is None:
                    player = subprocess.Popen(["aplay", "-q", "-r", str(sr), "-f", "S16_LE",
                                               "-c", "1", "-t", "raw", "-"], stdin=subprocess.PIPE)
                player.stdin.write(data)
            if player is not None:
                player.stdin.close()
                player.wait()
            return
        except (OSError, RuntimeError) as e:  # aplay missing, audio device busy, engine failed: fall through to espeak
            print("(voice playback failed: %s)" % e)
    if os.system("which espeak >/dev/null 2>&1") == 0:
        os.system('espeak "%s"' % text.replace('"', ""))


def speak(text, yan):
    print("Q:", text)
    if yan is not None:
        for name in ("sync_do_tts", "start_voice_tts"):
            fn = getattr(yan, name, None)
            if fn:
                try:
                    fn(text)
                    return
                except TypeError:
                    fn(tts=text)
                    return
    pc_speak(text)  # laptop without the SDK


def listen(yan):
    """Return recognised text from the robot's ASR. Names differ between SDK versions, so try the known ones."""
    if yan is None:
        raise RuntimeError("YanAPI is not available here. Use --text, or run this on the Yanshee / with --ip.")
    starters = [n for n in ("start_voice_asr", "start_voice_asr_offline") if hasattr(yan, n)]
    getters = [n for n in ("get_voice_asr_result", "get_voice_asr", "get_voice_asr_state") if hasattr(yan, n)]
    sync = getattr(yan, "sync_do_voice_asr", None) or getattr(yan, "sync_voice_asr", None)
    if sync:
        return extract_text(sync())
    if not starters or not getters:
        raise RuntimeError("No ASR function found in YanAPI. Run: python3 q_yanshee.py --probe  and send me the output.")
    getattr(yan, starters[0])()
    for _ in range(100):  # wait up to ~10 s
        time.sleep(0.1)
        text = extract_text(getattr(yan, getters[0])())
        if text:
            return text
    return ""


def extract_text(result):
    """YanAPI returns dicts like {'code':0,'data':{'text':'...'}}; cope with several shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "result", "asr", "content"):
            if result.get(key):
                return str(result[key]).strip()
        for v in result.values():
            t = extract_text(v)
            if t:
                return t
    return ""


def probe(yan):
    if yan is None:
        print("YanAPI not importable on this machine.")
        return
    names = [n for n in dir(yan) if any(w in n.lower() for w in ("voice", "tts", "asr", "speech", "audio"))]
    print("Voice-related functions in YanAPI:")
    for n in names:
        print("  ", n)
    print("\nSend this list to the developer if listening/speaking does not work.")


# ---------------------------------------------------------------- laptop microphone (not used on the robot)
STT_DIR = os.path.join(HERE, "stt")
DEFAULT_STT = "vosk-model-small-en-us-0.15"
MIC_RATE = 16000
MIC_TIMEOUT = 20  # seconds of nothing before we give up this round and listen again
SPOKEN_FIXES = (  # what the recogniser tends to write -> what the keywords expect
    ("twenty fifty", "2050"), ("two thousand fifty", "2050"), ("two thousand and fifty", "2050"),
    ("twenty thirty", "2030"), ("two thousand thirty", "2030"), ("two thousand and thirty", "2030"),
    ("for cares", "four cares"), ("4 cares", "four cares"), ("for care", "four cares"))
MIC_JUNK = {"huh", "the", "a", "uh", "um", "oh", "ah", "hmm", "mm", "yeah", "but", "and", "i", "it", "to"}  # noise-only "words"
_vosk = {}


def load_mic(model_name):
    """Load the Vosk offline recogniser from ./stt/<model_name>. Returns the name, or raises RuntimeError saying why not."""
    path = os.path.join(STT_DIR, model_name)
    if not os.path.isdir(path):
        raise RuntimeError("no model folder %s (download an English model from alphacephei.com/vosk/models and unzip it in ./stt)" % path)
    try:
        import vosk  # pip install vosk
    except ImportError:
        raise RuntimeError("Vosk is not installed for this Python: pip install vosk")
    vosk.SetLogLevel(-1)
    _vosk["model"] = vosk.Model(path)
    _vosk["Recognizer"] = vosk.KaldiRecognizer
    return model_name


def mic_listen(timeout=MIC_TIMEOUT):
    """Record from the default microphone until a phrase ends; return its text, or '' if nothing was said in time."""
    import json
    import subprocess
    rec = _vosk["Recognizer"](_vosk["model"], MIC_RATE)
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1", "-t", "raw", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # opened fresh so Q never hears itself
    print("Listening...")
    text, deadline = "", time.time() + timeout
    try:
        while time.time() < deadline:
            data = proc.stdout.read(4000)  # 0.125 s of audio
            if not data:
                break
            if rec.AcceptWaveform(data):  # True when Vosk hears the end of a phrase
                text = json.loads(rec.Result()).get("text", "")
                if text and text not in MIC_JUNK:
                    break
                text = ""  # a lone noise word: keep listening
    finally:
        proc.kill()
        proc.wait()
    text = " " + text + " "
    for spoken, wanted in SPOKEN_FIXES:  # "cairo twenty fifty" -> "cairo 2050", "for cares" -> "four cares"
        text = text.replace(" " + spoken + " ", " " + wanted + " ")
    return text.strip()


# ---------------------------------------------------------------- tests and main
def selftest(rules, fallback):
    arabic = any(AR.search(k) for r in rules for k in r["kws"])
    cases = [
        ("ازيك يا كيو", "Greeting"),
        ("انت مين", "Who is Q?"),
        ("مين اللي اسس كيوب", "Who founded Cube?"),
        ("القاهرة ٢٠٥٠ دي ايه", "What is Cairo 2050?"),
        ("احكيلي عن الاهتمامات الاربعة", "The Four Cares"),
        ("عندكم كام استوديو", "How many studios?"),
        ("مكتبكم فين", "Where are you?"),
        ("رؤيتكم ايه للمستقبل", "Vision"),
        ("المشروع ده هيتكلف كام", "Projects / prices / safety"),
        ("المبنى ده آمن", "Projects / prices / safety"),
        ("شكرا يا كيو مع السلامة", "Closing"),
        ("الجو عامل ايه النهارده", "(fallback)"),
    ] if arabic else [
        ("hello there", "Greeting"),
        ("who are you", "Who is Q?"),
        ("what is cairo 2050", "What is Cairo 2050?"),
        ("tell me about the four cares", "The Four Cares"),
        ("who founded the company", "Who founded Cube?"),
        ("how many studios do you have", "How many studios?"),
        ("where is your office", "Where are you?"),
        ("what is your vision", "Vision"),
        ("how much does a project cost", "Projects / prices / safety"),
        ("is this building safe", "Projects / prices / safety"),
        ("thank you goodbye", "Closing"),
        ("what is the weather", "(fallback)"),
    ]
    bad = 0
    for q, want in cases:
        got, ans = best_answer(q, rules, fallback)
        got = got.split(" · ")[0]
        ok = got == want and len(ans.split()) <= MAX_WORDS
        bad += 0 if ok else 1
        print("%-4s %-38s -> %-28s (%d words)" % ("OK" if ok else "FAIL", q, got, len(ans.split())))
    print("\n%d/%d passed" % (len(cases) - bad, len(cases)))
    return bad == 0


# "Hey Q" / "يا كيو", including what the recognisers make of it ("ياكي و", "يا كيوب", "ياكيو", "ياكويو", "ياتيو", "يا قيو"),
# and a bare "كيو" that is its own word (not "كيوب"). Works on the raw text, before norm(). After "يا": one consonant
# (ك, ق or ت), up to four vowel letters, an optional ب, and then no further Arabic letter ("يا كابتن", "يا كمان" fail).
AR_LETTER = "[\u0621-\u063a\u0641-\u064a]"          # Arabic letters only (not ، ؟ or diacritics)
WAKE = re.compile(r"\b(?:hey|hay|he|hi|hello|ok|okay)[ ,.!?-]*(?:q|queue|cue|kew)\b"
                  r"|(?:يا|هاي|اهلا|أهلا|ya)\s*[كقت]\s*(?:[يىو]{1,4}ب?|ه)?(?!%s)"
                  r"|(?<!%s)كيو(?!%s)" % (AR_LETTER, AR_LETTER, AR_LETTER), re.I)
WAKE_REPLY = {"ar": "أيوه، اتفضل.", "en": "Yes?"}


def wake_split(text):
    """Was Q addressed? -> (True, the rest of the sentence) or (False, text). "يا كيو" alone gives (True, "")."""
    m = WAKE.search(text)
    if not m:
        return False, text
    rest = (text[:m.start()] + " " + text[m.end():]).strip(" ,.!?-،؟")
    return True, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(HERE, "Q_Brain_Brief_Yanshee.md"))
    ap.add_argument("--text", action="store_true", help="keyboard mode, no audio")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--wake", action="store_true", help="only answer after 'Hey Q'")
    ap.add_argument("--internal", action="store_true", help="also answer Part 2 (private tests only)")
    ap.add_argument("--ip", default=None, help="Yanshee IP when running from a PC")
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help="laptop voice: name of a .onnx in ./voices (default %(default)s), 'espeak', "
                         "or an Egyptian engine: voicetut[:Speaker] (local), munsit[:Voice] (cloud)")
    ap.add_argument("--mic", action="store_true", help="listen with the laptop microphone (offline Vosk)")
    ap.add_argument("--stt", default=DEFAULT_STT, help="Vosk model folder inside ./stt (default %(default)s)")
    a = ap.parse_args()

    if not os.path.exists(a.md):
        sys.exit("Cannot find the md file: %s  (copy Q_Brain_Brief_Yanshee.md next to this script)" % a.md)
    rules, fallback = load_brain(a.md, internal=a.internal)
    arabic = any(AR.search(k) for r in rules for k in r["kws"])
    print("Loaded %d answers from %s%s%s" % (len(rules), os.path.basename(a.md), " (+internal)" if a.internal else "", " [Arabic]" if arabic else ""))
    if arabic and a.voice == DEFAULT_VOICE and os.path.exists(os.path.join(VOICES_DIR, "ar_JO-kareem-medium.onnx")):
        a.voice = "ar_JO-kareem-medium"

    if a.selftest:
        sys.exit(0 if selftest(rules, fallback) else 1)

    yan = None if a.text or (a.mic and not a.ip) else load_yanapi(a.ip)
    if a.probe:
        probe(load_yanapi(a.ip))
        return

    if yan is None:
        loaded = None if a.voice == "espeak" else load_pc_voice(a.voice)
        why = "chosen with --voice espeak" if a.voice == "espeak" else _piper.get("why", "")
        print("Laptop voice:", ("" if _piper.get("engine") else "Piper ") + loaded if loaded else "espeak (%s)" % why)
    if a.mic:
        try:
            print("Laptop microphone: Vosk", load_mic(a.stt))
        except RuntimeError as e:
            sys.exit("Cannot use --mic: %s" % e)

    speak("كيو جاهز. اسألني عن كيوب للاستشارات." if arabic else "Q is online. Ask me about Cube Consultants.", yan)
    while True:
        try:
            heard = (input("You> ") if a.text else mic_listen() if a.mic else listen(yan)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not heard:
            continue
        if not a.text:
            print("Heard:", heard)
        addressed, rest = wake_split(heard)
        if a.wake and not addressed:
            continue                       # not addressed to Q
        if addressed:                      # remove "Hey Q" so it is not mistaken for a greeting
            if not rest:
                speak(WAKE_REPLY["ar" if re.search("[\u0600-\u06ff]", fallback) else "en"], yan)
                continue
            heard = rest
        title, answer = best_answer(heard, rules, fallback)
        if a.text:
            print("[matched: %s]" % title)
        speak(answer, yan)


if __name__ == "__main__":
    main()
