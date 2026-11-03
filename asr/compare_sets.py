#!/usr/bin/env python3
"""Compare every system across two 30-clip Egyptian sets: Casablanca (first 30) vs fabhaus30.

  python3 asr/compare_sets.py            # prints one table, old vs new, plus per-clip wins
"""
import io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from eval_casablanca_egypt import score

SYSTEMS = [("Cohere beam8", "cohere-beam8"), ("Cohere + Sygma", "cohere-sygma"), ("Cohere greedy", "cohere-greedy"),
           ("Munsit", "cloud_munsit"), ("ElevenLabs v3", "cloud_elevenlabs-v3"), ("ElevenLabs v1", "cloud_elevenlabs-v1"),
           ("ElevenLabs v2", "cloud_elevenlabs"), ("Deepgram Nova-3", "cloud_deepgram"), ("Whisper-large-v3", "whisper"), ("Whisper + adapter", "whisper-adapter")]
SETS = [("Casablanca first 30", os.path.join(HERE, "results", "casablanca_egypt"), "_first30"), ("fabhaus30 (new)", os.path.join(HERE, "results", "fabhaus30"), "")]


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8")] if os.path.exists(path) else None


def rows_for(setdir, suffix, key, ids=None):
    for cand in (os.path.join(setdir, key + suffix + ".jsonl"), os.path.join(setdir, key + ".jsonl")):
        r = load(cand)
        if r:
            if ids: m = {x["seg_id"]: x for x in r}; r = [m[i] for i in ids if i in m]
            return r
    return None


def main():
    results = {}   # (system, set) -> (wer, cer, n)
    wins = {}
    for setname, setdir, suffix in SETS:
        base = rows_for(setdir, suffix, "cloud_elevenlabs-v3") or rows_for(setdir, suffix, "whisper")
        if not base:
            print("no results yet for", setname); continue
        ids = [r["seg_id"] for r in base]
        keep = [i for i, r in enumerate(base) if not r["text"].strip().startswith("[")]
        per = {}
        for name, key in SYSTEMS:
            r = rows_for(setdir, suffix, key, ids)
            if not r or len(r) < len(ids): continue
            w, c = score([r[i]["text"] for i in keep], [r[i]["pred_text"] for i in keep])
            results[(name, setname)] = (w, c, len(keep)); per[name] = r
        wins[setname] = {n: 0 for n in per}
        for i in keep:
            sc = {n: score([per[n][i]["text"]], [per[n][i]["pred_text"]])[0] for n in per}; m = min(sc.values())
            for n, v in sc.items():
                if v == m: wins[setname][n] += 1
    names = [n for n, _ in SYSTEMS if any((n, s) in results for s, _, _ in SETS)]
    print("%-20s | %-22s | %-22s | %s" % ("system", "Casablanca 30: WER CER", "fabhaus30: WER   CER", "change in WER"))
    def key(n):
        v = results.get((n, SETS[1][0])); return v[0] if v else 999
    for n in sorted(names, key=key):
        a = results.get((n, SETS[0][0])); b = results.get((n, SETS[1][0]))
        fa = "%6.1f %6.1f" % a[:2] if a else "     -      -"; fb = "%6.1f %6.1f" % b[:2] if b else "     -      -"
        ch = "%+.1f" % (b[0] - a[0]) if a and b else "-"
        print("%-20s | %-22s | %-22s | %s" % (n, fa, fb, ch))
    for setname in wins:
        print("\nper-clip wins on %s (ties shared): %s" % (setname, dict(sorted(wins[setname].items(), key=lambda kv: -kv[1]))))


if __name__ == "__main__":
    main()
