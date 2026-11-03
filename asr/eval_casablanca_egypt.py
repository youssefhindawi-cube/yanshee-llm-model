#!/usr/bin/env python3
"""Compare ASR systems on the Casablanca Egyptian test set (UBC-NLP/Casablanca, config Egypt, split test: 846 clips, 1.02 h).

  python3 asr/eval_casablanca_egypt.py --systems whisper whisper-adapter
  python3 asr/eval_casablanca_egypt.py --systems cohere-greedy cohere-beam8 cohere-sygma
  python3 asr/eval_casablanca_egypt.py --systems whisper --limit 50        # quick look

Systems
  whisper          openai/whisper-large-v3, greedy, language forced to Arabic
  whisper-adapter  the same with the MohamedAhmedAE Egyptian LoRA adapter merged in
  cohere-greedy    CohereLabs/cohere-transcribe-arabic-07-2026, greedy (gated: accept its terms on Hugging Face first)
  cohere-beam8     the same, plain 8-beam search, top hypothesis
  cohere-sygma     the same, Sygma decoding: 8-beam search + per-word reward + breakdown filter (asr/sygma/)
  cloud:<name>     a cloud API: cloud:openai, cloud:elevenlabs, cloud:deepgram, cloud:azure, cloud:google
                   (keys from environment variables, see asr/cloud_asr.py; results are cached per clip so reruns do not re-bill)

Scoring uses the Open Universal Arabic ASR Leaderboard normalisation (its eval.py, copied verbatim inside
asr/sygma/sygma_asr.py) with word and character error rates. Predictions are saved under asr/results/.
"""
import argparse
import gc
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sygma"))
from sygma_asr import is_degenerate, normalize_arabic_text  # noqa: E402  (leaderboard normalisation, verbatim)

RATE = 16000
OUT = os.path.join(HERE, "results", "casablanca_egypt")   # overridden per --dataset in main()
DATASETS = {"casablanca": "the Casablanca Egypt test split (846 clips, downloaded from Hugging Face)",
            "fabhaus30": "30 clips from fabhaus/masri_audio_transcription (test split), stored in asr/clips_fabhaus30/"}


def load_local_clips(folder, limit=0):
    """A local clip set: <folder>/refs.jsonl with seg_id/text/duration, and <folder>/<seg_id>.wav at 16 kHz mono."""
    rows = []
    for line in io.open(os.path.join(folder, "refs.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows.append({"seg_id": r["seg_id"], "text": r["text"], "duration": r["duration"], "bytes": open(os.path.join(folder, r["seg_id"] + ".wav"), "rb").read()})
        if limit and len(rows) >= limit:
            break
    return rows


def load_egypt_test(limit=0):
    from huggingface_hub import snapshot_download
    d = snapshot_download("UBC-NLP/Casablanca", repo_type="dataset", allow_patterns=["Egypt/test-*"])
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "Egypt", "test-*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            rows.append({"seg_id": r["seg_id"], "text": r["transcription"], "duration": r["duration"], "bytes": r["audio"]["bytes"]})
            if limit and len(rows) >= limit:
                return rows
    return rows


def decode(row):
    x, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != RATE:
        x = resample_poly(x, RATE, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def score(refs, hyps):
    we = wn = ce = cn = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize_arabic_text(ref), normalize_arabic_text(hyp or "")
        we += edit_distance(r.split(), h.split())
        wn += len(r.split())
        ce += edit_distance(list(r), list(h))
        cn += len(r)
    return 100.0 * we / max(1, wn), 100.0 * ce / max(1, cn)


def run_whisper(rows, adapter):
    from live_arabic import Transcriber
    asr = Transcriber("cuda", adapter, 1)
    out = []
    for i, r in enumerate(rows):
        out.append(asr((decode(r) * 32767).astype(np.int16), "ar"))
        if (i + 1) % 100 == 0:
            print("   %d/%d" % (i + 1, len(rows)), flush=True)
    return out


def _load_cohere_bf16(device):
    """Sygma's loader, but bfloat16 weights (its float32 load needs 8.4 GB; an 8 GB GPU cannot hold that)."""
    import json as _json
    from pathlib import Path
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    import sygma_asr
    local = snapshot_download(sygma_asr.BASE_MODEL, revision=sygma_asr.BASE_REVISION)
    from cohere_asr.modeling_cohere_asr import CohereAsrForConditionalGeneration
    from cohere_asr.processing_cohere_asr import (CohereAsrFeatureExtractor, CohereAsrProcessor,
                                                  _maybe_load_preprocessor_buffers_from_checkpoint)
    from cohere_asr.tokenization_cohere_asr import CohereAsrTokenizer
    transformers.CohereAsrFeatureExtractor = CohereAsrFeatureExtractor
    transformers.CohereAsrTokenizer = CohereAsrTokenizer
    pp = _json.load(open(f"{local}/config.json"))["preprocessor"]
    fe = CohereAsrFeatureExtractor(
        feature_size=pp["features"], sampling_rate=pp["sample_rate"],
        n_window_size=int(pp["window_size"] * pp["sample_rate"]), n_window_stride=int(pp["window_stride"] * pp["sample_rate"]),
        window=pp["window"], normalize=pp["normalize"], n_fft=pp["n_fft"], log=pp["log"],
        frame_splicing=pp["frame_splicing"], dither=pp["dither"], pad_to=pp["pad_to"], padding_value=pp["pad_value"])
    _maybe_load_preprocessor_buffers_from_checkpoint(feature_extractor=fe, model_dir=Path(local))
    processor = CohereAsrProcessor(feature_extractor=fe, tokenizer=CohereAsrTokenizer.from_pretrained(local))
    model = CohereAsrForConditionalGeneration.from_pretrained(local, torch_dtype=torch.bfloat16).to(device).eval()
    return processor, model


COHERE_MODES = {
    "cohere-greedy": dict(beams=1, beta=0.0, use_filter=False, batch=16),
    "cohere-beam8": dict(beams=8, beta=0.0, use_filter=False, batch=4),
    "cohere-sygma": dict(beams=8, beta=1.0, use_filter=True, batch=4),
}


def run_cohere(rows, system, batch=None):
    import sygma_asr
    cfg = dict(COHERE_MODES[system])
    batch = batch or cfg.pop("batch")
    cfg.pop("batch", None)
    sygma_asr._load_cohere = _load_cohere_bf16
    asr = sygma_asr.SygmaASR(device="cuda", mode="benchmark", **cfg)
    audios = [decode(r) for r in rows]
    out = []
    for s in range(0, len(audios), 100):   # chunks of 100 so progress is visible and one failure does not lose everything
        out += asr.transcribe(audio_arrays=audios[s:s + 100], sample_rates=[RATE] * len(audios[s:s + 100]), batch_size=batch)
        print("   %d/%d" % (len(out), len(audios)), flush=True)
    return out


def run_cloud(rows, provider, workers=4):
    workers = int(os.environ.get("CLOUD_WORKERS", workers))   # free tiers allow few parallel requests: CLOUD_WORKERS=1
    """Calls the provider for every clip, 4 at a time, caching each answer on disk so a rerun costs nothing."""
    import concurrent.futures as cf
    from cloud_asr import PROVIDERS
    fn = PROVIDERS[provider]
    cache_path = os.path.join(OUT, "cache_%s.jsonl" % provider)
    cache = {}
    if os.path.exists(cache_path):
        for line in io.open(cache_path, encoding="utf-8"):
            d = json.loads(line)
            cache[d["seg_id"]] = d["pred_text"]
    todo = [r for r in rows if r["seg_id"] not in cache]
    print("   %d cached, %d to send" % (len(rows) - len(todo), len(todo)), flush=True)
    errors = 0
    def robust(audio):
        """Retry a clip on transient trouble (network drops, 429/5xx, time-outs) before giving up on it."""
        for attempt in range(4):
            try:
                return fn(audio)
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in ("HTTP 429", "HTTP 5", "timed out", "Network is unreachable", "EOF occurred",
                                                   "Remote end closed", "Connection reset", "Temporary failure", "urlopen error"))
                if attempt == 3 or not transient:
                    raise
                time.sleep(5 * (attempt + 1) ** 2)       # 5, 20, 45 s
    with io.open(cache_path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(robust, (decode(r) * 32767).astype(np.int16)): r for r in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                text = fut.result()
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print("   error on %s: %s" % (r["seg_id"], str(e)[:200]), flush=True)
                if errors >= 10 and errors > len(todo) // 2:
                    raise RuntimeError("too many errors from %s, stopping (last: %s)" % (provider, str(e)[:120]))
                continue
            cache[r["seg_id"]] = text
            f.write(json.dumps({"seg_id": r["seg_id"], "pred_text": text}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print("   %d/%d" % (i, len(todo)), flush=True)
    if todo and errors == len(todo):
        raise RuntimeError("no clip succeeded with %s (nothing was billed); fix the error above and rerun" % provider)
    return [cache.get(r["seg_id"], "") for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["whisper"],
                    help="any of: whisper whisper-adapter %s cloud:openai cloud:elevenlabs cloud:deepgram cloud:azure cloud:google" % " ".join(COHERE_MODES))
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--batch", type=int, default=0, help="override the Cohere batch size")
    ap.add_argument("--workers", type=int, default=0, help="parallel requests for cloud providers (default 4; use 1-2 on free tiers)")
    ap.add_argument("--dataset", default="casablanca", choices=list(DATASETS), help="which clip set to score (results go to asr/results/<dataset>/)")
    a = ap.parse_args()
    import torch
    if a.workers:
        os.environ["CLOUD_WORKERS"] = str(a.workers)
    global OUT
    if a.dataset == "casablanca":
        rows = load_egypt_test(a.limit)
    else:
        OUT = os.path.join(HERE, "results", a.dataset)
        rows = load_local_clips(os.path.join(HERE, "clips_" + a.dataset), a.limit)
    hours = sum(r["duration"] for r in rows) / 3600
    print("%s: %d clips, %.2f h" % (DATASETS[a.dataset], len(rows), hours), flush=True)
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for system in a.systems:
        print("\n== %s ==" % system, flush=True)
        t = time.time()
        try:
            if system.startswith("whisper"):
                hyps = run_whisper(rows, system == "whisper-adapter")
            elif system.startswith("cloud:"):
                hyps = run_cloud(rows, system.split(":", 1)[1])
            else:
                hyps = run_cohere(rows, system, a.batch or None)
        except Exception as e:
            print("   FAILED: %s" % str(e).splitlines()[0][:200], flush=True)
            gc.collect(); torch.cuda.empty_cache()
            continue
        dt = time.time() - t
        wer, cer = score([r["text"] for r in rows], hyps)
        degen = sum(is_degenerate(h or "") for h in hyps)
        summary[system] = {"WER": round(wer, 2), "CER": round(cer, 2), "clips": len(rows), "hours": round(hours, 3),
                           "seconds": round(dt, 1), "rtf": round(dt / (hours * 3600), 3), "breakdowns": degen}
        suffix = "" if not a.limit else "_first%d" % a.limit
        with io.open(os.path.join(OUT, system.replace(":", "_") + suffix + ".jsonl"), "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"seg_id": r["seg_id"], "text": r["text"], "pred_text": h}, ensure_ascii=False) + "\n")
        print("   WER %.2f%%  CER %.2f%%  | %.0f s (RTF %.3f) | breakdowns %d" % (wer, cer, dt, dt / (hours * 3600), degen), flush=True)
        for r, h in list(zip(rows, hyps))[:3]:
            print("   ref: %s\n   hyp: %s" % (r["text"], h), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    if summary:
        with io.open(os.path.join(OUT, "summary%s.json" % ("" if not a.limit else "_first%d" % a.limit)), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "results": summary}, ensure_ascii=False) + "\n")
        print("\n%-16s %7s %7s %8s %6s %s" % ("system", "WER%", "CER%", "time(s)", "RTF", "breakdowns"))
        for k, v in sorted(summary.items(), key=lambda kv: kv[1]["WER"]):
            print("%-16s %7.2f %7.2f %8.0f %6.3f %d" % (k, v["WER"], v["CER"], v["seconds"], v["rtf"], v["breakdowns"]))


if __name__ == "__main__":
    main()
