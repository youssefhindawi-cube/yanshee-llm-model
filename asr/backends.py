"""Speech-to-text backends with one interface, for the live transcriber and the Q brain:
    backend = make_backend("cohere", device="cuda")
    text = backend(audio_int16, "ar")      # 16 kHz mono int16 -> text; backend.last_conf: 0 = fine, below -0.8 = doubtful
Backends: whisper (large-v3 + Egyptian adapter), cohere (CohereLabs transcribe-arabic + Sygma decoding, local GPU),
munsit (cloud, Arabic specialist)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sygma"))
RATE = 16000


def load_cohere_bf16(device):
    """Sygma's loader with bfloat16 weights (its float32 load needs 8.4 GB; an 8 GB GPU cannot hold that)."""
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
    pp = json.load(open(f"{local}/config.json"))["preprocessor"]
    fe = CohereAsrFeatureExtractor(
        feature_size=pp["features"], sampling_rate=pp["sample_rate"],
        n_window_size=int(pp["window_size"] * pp["sample_rate"]), n_window_stride=int(pp["window_stride"] * pp["sample_rate"]),
        window=pp["window"], normalize=pp["normalize"], n_fft=pp["n_fft"], log=pp["log"],
        frame_splicing=pp["frame_splicing"], dither=pp["dither"], pad_to=pp["pad_to"], padding_value=pp["pad_value"])
    _maybe_load_preprocessor_buffers_from_checkpoint(feature_extractor=fe, model_dir=Path(local))
    processor = CohereAsrProcessor(feature_extractor=fe, tokenizer=CohereAsrTokenizer.from_pretrained(local))
    model = CohereAsrForConditionalGeneration.from_pretrained(local, torch_dtype=torch.bfloat16).to(device).eval()
    return processor, model


class CohereBackend:
    name = "Cohere transcribe-arabic + Sygma (local)"

    def __init__(self, device="cuda", beams=8, beta=1.0, use_filter=True):
        import sygma_asr
        sygma_asr._load_cohere = load_cohere_bf16
        self.asr = sygma_asr.SygmaASR(device=device, mode="benchmark", beams=beams, beta=beta, use_filter=use_filter)
        self.is_degenerate = sygma_asr.is_degenerate
        self.last_conf = 0.0

    def __call__(self, audio_int16, language="ar", beams=None):
        audio = np.ascontiguousarray(audio_int16.astype(np.float32) / 32768.0)
        text = self.asr.transcribe(audio_arrays=[audio], sample_rates=[RATE], batch_size=1)[0] or ""
        self.last_conf = -9.0 if self.is_degenerate(text) else 0.0
        return text.strip()


class MunsitBackend:
    name = "Munsit (cloud)"

    def __init__(self, **_):
        from cloud_asr import munsit_transcribe
        self.fn = munsit_transcribe
        self.last_conf = 0.0

    def __call__(self, audio_int16, language="ar", beams=None):
        self.last_conf = 0.0
        return (self.fn(audio_int16) or "").strip()


class WhisperBackend:
    name = "Whisper large-v3 + Egyptian adapter (local)"

    def __init__(self, device="cuda", adapter=True, beams=1):
        from live_arabic import Transcriber
        self.asr = Transcriber(device, adapter, beams)
        self.last_conf = 0.0

    def __call__(self, audio_int16, language="ar", beams=None):
        text = self.asr(audio_int16, language, beams=beams)
        self.last_conf = self.asr.last_conf
        return text


def make_backend(name, device="cuda", **kw):
    name = (name or "whisper").lower()
    if name == "cohere":
        return CohereBackend(device=device, **kw)
    if name == "munsit":
        return MunsitBackend()
    if name == "whisper":
        return WhisperBackend(device=device, **kw)
    raise ValueError("unknown backend %r (whisper, cohere, munsit)" % name)
