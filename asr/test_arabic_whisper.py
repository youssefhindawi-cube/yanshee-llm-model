"""Compare openai/whisper-large-v3 with and without the MohamedAhmedAE Arabic LoRA adapter on local 16 kHz clips.
usage: python3 test_arabic_whisper.py <clips_dir> [cuda|cpu]"""
import glob, io, os, sys, time
import numpy as np, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging
hf_logging.set_verbosity_error()

clips_dir = sys.argv[1]; device = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device == "cuda" else torch.float32
BASE, ADAPTER = "openai/whisper-large-v3", "MohamedAhmedAE/Whisper-large-V3-Arabic"
clips = sorted(glob.glob(os.path.join(clips_dir, "*.wav")))

def vram(tag):
    if device == "cuda":
        print("   [%s] VRAM allocated %.2f GB, peak %.2f GB" % (tag, torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9))

def transcribe_all(model, processor, label):
    out = {}
    for wav in clips:
        audio, sr = sf.read(wav, dtype="float32"); assert sr == 16000, wav
        lang = "en" if os.path.basename(wav).startswith("en") else "ar"
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype)
        t = time.time()
        with torch.inference_mode():
            ids = model.generate(input_features=feats, language=lang, task="transcribe", max_new_tokens=120)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        out[wav] = (text, time.time() - t)
    return out

t0 = time.time()
processor = WhisperProcessor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
print("base model loaded in %.1f s on %s (%s)" % (time.time() - t0, device, dtype)); vram("base loaded")
base_out = transcribe_all(model, processor, "base")
vram("after base inference")

from peft import PeftModel
t0 = time.time()
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
print("adapter applied + merged in %.1f s" % (time.time() - t0)); vram("adapter merged")
ft_out = transcribe_all(model, processor, "fine-tuned")
vram("after fine-tuned inference")

print("\n%-6s %-9s %s" % ("clip", "", "text"))
for wav in clips:
    name = os.path.splitext(os.path.basename(wav))[0]; ref = io.open(os.path.splitext(wav)[0] + ".txt", encoding="utf-8").read().strip()
    dur = len(sf.read(wav)[0]) / 16000.0
    print("-" * 100); print("%-6s %-9s %s   (%.1f s of audio)" % (name, "spoken:", ref, dur))
    print("%-6s %-9s %s   [%.2f s]" % ("", "base:", base_out[wav][0], base_out[wav][1]))
    print("%-6s %-9s %s   [%.2f s]" % ("", "adapter:", ft_out[wav][0], ft_out[wav][1]))
