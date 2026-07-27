"""Confirm generation actually runs on the GPU, not silently on CPU.

    python gpucheck.py [model-id]

torch falls back to CPU without erroring when its CUDA build does not match
the host driver, and everything still "works" -- just far slower on hardware
you are paying for. This reports where the weights live, how much VRAM they
take, and how fast a real batch generates.
"""

import subprocess
import sys
import time

import torch

from qwen3 import MODEL_ID, generate, load

model_id = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID

print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device         : {torch.cuda.get_device_name(0)}")
print(f"torch          : {torch.__version__}")

model, tok = load(model_id)
param = next(model.parameters())
print(f"\nmodel          : {model_id}")
print(f"weights on     : {param.device}   dtype {param.dtype}")

if torch.cuda.is_available():
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

start = time.time()
texts, lengths = generate(
    model, tok, ["List three colours."] * 16,
    max_new_tokens=64, think=False, progress=False,
)
elapsed = time.time() - start

if torch.cuda.is_available():
    print(f"peak VRAM      : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"16 x 64 tokens : {elapsed:.1f}s  ({sum(lengths) / elapsed:.0f} tok/s)")
print(f"sample         : {texts[0][:60]!r}")

print()
subprocess.run([
    "nvidia-smi",
    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
    "--format=csv",
], check=False)
