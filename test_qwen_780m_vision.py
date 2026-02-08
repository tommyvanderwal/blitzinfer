#!/usr/bin/env python3
"""Test Qwen3-VL-32B vision on 780M"""

import time
import os
from PIL import Image

# Ensure environment is set
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from vllm import LLM, SamplingParams

print("Loading Qwen3-VL-32B on 780M for vision test...")
start = time.time()

KV_CACHE_GB = 20
KV_CACHE_BYTES = KV_CACHE_GB * 1024 * 1024 * 1024

llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    max_num_seqs=4,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
    kv_cache_memory_bytes=KV_CACHE_BYTES,
)

load_time = time.time() - start
print(f"\nModel loaded in {load_time:.2f}s")

# Create a simple test image with text
print("\nCreating test image...")
img = Image.new('RGB', (400, 200), color='white')
from PIL import ImageDraw
draw = ImageDraw.Draw(img)
draw.text((50, 50), "Invoice #12345", fill='black')
draw.text((50, 80), "Total: $1,234.56", fill='black')
draw.text((50, 110), "Date: Jan 26, 2026", fill='black')
img.save("/tmp/test_invoice.png")
print("Saved test image to /tmp/test_invoice.png")

# Test vision
print("\nRunning vision test...")

import base64
with open("/tmp/test_invoice.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            {"type": "text", "text": "What is the total amount on this invoice? Answer briefly."},
        ]
    }
]

# Use chat completions
start = time.time()
outputs = llm.chat(messages, SamplingParams(max_tokens=100))
vision_time = time.time() - start

print(f"\nVision response ({vision_time:.2f}s):")
print(outputs[0].outputs[0].text)
print("\nSuccess!")
