#!/usr/bin/env python3
"""Test Qwen3-VL-32B on 780M with VLLM_SKIP_WARMUP=1 and kv_cache_memory_bytes"""

import time
import os

# Ensure environment is set
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from vllm import LLM, SamplingParams

print("Loading Qwen3-VL-32B on 780M with VLLM_SKIP_WARMUP=1...")
start = time.time()

# Model is ~63GB, with 109GB total and 0.85 util = ~93GB available
# So ~30GB for KV cache - use 20GB = 20 * 1024**3 bytes
KV_CACHE_GB = 20
KV_CACHE_BYTES = KV_CACHE_GB * 1024 * 1024 * 1024

llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    dtype="float16",
    max_model_len=4096,  # Increase for more context
    gpu_memory_utilization=0.85,  # Higher util since we have 109GB
    max_num_seqs=4,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
    kv_cache_memory_bytes=KV_CACHE_BYTES,  # Skip profiling with explicit KV cache
)

load_time = time.time() - start
print(f"\nModel loaded in {load_time:.2f}s")

# Quick text-only test
print("\nRunning text test...")
outputs = llm.generate(["Hello, I am"], SamplingParams(max_tokens=20))
print(f"Output: {outputs[0].outputs[0].text}")
print("\nSuccess!")
