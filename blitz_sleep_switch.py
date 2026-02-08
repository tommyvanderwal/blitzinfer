#!/usr/bin/env python3
"""
BlitzSwitch using vLLM sleep mode - offload weights to CPU between models

This approach uses vLLM's built-in sleep mode to:
1. Load first model with enable_sleep_mode=True
2. When switching, call sleep() to offload weights to CPU
3. Load new model
4. Old model's GPU memory is freed via sleep offload

This allows in-process model switching without subprocess overhead.
"""

import time
import os
import gc
import sys

# Pre-configure environment
IS_ROCM = os.path.exists("/opt/rocm")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if not IS_ROCM:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
else:
    os.environ["VLLM_SKIP_WARMUP"] = "1"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

print("Pre-importing vLLM modules...")
t0 = time.time()
from vllm import LLM, SamplingParams
import torch
print(f"Imports ready in {time.time()-t0:.1f}s")

# Model configurations
MODELS = {
    "gpt": {
        "name": "openai/gpt-oss-120b",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "kv_cache_bytes": 10 * 1024**3 if IS_ROCM else None,
    },
    "qwen": {
        "name": "Qwen/Qwen3-VL-32B-Instruct",
        "dtype": "float16",
        "max_model_len": 512,
        "kv_cache_bytes": 10 * 1024**3 if IS_ROCM else None,
    }
}


def get_gpu_memory():
    """Get GPU memory info"""
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def load_model(model_key: str, enable_sleep: bool = False) -> tuple:
    """Load a model and return (llm, load_time)"""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    config = MODELS[model_key]
    print(f"\nLoading {config['name']}...")
    free, total = get_gpu_memory()
    print(f"  GPU memory before: {free:.1f}/{total:.1f} GiB free")

    start = time.time()

    kwargs = {
        "model": config["name"],
        "dtype": config["dtype"],
        "max_model_len": config["max_model_len"],
        "max_num_seqs": 2,
        "disable_log_stats": True,
        "enforce_eager": True,
        "enable_sleep_mode": enable_sleep,  # Enable sleep mode for weight offloading
    }

    if IS_ROCM:
        kwargs["compilation_config"] = {"custom_ops": ["none"]}
        if config["kv_cache_bytes"]:
            kwargs["kv_cache_memory_bytes"] = config["kv_cache_bytes"]

    llm = LLM(**kwargs)
    load_time = time.time() - start

    free, total = get_gpu_memory()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  GPU memory after: {free:.1f}/{total:.1f} GiB free")

    return llm, load_time


def unload_with_sleep(llm):
    """Unload model using sleep mode"""
    print("\nUnloading model via sleep mode...")
    start = time.time()

    free_before, total = get_gpu_memory()

    # Get the worker and call sleep
    try:
        executor = llm.llm_engine.engine_core.executor
        executor.collective_rpc("sleep", args=(2,))  # level=2 for full offload
    except Exception as e:
        print(f"  Sleep mode error: {e}")
        return 0

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()

    free_after, _ = get_gpu_memory()
    elapsed = time.time() - start
    freed = free_after - free_before

    print(f"  Sleep completed in {elapsed:.1f}s")
    print(f"  Freed {freed:.1f} GiB GPU memory")
    print(f"  GPU memory now: {free_after:.1f}/{total:.1f} GiB free")

    return elapsed


def test_sleep_switch():
    """Test switching models using sleep mode"""
    print("="*60)
    print(f"Sleep Mode Switch Test - {'780M' if IS_ROCM else 'RTX PRO 6000'}")
    print("="*60)

    # Load first model with sleep mode enabled
    llm, t1 = load_model("gpt", enable_sleep=True)

    # Quick test
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"GPT output: {out[0].outputs[0].text[:40]}")

    # Try sleep mode to offload
    unload_with_sleep(llm)

    # Now try loading second model
    # Note: This may or may not work depending on vLLM internals
    print("\n--- Attempting to load second model ---")
    free, total = get_gpu_memory()
    print(f"GPU memory available: {free:.1f}/{total:.1f} GiB")

    try:
        llm2, t2 = load_model("qwen", enable_sleep=True)
        out = llm2.generate(["Hello"], SamplingParams(max_tokens=10))
        print(f"Qwen output: {out[0].outputs[0].text[:40]}")
        print(f"\nSUCCESS: Sleep mode switching works!")
        print(f"Total switch time: {t2:.1f}s (vs ~35s with subprocess)")
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("Sleep mode doesn't support creating new LLM instances.")
        print("Need to use subprocess isolation for model switching.")


if __name__ == "__main__":
    test_sleep_switch()
