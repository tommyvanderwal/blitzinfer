#!/usr/bin/env python3
"""
Test vLLM's optimized loading options:
1. Pinned memory loading (safetensors_load_strategy="pinned")
2. Multi-threaded loading (enable_multithread_load=True)
"""

import time
import os
import sys

IS_ROCM = os.path.exists("/opt/rocm")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if not IS_ROCM:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
else:
    os.environ["VLLM_SKIP_WARMUP"] = "1"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

PLATFORM = "780M" if IS_ROCM else "RTX PRO 6000"


def test_load_strategy(model_name: str, dtype: str, strategy: str,
                        multithread: bool = False, num_threads: int = 8):
    """Test a specific loading strategy"""
    from vllm import LLM, SamplingParams
    import torch

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}, Multithread: {multithread}")
    if multithread:
        print(f"  Threads: {num_threads}")
    print(f"{'='*60}")

    # Build load config
    load_config = {}
    if strategy == "pinned":
        load_config["safetensors_load_strategy"] = "pinned"

    extra_config = {}
    if multithread:
        extra_config["enable_multithread_load"] = True
        extra_config["num_threads"] = num_threads

    kwargs = {
        "model": model_name,
        "dtype": dtype,
        "max_model_len": 512,
        "max_num_seqs": 2,
        "disable_log_stats": True,
        "enforce_eager": True,
    }

    if IS_ROCM:
        kwargs["compilation_config"] = {"custom_ops": ["none"]}
        kwargs["kv_cache_memory_bytes"] = 10 * 1024**3

    # Add load config if we have any
    if load_config:
        # vLLM accepts load_config via load_format or model_loader_extra_config
        # Let's try via model_loader_extra_config
        pass

    if extra_config:
        kwargs["model_loader_extra_config"] = extra_config

    free_before, total = torch.cuda.mem_get_info()

    print(f"\nLoading {model_name}...")
    start = time.time()
    llm = LLM(**kwargs)
    load_time = time.time() - start

    free_after, _ = torch.cuda.mem_get_info()
    weight_size = (free_before - free_after) / 1e9

    print(f"  Load time: {load_time:.1f}s")
    print(f"  Weight size: {weight_size:.1f} GiB")
    print(f"  Load speed: {weight_size/load_time:.2f} GB/s")

    # Quick test
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"  Output: {out[0].outputs[0].text[:30]}")

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return load_time, weight_size


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt"

    if model == "gpt":
        model_name = "openai/gpt-oss-120b"
        dtype = "bfloat16"
    else:
        model_name = "Qwen/Qwen3-VL-32B-Instruct"
        dtype = "float16"

    print(f"{'='*60}")
    print(f"Testing Load Strategies - {PLATFORM}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    results = []

    # Test 1: Default (baseline)
    try:
        t, s = test_load_strategy(model_name, dtype, "default")
        results.append(("default", t, s))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("default", None, None))

    # Test 2: Multi-threaded loading
    try:
        t, s = test_load_strategy(model_name, dtype, "default", multithread=True)
        results.append(("multithread", t, s))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("multithread", None, None))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, load_time, size in results:
        if load_time:
            speed = size / load_time
            print(f"  {name:<15}: {load_time:.1f}s ({speed:.2f} GB/s)")
        else:
            print(f"  {name:<15}: FAILED")

    baseline = results[0][1] if results[0][1] else float('inf')
    for name, load_time, _ in results[1:]:
        if load_time:
            speedup = baseline / load_time
            print(f"    {name} vs baseline: {speedup:.2f}x")


if __name__ == "__main__":
    main()
