#!/usr/bin/env python3
"""
Benchmark different vLLM load formats to find the fastest option.

Formats to test:
1. Default (safetensors, lazy)
2. safetensors eager
3. runai_streamer
"""

import gc
import os
import sys
import time
import types

# ROCm setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    """Get GPU memory (used, free) in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def cleanup():
    """Full cleanup."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_format(model_name: str, load_format: str, extra_config: dict = None):
    """Benchmark a single load format."""
    from vllm import LLM

    cleanup()
    used, free = get_mem()
    print(f"  Before load: {used:.1f}GB used, {free:.1f}GB free")

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
        "load_format": load_format,
    }
    if extra_config:
        config.update(extra_config)

    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    load_time = (time.perf_counter() - t0) * 1000

    # Quick validation
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=10, temperature=0.7)
    outputs = llm.generate(["Hello"], params)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = len(text) > 0

    used, free = get_mem()
    print(f"  After load: {used:.1f}GB used")
    print(f"  Time: {load_time:.0f}ms")
    print(f"  Output: {text[:30]}... [{'OK' if valid else 'FAIL'}]")

    # Cleanup
    del llm
    cleanup()

    return load_time, valid


def main():
    print("="*70)
    print("VLLM LOAD FORMAT BENCHMARK")
    print("="*70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    cleanup()

    model = "mistralai/Mistral-7B-Instruct-v0.3"

    formats_to_test = [
        ("auto (lazy)", "auto", {}),
        ("safetensors eager", "safetensors", {"safetensors_load_strategy": "eager"}),
        ("runai_streamer", "runai_streamer", {}),
    ]

    results = []

    for name, load_format, extra in formats_to_test:
        print(f"\n{'-'*70}")
        print(f"Testing: {name}")
        print(f"{'-'*70}")

        try:
            load_time, valid = benchmark_format(model, load_format, extra)
            results.append((name, load_time, valid))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, None, False))

        cleanup()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    baseline = None
    for name, time_ms, valid in results:
        status = "OK" if valid else "FAIL"
        if time_ms:
            if baseline is None:
                baseline = time_ms
                print(f"  {name}: {time_ms:.0f}ms (baseline) [{status}]")
            else:
                diff = (baseline - time_ms) / baseline * 100
                faster = "faster" if diff > 0 else "slower"
                print(f"  {name}: {time_ms:.0f}ms ({abs(diff):.1f}% {faster}) [{status}]")
        else:
            print(f"  {name}: FAILED [{status}]")

    print("="*70)


if __name__ == '__main__':
    main()
