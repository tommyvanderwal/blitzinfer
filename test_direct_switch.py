#!/usr/bin/env python3
"""
Direct Qwen → Mistral switch test without individual tests first.
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

# Do NOT apply Blitz patch - use standard vLLM loading to isolate the issue
from vllm import LLM, SamplingParams


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def full_cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def main():
    print("="*70)
    print("DIRECT SWITCH TEST (No Blitz Patch)")
    print("="*70)

    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    # Conservative config
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.20,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "kv_cache_memory_bytes": 1 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    full_cleanup()

    print(f"\nInitial memory: {get_mem():.1f} GB free")

    # Load Qwen
    print(f"\n--- Loading {qwen_model.split('/')[-1]} ---")
    t0 = time.perf_counter()
    llm = LLM(model=qwen_model, **config)
    print(f"Load time: {(time.perf_counter() - t0)*1000:.0f}ms")
    print(f"Memory: {get_mem():.1f} GB free")

    # Test Qwen
    params = SamplingParams(max_tokens=30, temperature=0.7)
    outputs = llm.generate(["Capital of France?"], params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(f"Qwen output: {text[:50]}...")

    # Cleanup Qwen
    print("\n--- Cleaning up Qwen ---")
    del llm
    full_cleanup()
    print(f"Memory after cleanup: {get_mem():.1f} GB free")

    # Load Mistral
    print(f"\n--- Loading {mistral_model.split('/')[-1]} ---")
    t0 = time.perf_counter()
    llm = LLM(model=mistral_model, **config)
    print(f"Load time: {(time.perf_counter() - t0)*1000:.0f}ms")
    print(f"Memory: {get_mem():.1f} GB free")

    # Test Mistral
    print("Testing Mistral inference...")
    torch.cuda.synchronize()
    outputs = llm.generate(["What is 2+2?"], params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(f"Mistral output: {text[:50]}...")

    # Cleanup
    del llm
    full_cleanup()

    print("\n[SUCCESS] Direct switch completed!")


if __name__ == '__main__':
    main()
