#!/usr/bin/env python3
"""Profile vLLM initialization overhead - without actual weight loading.

This tests how fast vLLM can initialize if we had instant weight loading.
"""
import os
import sys
import time
import gc
import subprocess as sp

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def nvidia_smi_free_gb():
    result = sp.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def test_dummy_load(model_name: str):
    """Load model with dummy weights (no I/O)."""
    from vllm import LLM, SamplingParams

    print(f"\nTesting: {model_name}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    config = {
        "dtype": "bfloat16",
        "max_model_len": 100000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "trust_remote_code": True,
        "load_format": "dummy",  # Don't actually load weights - random init
    }

    print("Loading with dummy weights...")
    t0 = time.time()
    llm = LLM(model=model_name, **config)
    load_time = time.time() - t0

    print(f"Total load time: {load_time:.2f}s (no weight I/O)")
    print(f"GPU free after load: {nvidia_smi_free_gb():.1f} GB")

    # Note: inference won't be meaningful with dummy weights
    print("(Skipping inference - dummy weights won't produce sensible output)")

    # Cleanup
    print("Cleaning up...")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"Shutdown warning: {e}")
    time.sleep(1)
    del llm
    gc.collect()
    time.sleep(2)
    print(f"GPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")

    return load_time


def test_tokenizer_overhead():
    """Test just tokenizer loading overhead."""
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    # Cold
    print("\n--- Tokenizer cold load ---")
    t0 = time.time()
    tok1 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    cold_time = time.time() - t0
    print(f"Cold: {cold_time:.2f}s")
    del tok1

    # Warm
    print("--- Tokenizer warm load ---")
    t0 = time.time()
    tok2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    warm_time = time.time() - t0
    print(f"Warm: {warm_time:.2f}s")
    del tok2

    return cold_time, warm_time


def main():
    print("=" * 70)
    print("VLLM OVERHEAD PROFILING")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    model = "Qwen/Qwen3-VL-32B-Instruct"

    # Test 1: Tokenizer overhead
    print("\n" + "=" * 70)
    print("TEST 1: TOKENIZER LOADING")
    print("=" * 70)
    tok_cold, tok_warm = test_tokenizer_overhead()

    # Test 2: Dummy weights - just vLLM overhead
    print("\n" + "=" * 70)
    print("TEST 2: VLLM WITH DUMMY WEIGHTS (no I/O)")
    print("=" * 70)
    dummy_time = test_dummy_load(model)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tokenizer cold:       {tok_cold:.2f}s")
    print(f"Tokenizer warm:       {tok_warm:.2f}s")
    print(f"vLLM dummy load:      {dummy_time:.2f}s (pure initialization overhead)")
    print("=" * 70)
    print("\nThis represents the MINIMUM possible load time even with instant weight loading.")
    print("To go faster, we need to keep vLLM engine running and just swap weights.")


if __name__ == '__main__':
    main()
