#!/usr/bin/env python3
"""
Test switching between Qwen and Mistral models.

These are different architectures, so we need full model reload.
This tests the Blitz-patched vLLM approach for fast cross-architecture switching.
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

# Apply Blitz patch BEFORE importing vLLM
import blitz_vllm_patch
blitz_vllm_patch.patch_vllm()

from vllm import LLM, SamplingParams


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_output(llm, prompt, expected_contains=None):
    """Test model output and optionally check for expected content."""
    params = SamplingParams(max_tokens=50, temperature=0.7)
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

    valid = len(text.strip()) > 0
    if expected_contains:
        valid = valid and any(e.lower() in text.lower() for e in expected_contains)

    return text, valid


def main():
    print("=" * 70)
    print("CROSS-ARCHITECTURE MODEL SWITCHING TEST")
    print("Qwen <-> Mistral")
    print("=" * 70)

    # Models to test
    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    # vLLM config (conservative for 780M iGPU)
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Warmup GPU
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_mem = get_mem()
    print(f"\nInitial GPU memory: {initial_mem:.1f} GB free")

    results = []

    # === Load Qwen ===
    print("\n" + "-" * 70)
    print(f"LOADING: {qwen_model}")
    print("-" * 70)

    t0 = time.perf_counter()
    llm = LLM(model=qwen_model, **config)
    qwen_load_time = (time.perf_counter() - t0) * 1000

    print(f"Load time: {qwen_load_time:.0f}ms")
    print(f"GPU memory: {get_mem():.1f} GB free")

    # Test Qwen
    text, valid = test_output(llm, "What is the capital of France?", ["paris"])
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:60]}...")
    results.append(("Qwen load", qwen_load_time, valid))

    # === Switch to Mistral ===
    print("\n" + "-" * 70)
    print(f"SWITCHING TO: {mistral_model}")
    print("-" * 70)

    t_switch = time.perf_counter()

    # Cleanup Qwen
    del llm
    gc.collect()
    cleanup_time, cleared = blitz_vllm_patch.fast_cleanup()
    print(f"Cleanup: {cleanup_time:.0f}ms (cleared {cleared} params)")
    print(f"GPU memory after cleanup: {get_mem():.1f} GB free")

    # Load Mistral
    t_load = time.perf_counter()
    llm = LLM(model=mistral_model, **config)
    mistral_load_time = (time.perf_counter() - t_load) * 1000

    qwen_to_mistral_time = (time.perf_counter() - t_switch) * 1000
    print(f"Mistral load: {mistral_load_time:.0f}ms")
    print(f"Total switch (Qwen→Mistral): {qwen_to_mistral_time:.0f}ms")

    # Test Mistral
    text, valid = test_output(llm, "What is 2 + 2?", ["4", "four"])
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:60]}...")
    results.append(("Qwen→Mistral switch", qwen_to_mistral_time, valid))

    # === Switch back to Qwen ===
    print("\n" + "-" * 70)
    print(f"SWITCHING BACK TO: {qwen_model}")
    print("-" * 70)

    t_switch = time.perf_counter()

    # Cleanup Mistral
    del llm
    gc.collect()
    cleanup_time, cleared = blitz_vllm_patch.fast_cleanup()
    print(f"Cleanup: {cleanup_time:.0f}ms (cleared {cleared} params)")
    print(f"GPU memory after cleanup: {get_mem():.1f} GB free")

    # Load Qwen again
    t_load = time.perf_counter()
    llm = LLM(model=qwen_model, **config)
    qwen_reload_time = (time.perf_counter() - t_load) * 1000

    mistral_to_qwen_time = (time.perf_counter() - t_switch) * 1000
    print(f"Qwen reload: {qwen_reload_time:.0f}ms")
    print(f"Total switch (Mistral→Qwen): {mistral_to_qwen_time:.0f}ms")

    # Test Qwen again
    text, valid = test_output(llm, "Count from 1 to 5:", ["1", "2", "3", "4", "5"])
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:60]}...")
    results.append(("Mistral→Qwen switch", mistral_to_qwen_time, valid))

    # === One more round trip ===
    print("\n" + "-" * 70)
    print("SECOND ROUND TRIP (warmed up)")
    print("-" * 70)

    # Qwen → Mistral (warmed)
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()

    t_switch = time.perf_counter()
    llm = LLM(model=mistral_model, **config)
    switch_time = (time.perf_counter() - t_switch) * 1000

    text, valid = test_output(llm, "Hello!", None)
    print(f"Qwen→Mistral (warm): {switch_time:.0f}ms [{['FAIL','OK'][valid]}]")
    results.append(("Qwen→Mistral (warm)", switch_time, valid))

    # Mistral → Qwen (warmed)
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()

    t_switch = time.perf_counter()
    llm = LLM(model=qwen_model, **config)
    switch_time = (time.perf_counter() - t_switch) * 1000

    text, valid = test_output(llm, "Hi there!", None)
    print(f"Mistral→Qwen (warm): {switch_time:.0f}ms [{['FAIL','OK'][valid]}]")
    results.append(("Mistral→Qwen (warm)", switch_time, valid))

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_valid = True
    for name, time_ms, valid in results:
        status = "OK" if valid else "FAIL"
        print(f"  {name}: {time_ms:.0f}ms [{status}]")
        all_valid = all_valid and valid

    # Calculate averages
    switch_times = [r[1] for r in results if "switch" in r[0].lower() or "→" in r[0]]
    avg_switch = sum(switch_times) / len(switch_times) if switch_times else 0

    print(f"\n  Average switch time: {avg_switch:.0f}ms ({avg_switch/1000:.1f}s)")
    print(f"  All outputs valid: {all_valid}")

    # Cleanup
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()

    print("\n" + "=" * 70)
    if all_valid:
        print("SUCCESS: Cross-architecture switching works!")
    else:
        print("WARNING: Some outputs were invalid")
    print("=" * 70)


if __name__ == '__main__':
    main()
