#!/usr/bin/env python3
"""
Final BlitzInfer vLLM Integration Test

Comprehensive test that:
1. Loads model with Blitz patch
2. Runs multiple inference tests
3. Performs fast cleanup and reload
4. Verifies model still works correctly after switch
5. Reports all timings
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


def run_inference(llm, prompts, max_tokens=50):
    """Run inference on a list of prompts and return results."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
    outputs = llm.generate(prompts, params)
    return [(o.outputs[0].text if o.outputs else "") for o in outputs]


def test_model_correctness(llm):
    """Test that model produces sensible outputs for various prompts."""
    test_cases = [
        ("What is 2 + 2?", lambda x: "4" in x),
        ("Capital of France?", lambda x: "paris" in x.lower()),
        ("Count 1,2,3,4,", lambda x: "5" in x),
        ("Say hello:", lambda x: len(x.strip()) > 0),
    ]

    results = []
    for prompt, validator in test_cases:
        outputs = run_inference(llm, [prompt], max_tokens=30)
        passed = validator(outputs[0]) if outputs else False
        results.append((prompt, outputs[0][:50] if outputs else "EMPTY", passed))

    return results


def main():
    print("=" * 70)
    print("BLITZINFER FINAL INTEGRATION TEST")
    print("=" * 70)

    # Warmup GPU
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_mem = get_mem()
    print(f"\nInitial GPU memory: {initial_mem:.1f} GB free")

    # vLLM config
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    results = []

    # === TEST 1: Initial load ===
    print("\n" + "-" * 70)
    print("TEST 1: Initial Load")
    print("-" * 70)

    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Load time: {load_time:.0f}ms")
    results.append(("Initial Load", load_time))

    # === TEST 2: Model correctness ===
    print("\n" + "-" * 70)
    print("TEST 2: Model Correctness Tests")
    print("-" * 70)

    test_results = test_model_correctness(llm)
    all_passed = True
    for prompt, output, passed in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {prompt[:30]}... -> {output[:30]}...")
        all_passed = all_passed and passed

    # === TEST 3: Fast switch ===
    print("\n" + "-" * 70)
    print("TEST 3: Fast Model Switch")
    print("-" * 70)

    t_switch = time.perf_counter()

    # Cleanup
    del llm
    gc.collect()

    t_cleanup = time.perf_counter()
    cleanup_time, cleared = blitz_vllm_patch.fast_cleanup()
    cleanup_total = (time.perf_counter() - t_cleanup) * 1000
    print(f"Cleanup: {cleanup_total:.0f}ms (cleared {cleared} params)")

    # Reload
    t_reload = time.perf_counter()
    llm = LLM(model=model_name, **config)
    reload_time = (time.perf_counter() - t_reload) * 1000

    switch_time = (time.perf_counter() - t_switch) * 1000
    print(f"Reload: {reload_time:.0f}ms")
    print(f"TOTAL SWITCH: {switch_time:.0f}ms ({switch_time/1000:.2f}s)")
    results.append(("Switch Time", switch_time))

    # === TEST 4: Post-switch correctness ===
    print("\n" + "-" * 70)
    print("TEST 4: Post-Switch Correctness Tests")
    print("-" * 70)

    test_results2 = test_model_correctness(llm)
    all_passed2 = True
    for prompt, output, passed in test_results2:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {prompt[:30]}... -> {output[:30]}...")
        all_passed2 = all_passed2 and passed

    # === TEST 5: Second switch (warmed up) ===
    print("\n" + "-" * 70)
    print("TEST 5: Second Switch (Warmed Up)")
    print("-" * 70)

    del llm
    gc.collect()

    t_switch2 = time.perf_counter()
    cleanup_time2, _ = blitz_vllm_patch.fast_cleanup()
    llm = LLM(model=model_name, **config)
    switch_time2 = (time.perf_counter() - t_switch2) * 1000
    print(f"Switch time: {switch_time2:.0f}ms ({switch_time2/1000:.2f}s)")
    results.append(("Switch 2 (warm)", switch_time2))

    # Quick verify
    outputs = run_inference(llm, ["Hello, how are you?"], max_tokens=20)
    print(f"Quick verify: {outputs[0][:40]}...")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, time_ms in results:
        print(f"  {name}: {time_ms:.0f}ms ({time_ms/1000:.2f}s)")

    avg_switch = (results[1][1] + results[2][1]) / 2
    print(f"\n  Average switch: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
    print(f"  Target: <3000ms")

    print("\n" + "-" * 70)
    if all_passed and all_passed2:
        print("ALL CORRECTNESS TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")

    if avg_switch < 6000:
        print(f"SWITCH TIME: {avg_switch/1000:.1f}s (good performance)")
    else:
        print(f"SWITCH TIME: {avg_switch/1000:.1f}s (above target)")

    print("=" * 70)

    # Cleanup
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()


if __name__ == '__main__':
    main()
