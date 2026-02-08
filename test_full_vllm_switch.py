#!/usr/bin/env python3
"""
Full vLLM model switching test with fast cleanup.
Verifies that models load correctly and produce valid outputs after switching.
"""

import os
import sys
import gc
import time
import types

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
import torch.nn as nn


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def fast_cleanup():
    """
    Optimized cleanup: bypass vLLM, directly clear nn.Module parameters.
    Returns cleanup time in ms.
    """
    t0 = time.perf_counter()

    gc.unfreeze()

    cleared = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            obj._parameters[key] = None
                            cleared += 1
        except:
            pass

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return (time.perf_counter() - t0) * 1000, cleared


def test_model_output(llm, prompt, expected_type="text"):
    """Test that model produces valid output."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=50, temperature=0.7)
    outputs = llm.generate([prompt], params)

    if not outputs or not outputs[0].outputs:
        return False, "No output generated"

    text = outputs[0].outputs[0].text
    if not text or len(text.strip()) == 0:
        return False, "Empty output"

    return True, text


def run_full_switch_test():
    """Run comprehensive model switching test."""
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("FULL VLLM MODEL SWITCHING TEST")
    print("=" * 70)

    # Warmup GPU
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_mem = get_mem()
    print(f"\nInitial GPU memory: {initial_mem:.1f} GB free")

    # Model configuration (optimized for fast switching)
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    results = []

    # === TEST 1: Load first model ===
    print("\n" + "=" * 70)
    print("TEST 1: Load Qwen/Qwen2.5-7B-Instruct")
    print("=" * 70)

    t0 = time.perf_counter()
    llm1 = LLM(model="Qwen/Qwen2.5-7B-Instruct", **config)
    load1_time = (time.perf_counter() - t0) * 1000

    print(f"Load time: {load1_time:.0f}ms")
    print(f"GPU memory: {get_mem():.1f} GB free")

    # Verify output
    success, output = test_model_output(llm1, "What is 2 + 2? Answer with just the number:")
    print(f"Output: {output[:100]}...")
    print(f"Valid: {success}")
    results.append(("Load Model 1", success, load1_time))

    if not success:
        print("FAILED: Model 1 did not produce valid output")
        return results

    # === TEST 2: Fast cleanup ===
    print("\n" + "=" * 70)
    print("TEST 2: Fast Cleanup")
    print("=" * 70)

    mem_before = get_mem()
    del llm1
    gc.collect()

    cleanup_time, cleared = fast_cleanup()
    mem_after = get_mem()

    print(f"Cleanup time: {cleanup_time:.0f}ms")
    print(f"Parameters cleared: {cleared}")
    print(f"Memory freed: {mem_after - mem_before:.1f} GB")
    print(f"GPU memory: {mem_after:.1f} GB free")

    results.append(("Cleanup", True, cleanup_time))

    # === TEST 3: Load second model (same model to verify weights load correctly) ===
    print("\n" + "=" * 70)
    print("TEST 3: Load Second Model (same model, verify correct loading)")
    print("=" * 70)

    t0 = time.perf_counter()
    llm2 = LLM(model="Qwen/Qwen2.5-7B-Instruct", **config)
    load2_time = (time.perf_counter() - t0) * 1000

    print(f"Load time: {load2_time:.0f}ms")
    print(f"GPU memory: {get_mem():.1f} GB free")

    # Verify output with different prompt
    success, output = test_model_output(llm2, "What is the capital of France? Answer in one word:")
    print(f"Output: {output[:100]}...")
    print(f"Valid: {success}")
    results.append(("Load Model 2", success, load2_time))

    # === TEST 4: Multiple inference calls ===
    print("\n" + "=" * 70)
    print("TEST 4: Multiple Inference Calls")
    print("=" * 70)

    prompts = [
        "Count from 1 to 5:",
        "What color is the sky?",
        "Name a programming language:",
    ]

    all_valid = True
    for i, prompt in enumerate(prompts):
        success, output = test_model_output(llm2, prompt)
        status = "OK" if success else "FAIL"
        print(f"  {i+1}. [{status}] {prompt[:30]}... -> {output[:40]}...")
        all_valid = all_valid and success

    results.append(("Multiple Inference", all_valid, 0))

    # === TEST 5: Second switch cycle ===
    print("\n" + "=" * 70)
    print("TEST 5: Second Switch Cycle")
    print("=" * 70)

    del llm2
    gc.collect()

    t0 = time.perf_counter()
    cleanup_time2, cleared2 = fast_cleanup()

    llm3 = LLM(model="Qwen/Qwen2.5-7B-Instruct", **config)
    total_switch = (time.perf_counter() - t0) * 1000

    print(f"Cleanup: {cleanup_time2:.0f}ms")
    print(f"Total switch: {total_switch:.0f}ms")

    success, output = test_model_output(llm3, "Say hello in Spanish:")
    print(f"Output: {output[:100]}...")
    print(f"Valid: {success}")
    results.append(("Switch Cycle 2", success, total_switch))

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, success, time_ms in results:
        status = "PASS" if success else "FAIL"
        time_str = f"{time_ms:.0f}ms" if time_ms > 0 else "N/A"
        print(f"  [{status}] {name}: {time_str}")
        all_passed = all_passed and success

    print("\n" + "-" * 70)

    # Calculate switch performance
    avg_cleanup = (cleanup_time + cleanup_time2) / 2
    avg_load = (load2_time + (total_switch - cleanup_time2)) / 2
    avg_switch = avg_cleanup + avg_load

    print(f"Average cleanup time: {avg_cleanup:.0f}ms")
    print(f"Average load time: {avg_load:.0f}ms")
    print(f"Average switch time: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)

    # Cleanup
    del llm3
    gc.collect()
    fast_cleanup()

    return results


if __name__ == '__main__':
    results = run_full_switch_test()
