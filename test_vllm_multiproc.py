#!/usr/bin/env python3
"""
Test vLLM's native multiprocessing mode for model switching.
This uses vLLM's built-in worker process management.
"""

import os
# DO NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0 - let vLLM use its own workers
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

import sys
import types
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import gc
import time
import torch


def get_memory():
    """Get free GPU memory in GB."""
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_native_multiproc():
    """
    Test if vLLM's native multiprocessing handles cleanup better.
    """
    print("="*70)
    print("TEST: vLLM Native Multiprocessing Mode")
    print("="*70)

    initial = get_memory()
    print(f"\nInitial free memory: {initial:.2f} GB")

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    # First model
    print("\n>>> Loading first model...")
    start = time.perf_counter()
    llm1 = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.60,  # Conservative for testing
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load1_time = time.perf_counter() - start
    print(f"Loaded in {load1_time:.2f}s")

    after_load1 = get_memory()
    print(f"Free memory: {after_load1:.2f} GB (used {initial - after_load1:.2f} GB)")

    # Generate
    outputs = llm1.generate(["Hello, how are you?"], SamplingParams(max_tokens=10))
    print(f"Output: {outputs[0].outputs[0].text}")

    # Clean up first model
    print("\n>>> Cleaning up first model...")
    cleanup_start = time.perf_counter()
    del llm1
    gc.collect()
    try:
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_time = time.perf_counter() - cleanup_start
    print(f"Cleanup took {cleanup_time:.2f}s")

    after_cleanup = get_memory()
    print(f"Free memory: {after_cleanup:.2f} GB")

    # Small delay
    time.sleep(1.0)

    # Second model
    print("\n>>> Loading second model...")
    start = time.perf_counter()
    try:
        llm2 = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            gpu_memory_utilization=0.60,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )
        load2_time = time.perf_counter() - start
        print(f"Loaded in {load2_time:.2f}s")

        # Generate
        outputs = llm2.generate(["Tell me a joke"], SamplingParams(max_tokens=10))
        print(f"Output: {outputs[0].outputs[0].text}")

        after_load2 = get_memory()
        print(f"Free memory: {after_load2:.2f} GB")

        print("\n✓ SUCCESS! Both models loaded and ran!")
        total_switch = cleanup_time + load2_time
        print(f"Total switch time: {total_switch:.2f}s")

        del llm2

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    test_native_multiproc()
