#!/usr/bin/env python3
"""
Test ROCm environment variable workarounds for 780M memory issues.
"""

import os

# Known workarounds for gfx1103/780M
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'  # Force gfx1100 kernels
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Try to force memory release behavior
os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA, use shader copies
os.environ['GPU_MAX_HEAP_SIZE'] = '100'  # Allow full heap
os.environ['GPU_FORCE_64BIT_PTR'] = '1'  # Force 64-bit pointers

import sys
import types
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import gc
import time
import torch


def get_memory():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_inproc_with_workarounds():
    """Test in-process model switching with environment workarounds."""
    print("="*70)
    print("TEST: In-Process Switching with 780M Workarounds")
    print("="*70)
    print("\nEnvironment:")
    print(f"  HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print(f"  HSA_ENABLE_SDMA={os.environ.get('HSA_ENABLE_SDMA')}")

    initial = get_memory()
    print(f"\nInitial free: {initial:.2f} GB")

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    # First model
    print("\n>>> Loading first model...")
    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.60,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=4 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load1 = time.perf_counter() - start
    print(f"Loaded in {load1:.2f}s")

    after_load = get_memory()
    print(f"Free: {after_load:.2f} GB (used {initial - after_load:.2f} GB)")

    # Generate
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Cleanup
    print("\n>>> Cleaning up...")
    del llm
    gc.collect()

    try:
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception as e:
        print(f"  Warning: {e}")

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Extra: try to reset CUDA
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass

    time.sleep(1.0)
    gc.collect()
    torch.cuda.empty_cache()

    after_cleanup = get_memory()
    print(f"After cleanup free: {after_cleanup:.2f} GB")
    print(f"Memory recovered: {after_cleanup - after_load:.2f} GB")
    print(f"Memory leaked: {initial - after_cleanup:.2f} GB")

    if initial - after_cleanup > 2.0:
        print("\n⚠️  Memory leak still present")
        print("   Trying second model anyway...")
    else:
        print("\n✓ Memory properly released!")

    # Second model
    print("\n>>> Loading second model...")
    start = time.perf_counter()
    try:
        llm2 = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            gpu_memory_utilization=0.60,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            kv_cache_memory_bytes=4 * 1024**3,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )
        load2 = time.perf_counter() - start
        print(f"Loaded in {load2:.2f}s")

        out = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {out[0].outputs[0].text}")

        print(f"\n✓ SUCCESS! Both models ran!")
        print(f"Total switch time: {load2:.2f}s")

        del llm2

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_inproc_with_workarounds()
