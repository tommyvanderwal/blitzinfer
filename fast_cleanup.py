#!/usr/bin/env python3
"""
Optimized fast cleanup - minimize gc.collect() calls.
Target: <500ms cleanup time.
"""

import os
import sys
import gc
import time
import types
import weakref

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def fast_cleanup_v1():
    """
    Optimized cleanup - single gc pass, minimal overhead.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Single gc.collect to process deletions
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear weight dicts in single pass (combine with tensor finding)
    t0 = time.perf_counter()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict):
                # Clear weight dicts
                if 'weight' in obj:
                    for key in list(obj.keys()):
                        v = obj.get(key)
                        if torch.is_tensor(v) and v.is_cuda:
                            obj[key] = None
        except:
            pass
    times['clear_dicts'] = (time.perf_counter() - t0) * 1000

    # Step 3: Empty CUDA cache (this is what actually frees memory)
    t0 = time.perf_counter()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times


def fast_cleanup_v2():
    """
    Even faster - skip gc.collect entirely, just clear dicts.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Clear weight dicts directly
    t0 = time.perf_counter()
    cleared = 0
    # Use gc.get_objects() but don't call gc.collect first
    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict) and 'weight' in obj:
                for key in list(obj.keys()):
                    v = obj.get(key)
                    if torch.is_tensor(v) and v.is_cuda:
                        obj[key] = None
                        cleared += 1
        except:
            pass
    times['clear_dicts'] = (time.perf_counter() - t0) * 1000

    # Step 2: Single gc.collect after clearing
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 3: Empty CUDA cache
    t0 = time.perf_counter()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times, cleared


def fast_cleanup_v3():
    """
    Minimal cleanup - just empty_cache, let next allocation trigger gc.
    """
    start = time.perf_counter()
    torch.cuda.empty_cache()
    return (time.perf_counter() - start) * 1000


def test_cleanup_versions():
    print("=" * 70)
    print("TESTING CLEANUP OPTIMIZATION")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    for version, name in [(1, "v1: gc + clear + empty"),
                          (2, "v2: clear + gc + empty"),
                          (3, "v3: just empty_cache")]:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"{'='*70}")

        # Fresh state
        gc.collect()
        torch.cuda.empty_cache()
        initial = get_mem()
        print(f"Initial: {initial:.2f} GB")

        # Load model
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            gpu_memory_utilization=0.30,
            max_model_len=512,
            max_num_batched_tokens=512,
            kv_cache_memory_bytes=2 * 1024**3,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )
        out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
        after_load = get_mem()
        print(f"After load: {after_load:.2f} GB (used {initial - after_load:.2f} GB)")

        # Delete
        del llm
        del out

        # Cleanup
        if version == 1:
            total, times = fast_cleanup_v1()
            print(f"Cleanup: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
        elif version == 2:
            total, times, cleared = fast_cleanup_v2()
            print(f"Cleanup: {total:.1f}ms (cleared {cleared} refs)")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
        else:
            total = fast_cleanup_v3()
            print(f"Cleanup: {total:.1f}ms")

        after_cleanup = get_mem()
        leaked = initial - after_cleanup
        print(f"After cleanup: {after_cleanup:.2f} GB (leaked {leaked:.2f} GB)")

        # Try loading second model
        print("\n>>> Loading second model...")
        try:
            start = time.perf_counter()
            llm2 = LLM(
                model="Qwen/Qwen2.5-7B-Instruct",
                dtype="float16",
                gpu_memory_utilization=0.30,
                max_model_len=512,
                max_num_batched_tokens=512,
                kv_cache_memory_bytes=2 * 1024**3,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            )
            load_time = (time.perf_counter() - start) * 1000
            out2 = llm2.generate(["Test"], SamplingParams(max_tokens=3))
            print(f"SUCCESS! Load time: {load_time:.0f}ms")
            print(f"Total switch: {total + load_time:.0f}ms")
            del llm2
            del out2
        except Exception as e:
            print(f"FAILED: {e}")

        # Full cleanup for next test
        gc.collect()
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_cleanup_versions()
