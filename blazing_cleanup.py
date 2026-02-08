#!/usr/bin/env python3
"""
Blazing fast cleanup - skip gc.collect() entirely.
Let the next allocation trigger gc if needed.
Target: <300ms cleanup.
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


def count_tensors():
    count = 0
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                total += obj.numel() * obj.element_size()
        except:
            pass
    return count, total / (1024**3)


def blazing_cleanup():
    """
    Blazing fast: unfreeze + clear params + empty cache.
    Skip gc.collect() entirely - let next allocation trigger it.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze (instant)
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear _parameters (weights only, preserve buffers)
    t0 = time.perf_counter()
    params_cleared = 0
    bytes_cleared = 0

    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            bytes_cleared += param.numel() * param.element_size()
                            obj._parameters[key] = None
                            params_cleared += 1
        except:
            pass

    times['clear_params'] = (time.perf_counter() - t0) * 1000

    # Step 3: Empty cache ONLY - skip gc.collect()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times, params_cleared, bytes_cleared / (1024**3)


def blazing_cleanup_no_iterate():
    """
    Even faster: Skip iterating gc.get_objects() entirely.
    Just unfreeze + empty_cache.
    """
    start = time.perf_counter()

    gc.unfreeze()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return (time.perf_counter() - start) * 1000


def blazing_cleanup_delayed_gc():
    """
    Clear params, then schedule gc for later.
    """
    start = time.perf_counter()
    times = {}

    # Unfreeze
    gc.unfreeze()

    # Clear params
    t0 = time.perf_counter()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            obj._parameters[key] = None
        except:
            pass
    times['clear_params'] = (time.perf_counter() - t0) * 1000

    # Empty cache
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times


def test_blazing():
    print("=" * 70)
    print("BLAZING FAST CLEANUP TEST")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    methods = [
        ("blazing (no gc.collect)", "full"),
        ("no_iterate (just unfreeze+empty)", "min"),
        ("delayed_gc", "delayed"),
    ]

    for name, method in methods:
        print(f"\n{'='*70}")
        print(f"METHOD: {name}")
        print(f"{'='*70}")

        # Fresh state
        gc.collect()
        gc.unfreeze()
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
        print(f"Output: {out[0].outputs[0].text}")

        after_load = get_mem()
        print(f"After load: {after_load:.2f} GB (used {initial - after_load:.2f} GB)")

        # Delete
        print("\n>>> del llm")
        del llm
        del out

        # Cleanup
        print(f"\n>>> {name} cleanup")
        if method == "full":
            total, times, params, gb = blazing_cleanup()
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Params cleared: {params}, {gb:.2f} GB")
        elif method == "min":
            total = blazing_cleanup_no_iterate()
            print(f"Total: {total:.1f}ms")
        else:  # delayed
            total, times = blazing_cleanup_delayed_gc()
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")

        after_cleanup = get_mem()
        leaked = initial - after_cleanup
        print(f"\nAfter cleanup: {after_cleanup:.2f} GB")
        print(f"Leaked: {leaked:.2f} GB")

        # Try loading second model
        print("\n>>> Loading second model...")
        try:
            load_start = time.perf_counter()
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
            load_time = (time.perf_counter() - load_start) * 1000
            out2 = llm2.generate(["Test"], SamplingParams(max_tokens=3))
            print(f"SUCCESS! Output: {out2[0].outputs[0].text}")
            print(f"Load: {load_time:.0f}ms")
            print(f"TOTAL SWITCH: {total + load_time:.0f}ms")

            # Memory during second model
            after_load2 = get_mem()
            print(f"Second model memory: {after_load2:.2f} GB")

            del llm2
            del out2
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:100]}")

        # Cleanup for next test
        gc.collect()
        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)


if __name__ == '__main__':
    test_blazing()
