#!/usr/bin/env python3
"""
Ultra-fast cleanup - bypass vLLM cleanup, directly clear nn.Module internals.
Target: <300ms cleanup, 0 wasted gc.collect() calls.
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


def ultra_fast_cleanup_v1():
    """
    Ultra-fast cleanup: Directly clear nn.Module internals.
    Skips vLLM cleanup overhead.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze GC (so we can see frozen modules)
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Find and clear ALL nn.Module internals directly
    t0 = time.perf_counter()
    modules_cleared = 0
    params_cleared = 0
    buffers_cleared = 0

    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                # Clear _parameters dict
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            obj._parameters[key] = None
                            params_cleared += 1

                # Clear _buffers dict (includes cos_sin_cache, etc)
                if hasattr(obj, '_buffers') and obj._buffers:
                    for key in list(obj._buffers.keys()):
                        buf = obj._buffers.get(key)
                        if buf is not None and torch.is_tensor(buf) and buf.is_cuda:
                            obj._buffers[key] = None
                            buffers_cleared += 1

                # Clear _modules to break reference chains
                if hasattr(obj, '_modules') and obj._modules:
                    obj._modules.clear()

                modules_cleared += 1
        except Exception as e:
            pass

    times['clear_modules'] = (time.perf_counter() - t0) * 1000

    # Step 3: Single gc.collect after clearing references
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 4: Empty CUDA cache - this is what actually frees GPU memory
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000

    return total, times, {
        'modules': modules_cleared,
        'params': params_cleared,
        'buffers': buffers_cleared
    }


def ultra_fast_cleanup_v2():
    """
    Even faster: Clear dicts by pattern, skip module type check.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear any dict with CUDA tensors that looks like weight storage
    t0 = time.perf_counter()
    cleared = 0

    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict):
                # Check if it contains CUDA tensors
                has_cuda = False
                for v in list(obj.values())[:5]:  # Sample first 5
                    if torch.is_tensor(v) and v.is_cuda:
                        has_cuda = True
                        break

                if has_cuda:
                    # Clear all CUDA tensor values
                    for key in list(obj.keys()):
                        v = obj.get(key)
                        if torch.is_tensor(v) and v.is_cuda:
                            obj[key] = None
                            cleared += 1
        except:
            pass

    times['clear_dicts'] = (time.perf_counter() - t0) * 1000

    # Step 3: gc.collect
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 4: Empty cache
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times, cleared


def ultra_fast_cleanup_v3():
    """
    Minimal: Just unfreeze, collect, and empty cache.
    Let gc do the work.
    """
    start = time.perf_counter()
    times = {}

    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times


def test_ultra_cleanup():
    print("=" * 70)
    print("ULTRA-FAST CLEANUP TEST")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    for version in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"VERSION {version}")
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
        print(f"Output: {out[0].outputs[0].text}")

        after_load = get_mem()
        print(f"After load: {after_load:.2f} GB (used {initial - after_load:.2f} GB)")

        # Delete
        print("\n>>> del llm")
        del llm
        del out

        # Cleanup
        print(f"\n>>> Ultra cleanup v{version}")
        if version == 1:
            total, times, stats = ultra_fast_cleanup_v1()
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Stats: {stats}")
        elif version == 2:
            total, times, cleared = ultra_fast_cleanup_v2()
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Cleared: {cleared} tensor refs")
        else:
            total, times = ultra_fast_cleanup_v3()
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")

        after_cleanup = get_mem()
        leaked = initial - after_cleanup
        print(f"\nAfter cleanup: {after_cleanup:.2f} GB")
        print(f"Leaked: {leaked:.2f} GB")

        count, size = count_tensors()
        print(f"Tensors in gc: {count}, {size:.2f} GB")

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
            print(f"SUCCESS! Output: {out2[0].outputs[0].text}")
            print(f"Load: {load_time:.0f}ms")
            print(f"TOTAL SWITCH: {total + load_time:.0f}ms")
            del llm2
            del out2
        except Exception as e:
            print(f"FAILED: {e}")

        # Full cleanup for next test
        gc.collect()
        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)


if __name__ == '__main__':
    test_ultra_cleanup()
