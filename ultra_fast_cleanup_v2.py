#!/usr/bin/env python3
"""
Ultra-fast cleanup v2 - only clear parameters, preserve buffers.
Target: <300ms cleanup, working model switching.
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


def ultra_cleanup_params_only():
    """
    Clear only _parameters (weights), preserve _buffers (caches).
    This is the key insight: buffers are small, params are big.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze GC
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear only _parameters on nn.Modules
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

    # Step 3: gc.collect
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 4: Empty CUDA cache
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times, params_cleared, bytes_cleared / (1024**3)


def ultra_cleanup_weight_dicts():
    """
    Clear weight dicts (like BlitzModelSwitcher) but skip vLLM cleanup.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear weight dicts (but NOT cos_sin_cache dicts)
    t0 = time.perf_counter()
    cleared = 0

    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict) and 'weight' in obj:
                val = obj.get('weight')
                if torch.is_tensor(val) and val.is_cuda:
                    for key in list(obj.keys()):
                        v = obj.get(key)
                        if torch.is_tensor(v) and v.is_cuda:
                            obj[key] = None
                            cleared += 1
        except:
            pass

    times['clear_weights'] = (time.perf_counter() - t0) * 1000

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


def ultra_cleanup_combined():
    """
    Best of both: unfreeze + params + weight dicts + empty cache.
    No vLLM cleanup overhead.
    """
    start = time.perf_counter()
    times = {}

    # Step 1: Unfreeze (so we can see frozen modules)
    t0 = time.perf_counter()
    gc.unfreeze()
    times['gc.unfreeze'] = (time.perf_counter() - t0) * 1000

    # Step 2: Clear _parameters on nn.Modules (main weights)
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

    # Step 3: Clear weight dicts (safetensors, etc)
    t0 = time.perf_counter()
    dicts_cleared = 0

    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict) and 'weight' in obj:
                val = obj.get('weight')
                if torch.is_tensor(val) and val.is_cuda:
                    for key in list(obj.keys()):
                        v = obj.get(key)
                        if torch.is_tensor(v) and v.is_cuda:
                            obj[key] = None
                            dicts_cleared += 1
        except:
            pass

    times['clear_dicts'] = (time.perf_counter() - t0) * 1000

    # Step 4: gc.collect
    t0 = time.perf_counter()
    gc.collect()
    times['gc.collect'] = (time.perf_counter() - t0) * 1000

    # Step 5: Empty cache
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    times['empty_cache'] = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - start) * 1000
    return total, times, params_cleared, dicts_cleared, bytes_cleared / (1024**3)


def test_cleanup_methods():
    print("=" * 70)
    print("ULTRA-FAST CLEANUP v2 - PARAMS ONLY")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    methods = [
        ("params_only", ultra_cleanup_params_only),
        ("weight_dicts", ultra_cleanup_weight_dicts),
        ("combined", ultra_cleanup_combined),
    ]

    for name, cleanup_fn in methods:
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
        result = cleanup_fn()

        if name == "params_only":
            total, times, params, gb = result
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Params cleared: {params}, {gb:.2f} GB")
        elif name == "weight_dicts":
            total, times, cleared = result
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Refs cleared: {cleared}")
        else:  # combined
            total, times, params, dicts, gb = result
            print(f"Total: {total:.1f}ms")
            for k, v in times.items():
                print(f"  {k}: {v:.1f}ms")
            print(f"  Params: {params}, Dicts: {dicts}, {gb:.2f} GB")

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
            print(f"FAILED: {type(e).__name__}: {str(e)[:100]}")

        # Full cleanup for next test
        gc.collect()
        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)


if __name__ == '__main__':
    test_cleanup_methods()
