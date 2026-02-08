#!/usr/bin/env python3
"""
Detailed profiling of cleanup steps to find every millisecond.
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


def get_mem():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3)


def timed(name):
    """Context manager for timing."""
    class Timer:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.perf_counter()
            self.mem_start = get_mem()
            return self
        def __exit__(self, *args):
            self.elapsed = (time.perf_counter() - self.start) * 1000  # ms
            self.mem_end = get_mem()
            self.mem_freed = self.mem_end - self.mem_start
            print(f"  {self.name}: {self.elapsed:.1f}ms, mem: {self.mem_start:.2f} -> {self.mem_end:.2f} GB ({self.mem_freed:+.2f})")
    return Timer(name)


def count_gpu_tensors():
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                total_bytes += obj.numel() * obj.element_size()
        except:
            pass
    return count, total_bytes / (1024**3)


def profile_cleanup():
    print("=" * 70)
    print("DETAILED CLEANUP PROFILING")
    print("=" * 70)

    initial = get_mem()
    print(f"\nInitial: {initial:.2f} GB free")

    # Load model with smaller config for faster testing
    print("\n>>> Loading model (32GB max)...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.30,  # ~30GB max
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,  # 2GB KV cache
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    after_load = get_mem()
    print(f"After load: {after_load:.2f} GB free (used {initial - after_load:.2f} GB)")

    # Quick inference
    out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
    print(f"Output: {out[0].outputs[0].text}")

    count, size = count_gpu_tensors()
    print(f"GPU tensors: {count}, {size:.2f} GB")

    # === DETAILED CLEANUP PROFILING ===
    print("\n" + "=" * 70)
    print("CLEANUP PROFILING (each step timed)")
    print("=" * 70)

    total_start = time.perf_counter()

    # Step 1: Delete LLM object
    with timed("del llm"):
        del llm
        del out

    # Step 2: First gc.collect() - does this do anything?
    with timed("gc.collect() #1"):
        collected1 = gc.collect()
    print(f"    Objects collected: {collected1}")

    # Step 3: Check tensor count
    count, size = count_gpu_tensors()
    print(f"  Tensors in gc: {count}, {size:.2f} GB")

    # Step 4: vLLM cleanup
    with timed("vllm cleanup_dist_env_and_memory"):
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except Exception as e:
            print(f"    Error: {e}")

    # Step 5: Another gc.collect()
    with timed("gc.collect() #2"):
        collected2 = gc.collect()
    print(f"    Objects collected: {collected2}")

    count, size = count_gpu_tensors()
    print(f"  Tensors in gc: {count}, {size:.2f} GB")

    # Step 6: Clear weight dicts (our fix)
    with timed("clear weight dicts"):
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
    print(f"    Cleared: {cleared} tensor refs")

    # Step 7: gc.collect() after clearing
    with timed("gc.collect() #3"):
        collected3 = gc.collect()
    print(f"    Objects collected: {collected3}")

    count, size = count_gpu_tensors()
    print(f"  Tensors in gc: {count}, {size:.2f} GB")

    # Step 8: torch.cuda.synchronize
    with timed("torch.cuda.synchronize"):
        torch.cuda.synchronize()

    # Step 9: torch.cuda.empty_cache
    with timed("torch.cuda.empty_cache"):
        torch.cuda.empty_cache()

    # Step 10: torch.cuda.ipc_collect
    with timed("torch.cuda.ipc_collect"):
        torch.cuda.ipc_collect()

    # Step 11: Final gc
    with timed("gc.collect() #4"):
        collected4 = gc.collect()
    print(f"    Objects collected: {collected4}")

    total_time = (time.perf_counter() - total_start) * 1000
    after_cleanup = get_mem()
    leaked = initial - after_cleanup

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total cleanup time: {total_time:.1f}ms")
    print(f"Memory: {initial:.2f} -> {after_cleanup:.2f} GB")
    print(f"Leaked: {leaked:.2f} GB")

    # Analyze what's still holding memory
    print("\n>>> Analyzing remaining tensors...")
    count, size = count_gpu_tensors()
    print(f"Tensors in gc: {count}, {size:.2f} GB")

    if count > 0:
        print("\nLargest remaining tensors:")
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                    tensors.append((obj.numel() * obj.element_size(), tuple(obj.shape), obj))
            except:
                pass
        tensors.sort(reverse=True)
        for size_bytes, shape, t in tensors[:5]:
            print(f"  {size_bytes / (1024**2):.1f} MB: {shape}")
            # Check referrers
            refs = gc.get_referrers(t)
            for ref in refs[:3]:
                if isinstance(ref, dict):
                    keys = [k for k, v in list(ref.items())[:20] if v is t]
                    print(f"    -> dict with keys: {keys}")
                elif isinstance(ref, list):
                    print(f"    -> list len {len(ref)}")
                elif hasattr(ref, '__class__'):
                    print(f"    -> {ref.__class__.__module__}.{ref.__class__.__name__}")


def test_gc_automatic():
    """Test if GC happens automatically on allocation."""
    print("\n" + "=" * 70)
    print("TEST: Does GC happen automatically?")
    print("=" * 70)

    gc.disable()  # Disable automatic GC
    print("GC disabled")

    initial = get_mem()
    print(f"Initial: {initial:.2f} GB")

    # Allocate and delete tensors
    print("Allocating 5GB...")
    tensors = [torch.randn(64 * 1024 * 1024, device='cuda') for _ in range(5)]
    after_alloc = get_mem()
    print(f"After alloc: {after_alloc:.2f} GB")

    print("Deleting references...")
    del tensors
    after_del = get_mem()
    print(f"After del (no GC): {after_del:.2f} GB")

    print("Allocating again (should trigger cleanup?)...")
    tensors2 = [torch.randn(64 * 1024 * 1024, device='cuda') for _ in range(5)]
    after_alloc2 = get_mem()
    print(f"After 2nd alloc: {after_alloc2:.2f} GB")

    del tensors2

    gc.enable()
    gc.collect()
    torch.cuda.empty_cache()
    final = get_mem()
    print(f"After GC enabled + collect: {final:.2f} GB")


if __name__ == '__main__':
    profile_cleanup()
    test_gc_automatic()
