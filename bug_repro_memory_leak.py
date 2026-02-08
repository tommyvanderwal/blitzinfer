#!/usr/bin/env python3
"""
MINIMAL REPRODUCTION: ROCm 780M (gfx1103) GPU Memory Leak

This script demonstrates that GPU memory is not properly released after
tensor deletion on AMD Radeon 780M with ROCm unified memory.

EXPECTED: Memory returns to initial value after del + gc.collect() + empty_cache()
ACTUAL: ~50-80% of allocated memory remains unreclaimable

Environment:
  - AMD Ryzen 9 8945HS with Radeon 780M (gfx1103)
  - ROCm 7.2, PyTorch 2.6+
  - 96GB unified memory (63GB usable by GPU)

Usage:
  python bug_repro_memory_leak.py
"""

import os
import gc
import time

# Force single GPU, disable multiprocessing
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:False'

import torch

def get_memory_gb():
    """Get free GPU memory in GB."""
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3), total / (1024**3)

def full_cleanup():
    """Attempt all known cleanup methods."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass
    gc.collect()

def test_memory_leak():
    """
    Minimal test: allocate tensors, delete, check if memory is freed.
    """
    print("=" * 70)
    print("ROCm 780M MEMORY LEAK REPRODUCTION")
    print("=" * 70)

    # System info
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA/HIP: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

    initial_free, total = get_memory_gb()
    print(f"\nInitial state: {initial_free:.2f} GB free / {total:.2f} GB total")

    # Allocate ~20GB of GPU tensors (simulating model weights)
    print("\n>>> Allocating ~20GB of GPU tensors...")
    tensors = []
    target_gb = 20
    allocated = 0

    # Allocate in 1GB chunks
    chunk_size = 256 * 1024 * 1024  # 256M float32 = 1GB
    while allocated < target_gb:
        t = torch.randn(chunk_size, device='cuda', dtype=torch.float32)
        tensors.append(t)
        allocated += 1

    after_alloc_free, _ = get_memory_gb()
    memory_used = initial_free - after_alloc_free
    print(f"After allocation: {after_alloc_free:.2f} GB free (used {memory_used:.2f} GB)")
    print(f"Created {len(tensors)} tensors")

    # Do some computation to ensure tensors are "used"
    print("\n>>> Performing computation...")
    for i in range(len(tensors) - 1):
        tensors[i] = tensors[i] + 0.001 * tensors[i+1][:tensors[i].shape[0]]
    torch.cuda.synchronize()

    # Delete all tensors
    print("\n>>> Deleting all tensors...")
    del tensors

    # Full cleanup attempt
    print(">>> Running cleanup (gc.collect + empty_cache + ipc_collect)...")
    full_cleanup()
    time.sleep(1.0)  # Give driver time to process
    full_cleanup()

    after_cleanup_free, _ = get_memory_gb()
    memory_recovered = after_cleanup_free - after_alloc_free
    memory_leaked = initial_free - after_cleanup_free

    print(f"\nAfter cleanup: {after_cleanup_free:.2f} GB free")
    print(f"Memory recovered: {memory_recovered:.2f} GB")
    print(f"Memory leaked: {memory_leaked:.2f} GB")

    # Result
    print("\n" + "=" * 70)
    if memory_leaked < 1.0:
        print("RESULT: PASS - Memory properly released")
        return True
    else:
        print(f"RESULT: FAIL - {memory_leaked:.2f} GB leaked (not released to system)")
        print("\nThis is the ROCm 780M memory leak bug.")
        print("Memory is not returned to the system after tensor deletion.")
        return False

def test_reallocation_failure():
    """
    Test that leaked memory prevents reallocation.
    This simulates model switching failure.
    """
    print("\n" + "=" * 70)
    print("REALLOCATION TEST (Model Switching Simulation)")
    print("=" * 70)

    initial_free, total = get_memory_gb()
    print(f"\nInitial: {initial_free:.2f} GB free")

    # First allocation
    print("\n>>> First allocation (~40GB)...")
    tensors1 = []
    for _ in range(40):
        tensors1.append(torch.randn(256 * 1024 * 1024, device='cuda', dtype=torch.float32))

    after_first, _ = get_memory_gb()
    print(f"After first: {after_first:.2f} GB free (used {initial_free - after_first:.2f} GB)")

    # Delete first allocation
    print("\n>>> Deleting first allocation...")
    del tensors1
    full_cleanup()
    time.sleep(1.0)
    full_cleanup()

    after_cleanup, _ = get_memory_gb()
    print(f"After cleanup: {after_cleanup:.2f} GB free")

    # Second allocation - this should work if memory was released
    print("\n>>> Second allocation (~40GB)...")
    try:
        tensors2 = []
        for i in range(40):
            tensors2.append(torch.randn(256 * 1024 * 1024, device='cuda', dtype=torch.float32))
            if (i + 1) % 10 == 0:
                current_free, _ = get_memory_gb()
                print(f"  Allocated {i+1} GB, {current_free:.2f} GB remaining")

        after_second, _ = get_memory_gb()
        print(f"\nSecond allocation succeeded: {after_second:.2f} GB free")
        print("RESULT: PASS - Memory was properly released for reuse")
        del tensors2
        return True

    except RuntimeError as e:
        print(f"\nFAILED to allocate: {e}")
        print("\nRESULT: FAIL - Memory leak prevents reallocation")
        print("This blocks model switching in vLLM/LLM applications.")
        return False

if __name__ == '__main__':
    print("Testing on ROCm 780M (gfx1103)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run both tests
    leak_result = test_memory_leak()
    print()
    realloc_result = test_reallocation_failure()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Memory release test: {'PASS' if leak_result else 'FAIL'}")
    print(f"Reallocation test:   {'PASS' if realloc_result else 'FAIL'}")

    if not leak_result or not realloc_result:
        print("\nBUG CONFIRMED: ROCm 780M does not release GPU memory after deletion.")
        print("See: ROCM_780M_MEMORY_LEAK_BUG.md for full details.")
