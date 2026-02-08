#!/usr/bin/env python3
"""Debug pinned memory overhead."""

import torch
import subprocess
import gc

def get_mem():
    result = subprocess.run(["cat", "/proc/meminfo"], capture_output=True, text=True)
    mem = {}
    for line in result.stdout.split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            val = int(parts[1].strip().split()[0])  # kB
            mem[key] = val
    return mem

def show_mem(label):
    m = get_mem()
    print(f"[{label:20s}]", flush=True)
    print(f"  MemFree:    {m['MemFree']/1024/1024:.2f}GB", flush=True)
    print(f"  MemAvail:   {m['MemAvailable']/1024/1024:.2f}GB", flush=True)
    print(f"  Shmem:      {m['Shmem']/1024/1024:.2f}GB", flush=True)
    print(f"  Mapped:     {m['Mapped']/1024/1024:.2f}GB", flush=True)
    print(f"  AnonPages:  {m['AnonPages']/1024/1024:.2f}GB", flush=True)
    print(f"  Cached:     {m['Cached']/1024/1024:.2f}GB", flush=True)
    return m

def main():
    print("=" * 60)
    print("INVESTIGATING PINNED MEMORY OVERHEAD")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    m0 = show_mem("START")

    # Test 1: Regular allocation
    print("\n--- Test 1: 10GB regular tensor ---")
    t1 = torch.empty(10 * 1024**3, dtype=torch.uint8, device="cpu")
    m1 = show_mem("After regular")
    print(f"  Delta Shmem: {(m1['Shmem'] - m0['Shmem'])/1024/1024:.2f}GB")
    del t1
    gc.collect()

    # Test 2: Pinned allocation
    print("\n--- Test 2: 10GB pinned tensor ---")
    t2 = torch.empty(10 * 1024**3, dtype=torch.uint8, device="cpu", pin_memory=True)
    m2 = show_mem("After pinned")
    print(f"  Delta Shmem: {(m2['Shmem'] - m0['Shmem'])/1024/1024:.2f}GB")
    print(f"  Is pinned: {t2.is_pinned()}")
    print(f"  Data ptr: {hex(t2.data_ptr())}")

    # Check /proc/self/maps for this memory region
    print("\n  Checking memory mappings...")
    result = subprocess.run(["cat", "/proc/self/maps"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "shm" in line.lower() or "cuda" in line.lower() or "nv" in line.lower():
            print(f"    {line}")

    # Check nvidia-smi for host memory
    print("\n  Checking nvidia-smi...")
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv"],
                          capture_output=True, text=True)
    print(f"    {result.stdout.strip()}")

    del t2
    gc.collect()
    torch.cuda.empty_cache()

    # Test 3: Check CUDA pinned memory pools
    print("\n--- Test 3: Check CUDA memory pools ---")
    print(f"  CUDA caching allocator stats:")
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        for key in ['allocated_bytes.all.current', 'reserved_bytes.all.current']:
            if key in stats:
                print(f"    {key}: {stats[key]/1024/1024/1024:.2f}GB")

    # Test 4: Multiple smaller pinned allocations
    print("\n--- Test 4: 10x 1GB pinned allocations ---")
    m3 = get_mem()
    chunks = []
    for i in range(10):
        chunk = torch.empty(1 * 1024**3, dtype=torch.uint8, device="cpu", pin_memory=True)
        chunks.append(chunk)
        m = get_mem()
        delta = (m['Shmem'] - m3['Shmem'])/1024/1024
        print(f"  Chunk {i+1}: Shmem delta = {delta:.2f}GB (expected {i+1}GB)")

    del chunks
    gc.collect()
    show_mem("After cleanup")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
