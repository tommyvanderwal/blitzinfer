#!/usr/bin/env python3
"""Profile memory usage during arena allocation and premerge.

This test isolates memory consumption to find where the extra memory is going.
"""

import os
import gc
import time

os.environ['PYTHONUNBUFFERED'] = '1'

def get_mem():
    """Get memory stats from /proc/meminfo."""
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = int(parts[1].strip().split()[0]) / 1024 / 1024  # GB
                mem[key] = val
    return mem

def log_mem(label):
    mem = get_mem()
    avail = mem.get('MemAvailable', 0)
    cached = mem.get('Cached', 0)
    shmem = mem.get('Shmem', 0)
    print(f"[{label:30s}] Available: {avail:.1f}GB, Cached: {cached:.1f}GB, Shmem: {shmem:.1f}GB", flush=True)
    return mem

def main():
    import torch
    from blitzinfer.memory import PinnedMemoryArena, load_model_to_arena, get_premerged_tensors_for_vllm
    from huggingface_hub import snapshot_download

    MODEL = "openai/gpt-oss-120b"  # 65GB
    ARENA_SIZE = 70.0

    print("=" * 70)
    print("MEMORY PROFILING TEST")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Arena: {ARENA_SIZE}GB")
    print()

    log_mem("START")

    # Step 1: Allocate arena
    print("\n[STEP 1] Allocating arena...")
    arena = PinnedMemoryArena(ARENA_SIZE, chunk_size_gb=10.0)
    gc.collect()
    log_mem("AFTER ARENA ALLOC")

    # Step 2: Get model path
    print("\n[STEP 2] Getting model path...")
    model_path = snapshot_download(MODEL, local_files_only=True)
    print(f"  Path: {model_path}")

    # Step 3: Load into arena
    print("\n[STEP 3] Loading model into arena...")
    t0 = time.perf_counter()
    load_model_to_arena(model_path, arena, MODEL)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    gc.collect()
    log_mem("AFTER ARENA LOAD")

    # Step 4: Get tensor views
    print("\n[STEP 4] Getting tensor views...")
    tensors = arena.get_all_tensors(MODEL)
    print(f"  Got {len(tensors)} tensors")
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  Total size: {total_size / 1e9:.2f}GB")
    gc.collect()
    log_mem("AFTER GET TENSORS")

    # Step 5: Premerge (with pin_merged=False to avoid double allocation)
    print("\n[STEP 5] Pre-merging tensors (pin_merged=False)...")
    t0 = time.perf_counter()
    premerged = get_premerged_tensors_for_vllm(tensors, pin_merged=False)
    print(f"  Pre-merged in {time.perf_counter() - t0:.1f}s")
    print(f"  Got {len(premerged)} tensors")
    premerged_size = sum(t.numel() * t.element_size() for t in premerged.values())
    print(f"  Total premerged size: {premerged_size / 1e9:.2f}GB")
    gc.collect()
    log_mem("AFTER PREMERGE")

    # Step 6: Check what new memory was allocated
    print("\n[STEP 6] Checking merged tensor memory...")
    merged_count = 0
    merged_bytes = 0
    for name, tensor in premerged.items():
        # Check if this is a view into arena or a new tensor
        is_arena_view = any(
            tensor.data_ptr() >= arena._chunks[0].data_ptr() and
            tensor.data_ptr() < arena._chunks[0].data_ptr() + len(arena._chunks[0])
            for _ in [0]  # Only check first chunk for simplicity
        ) if arena._chunks else False

        if 'qkv_proj' in name or 'gate_up_proj' in name:
            merged_count += 1
            merged_bytes += tensor.numel() * tensor.element_size()
            is_pinned = tensor.is_pinned() if hasattr(tensor, 'is_pinned') else 'N/A'
            print(f"    {name[:60]:60s}: {tensor.shape}, pinned={is_pinned}")

    print(f"  Merged tensors: {merged_count}, {merged_bytes / 1e9:.2f}GB")

    # Step 7: Release tensors dict but keep arena
    print("\n[STEP 7] Releasing premerged dict...")
    del premerged
    del tensors
    gc.collect()
    log_mem("AFTER DEL PREMERGED")

    # Step 8: Release arena
    print("\n[STEP 8] Releasing arena...")
    arena.clear()
    del arena
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_mem("AFTER DEL ARENA")

    print("\n" + "=" * 70)
    print("MEMORY PROFILE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
