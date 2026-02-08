#!/usr/bin/env python3
"""Investigate if we can create Python references to phantom CUDA allocations.

These are the 3x160MB blocks allocated by internal CUDA libraries (NCCL, cuBLAS,
Flash Attention) through PyTorch's allocator but WITHOUT Python tensor wrappers.

Approaches to try:
1. Use data_ptr from memory snapshot to create tensor views
2. Force cuBLAS/cuDNN handle destruction
3. Use CUDA driver API to find and free allocations
4. Reset PyTorch's internal CUDA state more aggressively
"""

import os
import gc
import sys
import ctypes

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        'cuda_used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
    }


def get_phantom_blocks():
    """Get memory blocks that have no Python references (no trace)."""
    snapshot = torch.cuda.memory._snapshot()
    if not snapshot or 'segments' not in snapshot:
        return []

    phantom_blocks = []
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                # Check if this block has no stack trace
                frames = block.get('frames', [])
                history = block.get('history', [])

                has_trace = False
                if frames:
                    has_trace = True
                elif history:
                    for h in history:
                        if h.get('frames'):
                            has_trace = True
                            break

                if not has_trace:
                    phantom_blocks.append({
                        'address': block.get('address', segment.get('address', 0)),
                        'size': block.get('size', block.get('requested_size', 0)),
                        'segment_address': segment.get('address', 0),
                    })

    return phantom_blocks


def approach_1_tensor_from_pointer():
    """Try to create a tensor from the phantom block's data pointer.

    This would give us a Python reference that we could then resize_(0).
    """
    print("\n=== APPROACH 1: Create tensor from pointer ===")

    phantom_blocks = get_phantom_blocks()
    print(f"Found {len(phantom_blocks)} phantom blocks")

    for i, block in enumerate(sorted(phantom_blocks, key=lambda x: x['size'], reverse=True)[:5]):
        addr = block['address']
        size = block['size']
        print(f"\n  Block {i}: {size/1024**2:.1f} MB at 0x{addr:x}")

        if addr == 0 or size == 0:
            print("    Skipping: invalid address or size")
            continue

        # Try to create a tensor view of this memory
        try:
            # Method 1: torch.tensor with device and storage_offset
            # This requires knowing the exact layout...

            # Method 2: Use ctypes to create a CUDA pointer, then wrap it
            # Problem: PyTorch doesn't have a public API for this

            # Method 3: Try torch.frombuffer on CUDA memory (doesn't work)
            # buffer = ctypes.cast(addr, ctypes.POINTER(ctypes.c_char * size))
            # This won't work because we can't access GPU memory from CPU

            # Method 4: Try torch.cuda.memory._get_block_tensor (hypothetical)
            # This function doesn't exist in PyTorch

            # Method 5: Search for existing tensors that might share this storage
            print(f"    Searching for tensors sharing storage at this address...")
            found = False
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        if obj.data_ptr() == addr:
                            print(f"    FOUND! Tensor: shape={obj.shape}, dtype={obj.dtype}")
                            found = True
                except Exception:
                    pass

            if not found:
                print(f"    No Python tensor found for this address")

        except Exception as e:
            print(f"    Error: {e}")

    return False  # Approach 1 cannot create refs to orphaned memory


def approach_2_cublas_reset():
    """Try to destroy and recreate cuBLAS handles.

    cuBLAS allocates ~160MB workspace buffers.
    """
    print("\n=== APPROACH 2: Reset cuBLAS handles ===")

    before = get_memory()
    print(f"  Before: {before['allocated_gb']:.3f} GB allocated")

    try:
        # PyTorch's internal cuBLAS handle management
        # There's no public API, but we can try some approaches

        # Method 1: Clear all cached cuBLAS workspaces via dummy operations
        # Force cuBLAS to reallocate by changing algo preferences
        print("  Toggling cuBLAS preferences...")

        # Save current state
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_allow_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction

        # Toggle settings to force workspace reallocation
        torch.backends.cuda.matmul.allow_tf32 = not old_allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not old_allow_fp16

        # Do a matmul to trigger reallocation
        a = torch.randn(64, 64, device='cuda')
        b = torch.randn(64, 64, device='cuda')
        _ = torch.mm(a, b)
        del a, b, _

        # Restore
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_allow_fp16

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        after = get_memory()
        delta = after['allocated_gb'] - before['allocated_gb']
        print(f"  After: {after['allocated_gb']:.3f} GB allocated (delta: {delta:+.3f} GB)")

        # Method 2: Try to access cuBLAS via ctypes
        print("\n  Attempting direct cuBLAS access via ctypes...")
        try:
            libcublas = ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
            print("    Loaded libcublas.so.12")

            # cublasDestroy signature: cublasStatus_t cublasDestroy(cublasHandle_t handle)
            # But we don't have the handle that PyTorch uses internally
            # We'd need to hook into PyTorch's internal state

        except Exception as e:
            print(f"    Could not load libcublas: {e}")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def approach_3_nccl_deep_cleanup():
    """Try deeper NCCL cleanup.

    NCCL allocates persistent buffers for communication.
    """
    print("\n=== APPROACH 3: Deep NCCL cleanup ===")

    before = get_memory()
    print(f"  Before: {before['allocated_gb']:.3f} GB allocated")

    try:
        # Check if NCCL is initialized
        import torch.distributed as dist

        if not dist.is_initialized():
            print("  torch.distributed not initialized - NCCL not active")
            return False

        print(f"  dist initialized: world_size={dist.get_world_size()}")

        # Try to get NCCL comm and destroy it
        try:
            from torch._C._distributed_c10d import ProcessGroupNCCL

            # Get the process group
            pg = dist.group.WORLD
            print(f"  Process group: {pg}")

            # NCCL stores communicators internally
            # There's no public API to destroy them without destroying the group

        except Exception as e:
            print(f"  NCCL access error: {e}")

        # Force barrier and destroy
        print("  Destroying process group...")
        dist.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after = get_memory()
        delta = after['allocated_gb'] - before['allocated_gb']
        print(f"  After: {after['allocated_gb']:.3f} GB allocated (delta: {delta:+.3f} GB)")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def approach_4_cuda_driver_api():
    """Try using CUDA driver API to enumerate and free allocations.

    The CUDA driver maintains its own allocation tracking.
    """
    print("\n=== APPROACH 4: CUDA driver API ===")

    before = get_memory()
    print(f"  Before: {before['allocated_gb']:.3f} GB allocated")

    try:
        # Load CUDA driver library
        libcuda = ctypes.CDLL("libcuda.so.1")
        print("  Loaded libcuda.so.1")

        # We could try:
        # - cuMemGetAddressRange to find allocation ranges
        # - cuMemFree to free allocations
        # But this is dangerous and would corrupt PyTorch's state

        print("  CUDA driver API cannot safely free PyTorch allocations")
        print("  (would corrupt caching allocator state)")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def approach_5_allocator_trim():
    """Try more aggressive PyTorch allocator trimming.

    PyTorch 2.0+ has memory trimming APIs.
    """
    print("\n=== APPROACH 5: Aggressive allocator trim ===")

    before = get_memory()
    print(f"  Before: {before['allocated_gb']:.3f} GB allocated")

    try:
        # Method 1: Standard empty_cache
        print("  Calling empty_cache()...")
        torch.cuda.empty_cache()

        after1 = get_memory()
        print(f"    After empty_cache: {after1['allocated_gb']:.3f} GB")

        # Method 2: Reset peak stats (doesn't free memory but resets tracking)
        print("  Resetting peak stats...")
        torch.cuda.reset_peak_memory_stats()

        # Method 3: Try to trigger allocator defragmentation
        print("  Triggering defragmentation via large allocation...")
        try:
            # Allocate a large tensor to force defragmentation
            large = torch.empty(1024, 1024, 1024, device='cuda', dtype=torch.float16)  # 2GB
            del large
        except RuntimeError:
            # If OOM, try smaller
            try:
                large = torch.empty(512, 512, 512, device='cuda', dtype=torch.float16)
                del large
            except RuntimeError:
                pass

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after2 = get_memory()
        delta = after2['allocated_gb'] - before['allocated_gb']
        print(f"    After defrag attempt: {after2['allocated_gb']:.3f} GB (delta: {delta:+.3f} GB)")

        # Method 4: Check if there's a memory.trim function (PyTorch 2.1+)
        if hasattr(torch.cuda.memory, 'trim'):
            print("  Found torch.cuda.memory.trim(), calling...")
            torch.cuda.memory.trim()
            after3 = get_memory()
            print(f"    After trim: {after3['allocated_gb']:.3f} GB")
        else:
            print("  torch.cuda.memory.trim() not available in this PyTorch version")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def approach_6_reset_caching_allocator():
    """Try resetting the entire caching allocator.

    This is nuclear but might free orphaned allocations.
    """
    print("\n=== APPROACH 6: Reset caching allocator ===")

    before = get_memory()
    print(f"  Before: {before['allocated_gb']:.3f} GB allocated")

    try:
        # Check for reset function
        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'reset_peak_memory_stats'):
            print("  Available memory functions:")
            for attr in dir(torch.cuda.memory):
                if not attr.startswith('_'):
                    print(f"    - {attr}")

        # The nuclear option would be to call the C++ allocator directly
        # But this would require ctypes access to libtorch

        # Check for CUDACachingAllocator reset
        # In PyTorch source: c10/cuda/CUDACachingAllocator.cpp
        # There's emptyCache() and resetAccumulatedStats()
        # But no full reset without restarting the process

        print("\n  PyTorch's caching allocator cannot be fully reset without process restart")
        print("  The allocator maintains internal state that survives empty_cache()")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def approach_7_identify_source():
    """Identify which library allocated each phantom block.

    Even if we can't free them, knowing the source helps.
    """
    print("\n=== APPROACH 7: Identify allocation sources ===")

    # Load with detailed tracing enabled
    print("  Enabling detailed memory history...")
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=100000
    )

    # The phantom blocks were allocated BEFORE we enabled tracing
    # So they won't have traces
    # But we can infer from their sizes:

    phantom_blocks = get_phantom_blocks()

    print(f"\n  Phantom block analysis ({len(phantom_blocks)} blocks):")

    # Group by size
    size_groups = {}
    for block in phantom_blocks:
        size_mb = block['size'] / 1024**2
        # Round to nearest MB for grouping
        size_key = round(size_mb)
        if size_key not in size_groups:
            size_groups[size_key] = []
        size_groups[size_key].append(block)

    print("\n  Blocks grouped by size:")
    for size_mb, blocks in sorted(size_groups.items(), key=lambda x: -x[0]):
        if size_mb >= 1:
            # Try to identify source by size
            source = "unknown"
            if 150 <= size_mb <= 170:
                source = "likely NCCL comm buffer or cuBLAS workspace"
            elif 30 <= size_mb <= 50:
                source = "likely Flash Attention workspace"
            elif size_mb > 500:
                source = "likely model weights fragment"

            print(f"    {size_mb} MB: {len(blocks)} block(s) - {source}")

    # Small blocks
    small_count = sum(1 for b in phantom_blocks if b['size'] < 1024 * 1024)
    print(f"    < 1 MB: {small_count} block(s) - likely CUDA context overhead")

    torch.cuda.memory._record_memory_history(enabled=None)

    return True


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("INVESTIGATING PHANTOM ALLOCATION REFERENCES")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_memory()
    print(f"\n[BASELINE] allocated: {baseline['allocated_gb']:.3f} GB")

    # Load and unload a model to create phantom allocations
    print("\n--- Loading model to create phantom allocations ---")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )

    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    _ = out[0].outputs[0].text
    print("  Model loaded and tested")

    # Cleanup
    print("\n--- Cleanup ---")
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after_cleanup = get_memory()
    phantom_size = after_cleanup['allocated_gb'] - baseline['allocated_gb']
    print(f"[after cleanup] allocated: {after_cleanup['allocated_gb']:.3f} GB")
    print(f"[phantom size] {phantom_size:.3f} GB")

    # Now try each approach
    print("\n" + "=" * 70)
    print("ATTEMPTING TO CREATE REFERENCES / FREE PHANTOM ALLOCATIONS")
    print("=" * 70)

    approach_1_tensor_from_pointer()
    approach_5_allocator_trim()
    approach_6_reset_caching_allocator()
    approach_7_identify_source()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final = get_memory()
    phantom_remaining = final['allocated_gb'] - baseline['allocated_gb']

    print(f"""
Phantom allocations: {phantom_remaining:.3f} GB

The problem is:
1. These blocks were allocated by CUDA libraries (NCCL, cuBLAS, Flash Attention)
   through PyTorch's caching allocator
2. They are NOT wrapped in Python tensors - they're raw CUDA memory
3. PyTorch's empty_cache() only frees CACHED memory, not ALLOCATED memory
4. The allocations are considered "in use" by the allocator

Possible solutions:
A. Process isolation - run vLLM in subprocess, kill to free memory
B. Patch the allocating library to properly free buffers
C. Accept the ~0.5GB overhead per model load
D. Use CUDA IPC to share memory instead of reallocating

Option A (subprocess) is the most practical for production.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
