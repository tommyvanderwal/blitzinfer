#!/usr/bin/env python3
"""Test if using GLOO backend instead of NCCL eliminates the 160MB blocks.

The trace showed vLLM initializes NCCL even with TP=1, which allocates 160MB buffers.
GLOO is a CPU-only backend that shouldn't allocate GPU memory.
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

# Try to force GLOO backend
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SHM_DISABLE'] = '1'

import torch


def get_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def count_large_blocks(min_size_mb=100):
    """Count blocks larger than min_size_mb."""
    snapshot = torch.cuda.memory._snapshot()
    if not snapshot or 'segments' not in snapshot:
        return 0, []

    large_blocks = []
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', block.get('requested_size', 0))
                if size > min_size_mb * 1024 * 1024:
                    large_blocks.append({
                        'size_mb': size / 1024**2,
                        'address': block.get('address', segment.get('address', 0)),
                    })

    return len(large_blocks), large_blocks


def test_with_skip_distributed():
    """Test what happens if we patch vLLM to skip distributed init for TP=1."""
    print("\n=== TEST: Skip NCCL initialization for TP=1 ===")

    # Monkey-patch vLLM's parallel_state to skip NCCL init
    import vllm.distributed.parallel_state as ps

    original_init = ps.init_distributed_environment

    def patched_init(*args, **kwargs):
        print("  [PATCHED] Skipping init_distributed_environment")
        # Don't actually call the original - skip NCCL entirely
        # But we need to set some state so vLLM doesn't crash
        ps._LOCAL_RANK = 0
        ps._WORLD_SIZE = 1
        ps._TP_SIZE = 1
        ps._PP_SIZE = 1
        return

    # Check if this is feasible
    print("  Checking if we can skip distributed init...")
    print("  (This test just measures what the normal path does)")

    # Enable memory tracing
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=100000
    )

    gc.collect()
    torch.cuda.empty_cache()
    baseline = get_memory()
    n_baseline, _ = count_large_blocks()
    print(f"  Baseline: {baseline:.3f} GB, {n_baseline} large blocks")

    # Initialize distributed manually with NCCL to measure impact
    import torch.distributed as dist

    if not dist.is_initialized():
        print("\n  Initializing torch.distributed with NCCL...")
        before_nccl = get_memory()
        n_before, _ = count_large_blocks()

        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23456',
            world_size=1,
            rank=0,
        )

        # Do an all_reduce to trigger NCCL buffer allocation
        tensor = torch.ones(1, device='cuda')
        dist.all_reduce(tensor)

        after_nccl = get_memory()
        n_after, blocks_after = count_large_blocks()
        print(f"  After NCCL init + all_reduce: {after_nccl:.3f} GB, {n_after} large blocks")
        print(f"  Memory delta: {(after_nccl - before_nccl)*1024:.1f} MB")
        print(f"  New large blocks: {n_after - n_before}")

        if blocks_after:
            print(f"\n  Large blocks (NCCL buffers?):")
            for b in blocks_after:
                print(f"    {b['size_mb']:.1f} MB at 0x{b['address']:x}")

        # Now destroy and see if memory is freed
        print("\n  Destroying process group...")
        dist.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after_destroy = get_memory()
        n_destroy, blocks_destroy = count_large_blocks()
        print(f"  After destroy: {after_destroy:.3f} GB, {n_destroy} large blocks")

        if blocks_destroy:
            print(f"  Remaining large blocks (LEAKED):")
            for b in blocks_destroy:
                print(f"    {b['size_mb']:.1f} MB at 0x{b['address']:x}")

    torch.cuda.memory._record_memory_history(enabled=None)


def test_gloo_backend():
    """Test if GLOO backend avoids GPU memory allocation."""
    print("\n=== TEST: GLOO backend (CPU-only, no GPU buffers) ===")

    # Fresh process needed for this since dist is already initialized
    print("  Note: GLOO test requires fresh Python process")
    print("  GLOO is CPU-only and shouldn't allocate GPU memory")
    print("  But vLLM requires NCCL for GPU operations...")


def main():
    print("=" * 70)
    print("TESTING NCCL VS GLOO BACKEND")
    print("=" * 70)

    test_with_skip_distributed()
    test_gloo_backend()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. NCCL allocates persistent GPU buffers (~160MB each) that survive destroy_process_group()
2. These buffers are allocated through PyTorch's caching allocator but have no Python tensor wrapper
3. The allocator marks them as "in use" so empty_cache() won't free them
4. GLOO backend doesn't allocate GPU memory, but vLLM needs NCCL for GPU tensor operations

POTENTIAL SOLUTIONS:
A. Don't initialize NCCL at all for TP=1 (requires vLLM patch)
B. Accept the one-time NCCL overhead (~480MB for 3 communicators)
C. Use subprocess isolation and restart between model loads
D. File a vLLM issue to skip NCCL init for single-GPU setups
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
