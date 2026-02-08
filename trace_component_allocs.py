#!/usr/bin/env python3
"""Test what individual components allocate 160MB blocks."""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_allocated_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def count_large_blocks(min_size_mb=100):
    snapshot = torch.cuda.memory._snapshot()
    if not snapshot or 'segments' not in snapshot:
        return 0, []

    blocks = []
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', block.get('requested_size', 0))
                if size > min_size_mb * 1024 * 1024:
                    blocks.append(size / 1024**2)
    return len(blocks), blocks


def main():
    print("=" * 70)
    print("COMPONENT ALLOCATION TEST")
    print("=" * 70)

    # Enable memory tracing
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=100000
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_allocated_mb()
    n_baseline, _ = count_large_blocks()
    print(f"\n[BASELINE] {baseline:.1f} MB, {n_baseline} large blocks")

    # Test 1: Flash Attention
    print("\n=== TEST 1: Flash Attention ===")
    try:
        from flash_attn import flash_attn_func

        q = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)

        out = flash_attn_func(q, k, v, causal=True)
        del q, k, v, out

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after, blocks = count_large_blocks()
        print(f"  After Flash Attention: {mem_after:.1f} MB, {n_after} large blocks")
        if blocks:
            print(f"  Block sizes: {blocks}")
    except ImportError:
        print("  Flash Attention not available")

    # Test 2: Larger Flash Attention (closer to model scale)
    print("\n=== TEST 2: Large Flash Attention ===")
    try:
        from flash_attn import flash_attn_func

        # Simulate a single layer attention (batch=1, heads=40, seq=4096, head_dim=128)
        q = torch.randn(1, 40, 4096, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 40, 4096, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 40, 4096, 128, device='cuda', dtype=torch.bfloat16)

        out = flash_attn_func(q, k, v, causal=True)
        del q, k, v, out

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after, blocks = count_large_blocks()
        print(f"  After Large Flash Attention: {mem_after:.1f} MB, {n_after} large blocks")
        if blocks:
            print(f"  Block sizes: {blocks}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: cuBLAS large matrix
    print("\n=== TEST 3: cuBLAS Large Matrix ===")
    a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    c = torch.mm(a, b)
    del a, b, c

    gc.collect()
    torch.cuda.empty_cache()

    mem_after = get_allocated_mb()
    n_after, blocks = count_large_blocks()
    print(f"  After large matmul: {mem_after:.1f} MB, {n_after} large blocks")
    if blocks:
        print(f"  Block sizes: {blocks}")

    # Test 4: NCCL with actual work
    print("\n=== TEST 4: NCCL Process Group ===")
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:34567',
            world_size=1,
            rank=0,
        )

        # Create a device mesh / communicator
        tensor = torch.ones(1024 * 1024, device='cuda')  # 4MB tensor
        dist.all_reduce(tensor)
        del tensor

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after, blocks = count_large_blocks()
        print(f"  After NCCL all_reduce: {mem_after:.1f} MB, {n_after} large blocks")
        if blocks:
            print(f"  Block sizes: {blocks}")

        dist.destroy_process_group()

    # Test 5: vLLM's custom NCCL ops
    print("\n=== TEST 5: vLLM Custom All-Reduce ===")
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        # This creates vLLM's custom NCCL communicator
        print("  Creating PyNcclCommunicator...")
        # Need process group first
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:34568',
                world_size=1,
                rank=0,
            )

        comm = PyNcclCommunicator(
            group=dist.group.WORLD,
            device=0,
        )

        # Try an all_reduce
        tensor = torch.ones(1024 * 1024, device='cuda')
        comm.all_reduce(tensor)
        del tensor

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after, blocks = count_large_blocks()
        print(f"  After vLLM PyNccl: {mem_after:.1f} MB, {n_after} large blocks")
        if blocks:
            print(f"  Block sizes: {blocks}")

    except Exception as e:
        print(f"  vLLM custom NCCL: {e}")

    # Test 6: vLLM parallel state initialization
    print("\n=== TEST 6: vLLM Parallel State Init ===")
    try:
        from vllm.distributed import parallel_state

        # This is what vLLM calls during model init
        if not parallel_state.is_initialized():
            print("  vLLM parallel state not initialized yet")
        else:
            print("  vLLM parallel state already initialized")

        mem_after = get_allocated_mb()
        n_after, blocks = count_large_blocks()
        print(f"  Current: {mem_after:.1f} MB, {n_after} large blocks")
        if blocks:
            print(f"  Block sizes: {blocks}")

    except Exception as e:
        print(f"  vLLM parallel state: {e}")

    torch.cuda.memory._record_memory_history(enabled=None)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    final = get_allocated_mb()
    n_final, blocks_final = count_large_blocks()
    print(f"Memory: {final:.1f} MB, {n_final} large blocks")
    if blocks_final:
        print(f"Block sizes: {blocks_final}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
