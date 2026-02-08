#!/usr/bin/env python3
"""Use PyTorch memory snapshot to find exactly what's holding 7.5GB.

Key finding: gc.get_objects() finds 0 CUDA tensors, but 7.5GB is "allocated".
This memory must be in PyTorch/CUDA internals, not Python objects.
"""

import os
import gc
import json

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
    }


def log_mem(label):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Alloc: {m['allocated_gb']:.2f}GB, "
          f"Reserved: {m['reserved_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")
    return m


def analyze_memory_snapshot():
    """Analyze PyTorch memory snapshot to find allocations."""
    print("\n=== MEMORY SNAPSHOT ANALYSIS ===\n")

    try:
        # Enable memory history recording
        torch.cuda.memory._record_memory_history(enabled='all')

        # Take snapshot
        snapshot = torch.cuda.memory._snapshot()

        torch.cuda.memory._record_memory_history(enabled=None)

        if not snapshot:
            print("No snapshot data")
            return

        # Analyze segments
        segments = snapshot.get('segments', [])
        print(f"Total segments: {len(segments)}")

        total_allocated = 0
        total_reserved = 0
        allocation_types = {}

        for seg in segments:
            seg_size = seg.get('total_size', 0)
            allocated_size = seg.get('allocated_size', 0)
            total_reserved += seg_size
            total_allocated += allocated_size

            # Track allocation types
            seg_type = seg.get('segment_type', 'unknown')
            if seg_type not in allocation_types:
                allocation_types[seg_type] = {'count': 0, 'size': 0}
            allocation_types[seg_type]['count'] += 1
            allocation_types[seg_type]['size'] += allocated_size

        print(f"\nTotal reserved: {total_reserved / 1024**3:.2f}GB")
        print(f"Total allocated: {total_allocated / 1024**3:.2f}GB")

        print("\nBy segment type:")
        for stype, info in sorted(allocation_types.items(), key=lambda x: x[1]['size'], reverse=True):
            print(f"  {stype}: {info['count']} segments, {info['size'] / 1024**3:.2f}GB")

        # Look at individual blocks
        print("\n=== INDIVIDUAL ALLOCATIONS ===\n")
        blocks = []
        for seg in segments:
            for block in seg.get('blocks', []):
                if block.get('state') == 'active_allocated':
                    size = block.get('size', 0)
                    blocks.append({
                        'size': size,
                        'size_mb': size / 1024**2,
                        'frames': block.get('frames', []),
                    })

        # Sort by size
        blocks.sort(key=lambda x: x['size'], reverse=True)

        print(f"Total active allocations: {len(blocks)}")
        total_block_size = sum(b['size'] for b in blocks)
        print(f"Total size: {total_block_size / 1024**3:.2f}GB")

        print("\nLargest allocations:")
        for i, block in enumerate(blocks[:20]):
            print(f"\n  [{i+1}] {block['size_mb']:.1f}MB")
            # Show stack frames if available
            for frame in block.get('frames', [])[:5]:
                filename = frame.get('filename', 'unknown')
                line = frame.get('line', 0)
                name = frame.get('name', 'unknown')
                if 'site-packages' in filename:
                    filename = filename.split('site-packages/')[-1]
                print(f"      {filename}:{line} {name}")

    except Exception as e:
        print(f"Snapshot error: {e}")
        import traceback
        traceback.print_exc()


def check_cublas_workspace():
    """Check cuBLAS workspace allocations."""
    print("\n=== CHECKING cuBLAS/cuDNN WORKSPACES ===\n")

    # cuBLAS and cuDNN allocate internal workspaces
    # These can be controlled with environment variables

    print("Environment variables that affect CUDA workspaces:")
    env_vars = [
        'CUBLAS_WORKSPACE_CONFIG',
        'CUDA_MODULE_LOADING',
        'PYTORCH_CUDA_ALLOC_CONF',
    ]
    for var in env_vars:
        val = os.environ.get(var, 'not set')
        print(f"  {var}: {val}")

    # Try to get cuBLAS info
    try:
        # PyTorch doesn't expose cuBLAS workspace directly
        # But we can check memory stats
        stats = torch.cuda.memory_stats()
        print("\nRelevant memory stats:")
        for key in sorted(stats.keys()):
            if 'caching' in key.lower() or 'pool' in key.lower():
                print(f"  {key}: {stats[key]}")
    except Exception as e:
        print(f"Stats error: {e}")


def check_triton_cache():
    """Check triton kernel cache."""
    print("\n=== CHECKING TRITON CACHE ===\n")

    try:
        import triton
        # Triton caches compiled kernels
        cache_dir = os.path.expanduser("~/.triton/cache")
        if os.path.exists(cache_dir):
            import subprocess
            result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
            print(f"Triton cache on disk: {result.stdout.strip()}")

        # Check if triton has any GPU memory
        # Triton kernels are loaded into GPU memory when used
        print("Note: Triton kernels stay in GPU memory until process exits")

    except Exception as e:
        print(f"Triton check error: {e}")


def try_release_cublas():
    """Try to release cuBLAS workspace."""
    print("\n=== TRYING TO RELEASE cuBLAS WORKSPACE ===\n")

    log_mem("before")

    # Try to clear cuBLAS workspace
    # This is a hack but might work
    try:
        # Create a small matmul to initialize cuBLAS
        a = torch.randn(10, 10, device='cuda')
        b = torch.randn(10, 10, device='cuda')
        c = torch.mm(a, b)
        del a, b, c

        # Try to hint that we want minimal workspace
        torch.backends.cuda.preferred_linalg_library('cusolver')

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"cuBLAS release error: {e}")

    log_mem("after cuBLAS hint")


def try_release_nccl():
    """Try to release all NCCL resources."""
    print("\n=== TRYING TO RELEASE NCCL ===\n")

    log_mem("before")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            print("Destroying process group...")
            dist.destroy_process_group()
        else:
            print("Process group not initialized")
    except Exception as e:
        print(f"NCCL release error: {e}")

    # Also try to release NCCL native resources
    try:
        # PyTorch's NCCL backend keeps native resources
        import torch.distributed.distributed_c10d as c10d
        if hasattr(c10d, '_world'):
            c10d._world = None
        if hasattr(c10d, '_pg_map'):
            c10d._pg_map.clear()
        if hasattr(c10d, '_pg_names'):
            c10d._pg_names.clear()
    except Exception as e:
        print(f"NCCL native release error: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after NCCL release")


def try_force_deallocation():
    """Try to force PyTorch to release all allocations."""
    print("\n=== FORCE DEALLOCATION ATTEMPTS ===\n")

    log_mem("before")

    # Attempt 1: Reset allocator
    print("\n--- Attempt 1: Reset allocator settings ---")
    try:
        # Force garbage collection threshold to 0 to release all cached memory
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.0")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_mem("after GC threshold 0")

        # Reset to default
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
    except Exception as e:
        print(f"Allocator settings error: {e}")

    # Attempt 2: Release all cached blocks
    print("\n--- Attempt 2: IPC collect ---")
    try:
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        log_mem("after ipc_collect")
    except Exception as e:
        print(f"IPC collect error: {e}")

    # Attempt 3: Synchronize all streams
    print("\n--- Attempt 3: Sync all streams ---")
    try:
        torch.cuda.synchronize()
        # Get current stream and sync
        stream = torch.cuda.current_stream()
        stream.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        log_mem("after stream sync")
    except Exception as e:
        print(f"Stream sync error: {e}")

    # Attempt 4: Clear autograd state
    print("\n--- Attempt 4: Clear autograd ---")
    try:
        # Clear any saved tensors
        torch.autograd.set_grad_enabled(False)
        gc.collect()
        torch.cuda.empty_cache()
        torch.autograd.set_grad_enabled(True)
        log_mem("after autograd clear")
    except Exception as e:
        print(f"Autograd clear error: {e}")


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("DEEP MEMORY ANALYSIS - FINDING THE 7.5GB LEAK")
    print("=" * 70)

    baseline = log_mem("baseline")

    # Load and cleanup model
    print("\n=== Loading model ===")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    log_mem("after load")

    # Quick inference
    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"Output: {out[0].outputs[0].text.strip()[:30]}")

    # Cleanup
    print("\n=== Standard cleanup ===")
    freed = full_cleanup(llm)
    llm = None
    print(f"Freed: {freed:.1f}GB")
    log_mem("after cleanup")

    # Now analyze what's left
    print("\n" + "=" * 70)
    print("ANALYZING REMAINING 7.5GB")
    print("=" * 70)

    # Memory snapshot
    analyze_memory_snapshot()

    # Check specific subsystems
    check_cublas_workspace()
    check_triton_cache()

    # Try to release
    try_release_cublas()
    try_release_nccl()
    try_force_deallocation()

    # Final state
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    final = log_mem("final")

    print(f"\nRemaining above baseline: {final['used_gb'] - baseline['used_gb']:.2f}GB")
    print(f"Allocated (PyTorch): {final['allocated_gb']:.2f}GB")
    print(f"Reserved (caching): {final['reserved_gb']:.2f}GB")


if __name__ == "__main__":
    main()
