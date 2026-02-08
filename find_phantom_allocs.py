#!/usr/bin/env python3
"""Find phantom allocations - memory allocated but no visible tensors.

The investigation showed 0.47GB allocated per round but 0 visible GPU tensors.
This script uses torch.cuda.memory_snapshot() to find the source.
"""

import os
import gc
import sys

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


def analyze_snapshot():
    """Analyze memory snapshot to find allocation sources."""
    try:
        snapshot = torch.cuda.memory_snapshot()
        if not snapshot:
            print("  No allocations in snapshot")
            return

        # Group by allocation category
        by_category = {}
        total_allocated = 0
        for block in snapshot:
            if block['state'] == 'active_allocated':
                size = block['total_size']
                total_allocated += size

                # Get stack frames if available
                frames = block.get('frames', [])
                if frames:
                    # Use first meaningful frame as category
                    for frame in frames:
                        filename = frame.get('filename', 'unknown')
                        if 'torch' not in filename and 'python' not in filename:
                            key = f"{filename}:{frame.get('line', '?')}"
                            break
                    else:
                        key = frames[0].get('filename', 'unknown')
                else:
                    key = 'no_frames'

                if key not in by_category:
                    by_category[key] = {'count': 0, 'size': 0}
                by_category[key]['count'] += 1
                by_category[key]['size'] += size

        print(f"\n  Total allocated in snapshot: {total_allocated / 1024**3:.3f} GB")
        print(f"  Top allocation sources:")
        sorted_cats = sorted(by_category.items(), key=lambda x: x[1]['size'], reverse=True)
        for cat, info in sorted_cats[:10]:
            print(f"    {info['size']/1024**2:.1f} MB ({info['count']} blocks) - {cat}")

    except Exception as e:
        print(f"  Snapshot analysis failed: {e}")


def find_storage_objects():
    """Find all CUDA storage objects in memory."""
    print("\n  Searching for CUDA storage objects...")
    storages = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.UntypedStorage):
                if obj.device.type == 'cuda' and obj.size() > 0:
                    storages.append({
                        'size': obj.size() * obj.element_size() if hasattr(obj, 'element_size') else obj.size(),
                        'id': id(obj),
                    })
        except Exception:
            pass

    if storages:
        total = sum(s['size'] for s in storages)
        print(f"  Found {len(storages)} CUDA storages, total {total/1024**2:.1f} MB")
        for s in sorted(storages, key=lambda x: x['size'], reverse=True)[:5]:
            print(f"    Storage {s['id']}: {s['size']/1024**2:.1f} MB")
    else:
        print("  No CUDA storage objects found")


def check_nccl_allocations():
    """Check for NCCL-related allocations."""
    print("\n  Checking NCCL state...")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            print("    torch.distributed: initialized")
            # Get info about process groups
            if hasattr(dist, 'group'):
                print(f"    World size: {dist.get_world_size()}")
        else:
            print("    torch.distributed: not initialized")
    except Exception as e:
        print(f"    torch.distributed check: {e}")

    # Check vLLM parallel state
    try:
        from vllm.distributed import parallel_state
        groups = []
        for attr in dir(parallel_state):
            if 'GROUP' in attr:
                val = getattr(parallel_state, attr, None)
                if val is not None:
                    groups.append(attr)
        if groups:
            print(f"    vLLM parallel groups: {groups}")
        else:
            print("    vLLM parallel groups: all None")
    except Exception as e:
        print(f"    vLLM parallel state: {e}")


def check_cublas_cudnn():
    """Check cuBLAS/cuDNN workspace state."""
    print("\n  Checking cuBLAS/cuDNN...")
    print(f"    cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"    cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"    cuDNN allow_tf32: {torch.backends.cudnn.allow_tf32}")

    # Try to get cuBLAS workspace size
    try:
        # Do a matmul to force cuBLAS init
        a = torch.randn(128, 128, device='cuda')
        b = torch.randn(128, 128, device='cuda')
        _ = torch.mm(a, b)
        del a, b, _
        torch.cuda.synchronize()
        print(f"    cuBLAS: initialized")
    except Exception as e:
        print(f"    cuBLAS check: {e}")


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("PHANTOM ALLOCATION INVESTIGATION")
    print("=" * 70)

    # Enable memory tracking for snapshots
    torch.cuda.memory._record_memory_history(max_entries=100000)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_memory()
    print(f"\n[BASELINE] allocated: {baseline['allocated_gb']:.3f} GB")

    for round_num in range(2):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}")
        print("=" * 70)

        # Load model
        print("\n--- Loading model ---")
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

        after_load = get_memory()
        print(f"[after load] allocated: {after_load['allocated_gb']:.3f} GB")

        # Cleanup
        print("\n--- Cleanup ---")
        freed = full_cleanup(llm, nuclear=True)
        llm = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after_cleanup = get_memory()
        print(f"[after cleanup] allocated: {after_cleanup['allocated_gb']:.3f} GB")

        # Investigate what's left
        print("\n--- Investigating phantom allocations ---")
        analyze_snapshot()
        find_storage_objects()
        check_nccl_allocations()

    # Disable memory history
    torch.cuda.memory._record_memory_history(enabled=None)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    final = get_memory()
    drift = final['allocated_gb'] - baseline['allocated_gb']
    print(f"\nTotal allocated drift: {drift:+.3f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
