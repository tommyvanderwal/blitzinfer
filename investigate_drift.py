#!/usr/bin/env python3
"""Deep investigation of the 0.6GB per-switch memory drift.

This script investigates where the leaked memory is:
1. PyTorch allocated memory (model weights, tensors)
2. PyTorch reserved memory (caching allocator)
3. CUDA driver memory (outside PyTorch)
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_detailed_memory():
    """Get detailed memory breakdown."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    # Driver memory = total used - reserved (memory outside PyTorch's control)
    cuda_used = total - free
    driver_mem = cuda_used - reserved

    return {
        'total_gb': total / 1024**3,
        'free_gb': free / 1024**3,
        'cuda_used_gb': cuda_used / 1024**3,
        'allocated_gb': allocated / 1024**3,
        'reserved_gb': reserved / 1024**3,
        'driver_gb': driver_mem / 1024**3,
        'allocator_overhead_gb': (reserved - allocated) / 1024**3,
    }


def log_memory(label):
    """Log detailed memory breakdown."""
    m = get_detailed_memory()
    print(f"\n[{label}]")
    print(f"  CUDA used:          {m['cuda_used_gb']:.3f} GB")
    print(f"  ├─ PyTorch reserved: {m['reserved_gb']:.3f} GB")
    print(f"  │   ├─ allocated:    {m['allocated_gb']:.3f} GB")
    print(f"  │   └─ overhead:     {m['allocator_overhead_gb']:.3f} GB")
    print(f"  └─ Driver memory:    {m['driver_gb']:.3f} GB")
    return m


def compare_memory(label, before, after):
    """Compare memory states and show deltas."""
    print(f"\n[DELTA: {label}]")
    print(f"  CUDA used:       {after['cuda_used_gb'] - before['cuda_used_gb']:+.3f} GB")
    print(f"  ├─ reserved:     {after['reserved_gb'] - before['reserved_gb']:+.3f} GB")
    print(f"  │   ├─ alloc:    {after['allocated_gb'] - before['allocated_gb']:+.3f} GB")
    print(f"  │   └─ overhead: {after['allocator_overhead_gb'] - before['allocator_overhead_gb']:+.3f} GB")
    print(f"  └─ driver:       {after['driver_gb'] - before['driver_gb']:+.3f} GB")


def investigate_tensors():
    """List all GPU tensors that exist."""
    print("\n[GPU TENSORS]")
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                count += 1
                total_bytes += obj.numel() * obj.element_size()
        except Exception:
            pass
    print(f"  Total GPU tensors: {count}")
    print(f"  Total size: {total_bytes / 1024**3:.3f} GB")
    return count, total_bytes


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("DEEP MEMORY DRIFT INVESTIGATION")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print()

    # Initial state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = log_memory("BASELINE")
    investigate_tensors()

    states = [baseline]

    for round_num in range(3):
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

        # Quick inference
        out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
        _ = out[0].outputs[0].text

        after_load = log_memory("after load + inference")
        compare_memory("load", states[-1], after_load)

        # Cleanup
        print("\n--- Cleanup ---")
        freed = full_cleanup(llm, nuclear=True)
        llm = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after_cleanup = log_memory("after cleanup")
        compare_memory("cleanup", after_load, after_cleanup)
        compare_memory(f"round {round_num+1} total", states[-1], after_cleanup)

        # Check remaining tensors
        tensor_count, tensor_bytes = investigate_tensors()

        # Try to identify what's holding memory
        print("\n--- Investigating remaining allocations ---")

        # Check PyTorch internal caches
        print("\n  PyTorch internal state:")
        try:
            # Check if there are any cached kernels
            import torch._dynamo as dynamo
            cache_size = len(dynamo.utils.cache_size())
            print(f"    dynamo cache entries: {cache_size}")
        except Exception as e:
            print(f"    dynamo cache: {e}")

        # Check NCCL state
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                print(f"    distributed initialized: True")
            else:
                print(f"    distributed initialized: False")
        except Exception as e:
            print(f"    distributed check: {e}")

        # Check for vLLM singletons
        print("\n  vLLM singletons:")
        try:
            from vllm.model_executor.layers.fused_moe import workspace
            if hasattr(workspace, 'WorkspaceManager'):
                mgr = workspace.WorkspaceManager
                if hasattr(mgr, '_instance') and mgr._instance is not None:
                    print(f"    WorkspaceManager: ACTIVE")
                else:
                    print(f"    WorkspaceManager: cleared")
        except Exception as e:
            print(f"    WorkspaceManager: {e}")

        try:
            from vllm.model_executor.layers import rotary_embedding
            if hasattr(rotary_embedding, '_ROPE_DICT'):
                rope_entries = len(rotary_embedding._ROPE_DICT)
                print(f"    ROPE cache entries: {rope_entries}")
        except Exception as e:
            print(f"    ROPE cache: {e}")

        states.append(after_cleanup)

        print(f"\n  Cumulative drift from baseline: {after_cleanup['cuda_used_gb'] - baseline['cuda_used_gb']:+.3f} GB")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    final = states[-1]
    total_drift = final['cuda_used_gb'] - baseline['cuda_used_gb']
    driver_drift = final['driver_gb'] - baseline['driver_gb']
    reserved_drift = final['reserved_gb'] - baseline['reserved_gb']
    allocated_drift = final['allocated_gb'] - baseline['allocated_gb']

    print(f"\nTotal drift: {total_drift:+.3f} GB")
    print(f"  ├─ Driver memory:   {driver_drift:+.3f} GB ({100*driver_drift/max(total_drift, 0.001):.0f}%)")
    print(f"  └─ PyTorch reserved: {reserved_drift:+.3f} GB ({100*reserved_drift/max(total_drift, 0.001):.0f}%)")
    print(f"       └─ allocated:  {allocated_drift:+.3f} GB")

    if driver_drift > reserved_drift:
        print("\n>>> DRIFT IS PRIMARILY IN DRIVER MEMORY <<<")
        print("    This is NCCL buffers, CUDA contexts, or driver-level allocations")
        print("    These cannot be freed from Python - they persist for the process lifetime")
    else:
        print("\n>>> DRIFT IS PRIMARILY IN PYTORCH ALLOCATOR <<<")
        print("    This is fragmentation or unreleased cached blocks")
        print("    Try more aggressive allocator cleanup")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
