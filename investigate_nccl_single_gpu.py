#!/usr/bin/env python3
"""Investigate if NCCL is being used unnecessarily in single-GPU mode.

The 3x160MB blocks might be NCCL communication buffers that shouldn't exist
for single-GPU inference.
"""

import os
import gc
import sys

# Try to disable NCCL entirely for single-GPU
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
# Disable distributed
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
    }


def check_nccl_state():
    """Check if NCCL/distributed is initialized."""
    print("\n=== NCCL/Distributed State ===")

    import torch.distributed as dist
    print(f"  dist.is_initialized(): {dist.is_initialized()}")
    if dist.is_initialized():
        print(f"  dist.get_world_size(): {dist.get_world_size()}")
        print(f"  dist.get_backend(): {dist.get_backend()}")

    # Check vLLM parallel state
    try:
        from vllm.distributed import parallel_state
        print(f"\n  vLLM parallel_state:")
        for attr in ['_TP_DEVICE_GROUP', '_TP_CPU_GROUP', '_PP_DEVICE_GROUP']:
            val = getattr(parallel_state, attr, 'N/A')
            print(f"    {attr}: {val}")

        # Check if model parallel is needed
        from vllm.config import ParallelConfig
        print(f"\n  ParallelConfig defaults:")
        pc = ParallelConfig()
        print(f"    tensor_parallel_size: {pc.tensor_parallel_size}")
        print(f"    pipeline_parallel_size: {pc.pipeline_parallel_size}")
    except Exception as e:
        print(f"  vLLM parallel state check failed: {e}")


def test_with_gloo_backend():
    """Test if using GLOO instead of NCCL reduces memory."""
    print("\n=== TEST: GLOO backend instead of NCCL ===")

    # Try to force GLOO for CPU-based distributed
    os.environ['NCCL_DEBUG'] = 'WARN'

    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_memory()
    print(f"\n[BASELINE] allocated: {baseline['allocated_gb']:.3f} GB")

    # Load model
    print("\n--- Loading model ---")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Explicit single GPU
    )

    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    _ = out[0].outputs[0].text

    check_nccl_state()

    # Cleanup
    print("\n--- Cleanup ---")
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after = get_memory()
    phantom = after['allocated_gb'] - baseline['allocated_gb']
    print(f"\n[after cleanup] allocated: {after['allocated_gb']:.3f} GB")
    print(f"[phantom] {phantom:.3f} GB")

    return phantom


def test_disable_nccl_init():
    """Test if we can prevent NCCL initialization entirely."""
    print("\n=== TEST: Disable NCCL initialization ===")

    # These env vars might help
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback
    os.environ['NCCL_P2P_DISABLE'] = '1'      # Disable P2P
    os.environ['NCCL_SHM_DISABLE'] = '1'      # Disable shared memory

    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_memory()
    print(f"\n[BASELINE] allocated: {baseline['allocated_gb']:.3f} GB")

    # Load model
    print("\n--- Loading model (NCCL disabled) ---")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    _ = out[0].outputs[0].text

    check_nccl_state()

    # Cleanup
    print("\n--- Cleanup ---")
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after = get_memory()
    phantom = after['allocated_gb'] - baseline['allocated_gb']
    print(f"\n[after cleanup] allocated: {after['allocated_gb']:.3f} GB")
    print(f"[phantom] {phantom:.3f} GB")

    return phantom


def analyze_160mb_blocks():
    """Analyze what the 160MB blocks might be."""
    print("\n=== ANALYSIS: 160MB Block Sources ===")

    print("""
Known sources of ~160MB GPU allocations:

1. NCCL communicator buffers (~128-256MB each)
   - Created when torch.distributed is initialized with NCCL backend
   - One per GPU per communicator
   - Should NOT exist for single-GPU TP=1 workloads

2. cuBLAS workspace (~128-512MB)
   - Created on first matmul operation
   - Persists for session lifetime
   - Size depends on matrix dimensions and CUDA version

3. cuDNN workspace (~64-256MB)
   - Created on first conv/attention operation
   - Persists for session lifetime
   - Can be controlled via cudnn.benchmark settings

4. Flash Attention workspace (~50-200MB)
   - Used for attention computation
   - May persist between calls

5. Triton kernel workspace (~50-100MB per kernel)
   - Autotune cache may hold compiled kernels
   - Should be clearable via triton cache reset

Let's check which is most likely by looking at when they're allocated:
""")

    # Enable memory tracing before any operations
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=100000
    )

    gc.collect()
    torch.cuda.empty_cache()
    baseline = get_memory()
    print(f"After reset: {baseline['allocated_gb']:.3f} GB")

    # Step 1: Just import torch.distributed
    print("\n1. After importing torch.distributed...")
    import torch.distributed as dist
    step1 = get_memory()
    print(f"   Allocated: {step1['allocated_gb']:.3f} GB (+{(step1['allocated_gb']-baseline['allocated_gb'])*1024:.1f} MB)")

    # Step 2: Do a simple matmul (triggers cuBLAS)
    print("\n2. After first matmul (cuBLAS init)...")
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = torch.mm(a, b)
    del a, b, c
    gc.collect()
    torch.cuda.empty_cache()
    step2 = get_memory()
    print(f"   Allocated: {step2['allocated_gb']:.3f} GB (+{(step2['allocated_gb']-step1['allocated_gb'])*1024:.1f} MB)")

    # Step 3: Do Flash Attention if available
    print("\n3. After Flash Attention...")
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
        out = flash_attn_func(q, k, v)
        del q, k, v, out
        gc.collect()
        torch.cuda.empty_cache()
        step3 = get_memory()
        print(f"   Allocated: {step3['allocated_gb']:.3f} GB (+{(step3['allocated_gb']-step2['allocated_gb'])*1024:.1f} MB)")
    except ImportError:
        print("   Flash Attention not available")
        step3 = step2

    # Step 4: Initialize NCCL (if we can do it manually)
    print("\n4. After NCCL init...")
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:12345',
                world_size=1,
                rank=0,
            )
            step4 = get_memory()
            print(f"   Allocated: {step4['allocated_gb']:.3f} GB (+{(step4['allocated_gb']-step3['allocated_gb'])*1024:.1f} MB)")

            # Check snapshot for new allocations
            snapshot = torch.cuda.memory._snapshot()
            large_blocks = []
            if snapshot and 'segments' in snapshot:
                for segment in snapshot['segments']:
                    for block in segment.get('blocks', []):
                        if block.get('state') == 'active_allocated':
                            size = block.get('size', block.get('requested_size', 0))
                            if size > 100 * 1024 * 1024:  # > 100MB
                                large_blocks.append(size)
            if large_blocks:
                print(f"   Large blocks: {[f'{s/1024**2:.1f}MB' for s in large_blocks]}")

            # Clean up
            dist.destroy_process_group()
        except Exception as e:
            print(f"   NCCL init failed: {e}")
            step4 = step3
    else:
        print("   Already initialized")
        step4 = step3

    torch.cuda.memory._record_memory_history(enabled=None)


def main():
    print("=" * 70)
    print("INVESTIGATING NCCL/DISTRIBUTED IN SINGLE-GPU MODE")
    print("=" * 70)

    # First, analyze what creates 160MB blocks
    analyze_160mb_blocks()

    return 0


if __name__ == "__main__":
    sys.exit(main())
