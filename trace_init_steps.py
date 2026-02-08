#!/usr/bin/env python3
"""Trace memory during vLLM initialization to find where 160MB blocks come from.

Based on trace_160mb_source.py, we know:
- First 160MB block appears in GPUModelRunner.__init__
- Before model weights are loaded
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
# Enable workspace debug
os.environ['VLLM_DEBUG_WORKSPACE'] = '1'

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
                    blocks.append({
                        'size_mb': size / 1024**2,
                        'address': block.get('address', 0),
                    })
    return len(blocks), blocks


def trace_init():
    """Trace vLLM initialization step by step."""
    print("=" * 70)
    print("TRACING vLLM INITIALIZATION")
    print("=" * 70)

    # Enable memory tracing
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=500000
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def log_state(step):
        mem = get_allocated_mb()
        n, blocks = count_large_blocks()
        print(f"  [{step}] {mem:.1f} MB, {n} large blocks")
        if blocks:
            for b in blocks[-3:]:
                print(f"    {b['size_mb']:.1f} MB at 0x{b['address']:x}")
        return n

    n_prev = log_state("BASELINE")

    # Step 1: Import vLLM core
    print("\n--- Step 1: Import vLLM ---")
    from vllm import LLM
    n_curr = log_state("After import")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 2: Import parallel state
    print("\n--- Step 2: Import parallel_state ---")
    from vllm.distributed import parallel_state
    n_curr = log_state("After parallel_state import")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 3: Import attention backend
    print("\n--- Step 3: Import attention backends ---")
    from vllm.v1.attention.backends import flash_attn
    n_curr = log_state("After flash_attn import")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 4: Import workspace manager
    print("\n--- Step 4: Import workspace manager ---")
    from vllm.v1.worker.workspace import init_workspace_manager
    n_curr = log_state("After workspace import")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 5: Initialize workspace manager
    print("\n--- Step 5: Initialize workspace manager ---")
    init_workspace_manager(torch.device("cuda"))
    n_curr = log_state("After init_workspace_manager")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 6: Initialize torch.distributed
    print("\n--- Step 6: Initialize torch.distributed ---")
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:45678',
            world_size=1,
            rank=0,
        )
    n_curr = log_state("After dist.init_process_group")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 7: Do an all_reduce
    print("\n--- Step 7: NCCL all_reduce ---")
    tensor = torch.ones(1024, device='cuda')
    dist.all_reduce(tensor)
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    n_curr = log_state("After all_reduce")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 8: Create Flash Attention metadata
    print("\n--- Step 8: Create FlashAttentionMetadata ---")
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        # Simulate creating metadata
        metadata = FlashAttentionMetadata(
            num_actual_tokens=1024,
            max_query_len=512,
            query_start_loc=torch.zeros(2, dtype=torch.int32, device='cuda'),
            max_seq_len=1024,
            seq_lens=torch.ones(1, dtype=torch.int32, device='cuda') * 1024,
            block_table=torch.zeros((1, 64), dtype=torch.int32, device='cuda'),
            slot_mapping=torch.zeros(1024, dtype=torch.int64, device='cuda'),
        )
        del metadata
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Could not create FlashAttentionMetadata: {e}")

    n_curr = log_state("After FlashAttentionMetadata")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) ***")
    n_prev = n_curr

    # Step 9: Call flash_attn_with_kvcache (the actual attention op)
    print("\n--- Step 9: Flash Attention kernel call ---")
    try:
        # This is what vLLM uses internally
        from flash_attn import flash_attn_with_kvcache

        # Simulate attention with KV cache
        q = torch.randn(1, 1, 40, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1024, 40, 128, device='cuda', dtype=torch.bfloat16)  # KV cache
        v = torch.randn(1024, 40, 128, device='cuda', dtype=torch.bfloat16)

        # Reshape k, v for paged KV cache format
        k_cache = k.view(1, 1024, 40, 128).transpose(1, 2)
        v_cache = v.view(1, 1024, 40, 128).transpose(1, 2)

        out = flash_attn_with_kvcache(
            q.view(1, 1, 40, 128),
            k_cache,
            v_cache,
            causal=True,
        )
        del q, k, v, k_cache, v_cache, out
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  flash_attn_with_kvcache error: {e}")
        # Try basic flash attention
        try:
            from vllm._custom_ops import paged_attention_v1
            print("  Trying vLLM paged_attention_v1...")

            # Setup for paged attention
            query = torch.randn(1, 40, 128, device='cuda', dtype=torch.bfloat16)
            key_cache = torch.randn(64, 1, 16, 40, 128, device='cuda', dtype=torch.bfloat16)  # (num_blocks, 1, block_size, num_heads, head_dim)
            value_cache = torch.randn(64, 1, 16, 40, 128, device='cuda', dtype=torch.bfloat16)
            output = torch.empty_like(query)

            paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads=40,
                scale=1.0 / (128 ** 0.5),
                block_tables=torch.zeros((1, 64), dtype=torch.int32, device='cuda'),
                seq_lens=torch.tensor([1024], dtype=torch.int32, device='cuda'),
                block_size=16,
                max_seq_len=1024,
                alibi_slopes=None,
                kv_cache_dtype='auto',
            )
            del query, key_cache, value_cache, output
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e2:
            print(f"  paged_attention_v1 error: {e2}")

    n_curr = log_state("After flash_attn kernel")
    if n_curr > n_prev:
        print("  *** NEW LARGE BLOCK(S) - FOUND THE SOURCE! ***")
    n_prev = n_curr

    # Step 10: Actually create the LLM
    print("\n--- Step 10: Create LLM ---")
    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )

    n_curr = log_state("After LLM creation")

    # Cleanup
    from blitzinfer.engine.cleanup import full_cleanup
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    n_final, blocks_final = count_large_blocks()
    final_mem = get_allocated_mb()
    print(f"Memory: {final_mem:.1f} MB, {n_final} large blocks")
    if blocks_final:
        print("Remaining large blocks (LEAKED):")
        for b in blocks_final:
            print(f"  {b['size_mb']:.1f} MB at 0x{b['address']:x}")

    torch.cuda.memory._record_memory_history(enabled=None)

    return 0


if __name__ == "__main__":
    sys.exit(trace_init())
