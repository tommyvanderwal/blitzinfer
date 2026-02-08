#!/usr/bin/env python3
"""Trace exactly what in GPUModelRunner.__init__ creates the first 160MB block."""

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
        return 0

    count = 0
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', block.get('requested_size', 0))
                if size > min_size_mb * 1024 * 1024:
                    count += 1
    return count


def trace_model_runner_init():
    """Trace GPUModelRunner.__init__ step by step."""
    print("=" * 70)
    print("TRACING GPUModelRunner.__init__")
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
    n_baseline = count_large_blocks()
    print(f"\n[BASELINE] {baseline:.1f} MB, {n_baseline} large blocks")

    # Import vLLM modules and instrument them
    print("\n--- Setting up instrumentation ---")

    # Import the runner module
    from vllm.v1.worker import gpu_model_runner as runner_module

    # Save original __init__
    original_init = runner_module.GPUModelRunner.__init__

    # Track what happens inside __init__
    def instrumented_init(self, vllm_config, *args, **kwargs):
        """Instrumented init to trace memory allocations."""
        print("\n  [GPUModelRunner.__init__ ENTRY]")
        mem_before = get_allocated_mb()
        n_before = count_large_blocks()
        print(f"    Memory: {mem_before:.1f} MB, {n_before} large blocks")

        # Call parent __init__ step by step
        from vllm.config import set_current_vllm_config

        # Set config
        print("\n  Step 1: Setting vllm_config...")
        with set_current_vllm_config(vllm_config):
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.lora_config = vllm_config.lora_config
            self.load_config = vllm_config.load_config
            self.parallel_config = vllm_config.parallel_config
            self.scheduler_config = vllm_config.scheduler_config
            self.speculative_config = vllm_config.speculative_config
            self.prompt_adapter_config = vllm_config.prompt_adapter_config
            self.observability_config = vllm_config.observability_config

            mem_after_config = get_allocated_mb()
            n_after_config = count_large_blocks()
            print(f"    After config: {mem_after_config:.1f} MB, {n_after_config} large blocks")
            print(f"    Delta: {mem_after_config - mem_before:.1f} MB")

            # Device setup
            print("\n  Step 2: Device setup...")
            self.device = torch.device("cuda")
            self.pin_memory = True
            self.dtype = self.model_config.dtype

            mem_after_device = get_allocated_mb()
            n_after_device = count_large_blocks()
            print(f"    After device: {mem_after_device:.1f} MB, {n_after_device} large blocks")
            print(f"    Delta: {mem_after_device - mem_after_config:.1f} MB")

            # Block manager setup
            print("\n  Step 3: Block manager vars...")
            self.block_size = self.cache_config.block_size
            self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
            self.max_num_reqs = self.scheduler_config.max_num_seqs

            mem_after_block = get_allocated_mb()
            n_after_block = count_large_blocks()
            print(f"    After block vars: {mem_after_block:.1f} MB, {n_after_block} large blocks")

            # Sliding window
            print("\n  Step 4: Sliding window / num layers...")
            self.sliding_window = self.model_config.get_sliding_window()
            self.num_hidden_layers = self.model_config.get_num_layers(self.parallel_config)
            self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

            mem_after_sliding = get_allocated_mb()
            n_after_sliding = count_large_blocks()
            print(f"    After sliding window: {mem_after_sliding:.1f} MB, {n_after_sliding} large blocks")

            # This is where things get interesting - attention backend setup
            print("\n  Step 5: Attention backend setup...")
            from vllm.attention import AttentionType, get_attn_backend

            mem_before_attn = get_allocated_mb()
            n_before_attn = count_large_blocks()

            # Get attention backend (this may allocate!)
            attn_backend = get_attn_backend(
                self.model_config.get_head_size(),
                self.model_config.dtype,
                self.cache_config.cache_dtype,
                self.block_size,
                self.model_config.is_attention_free,
                use_mla=self.model_config.use_mla,
                num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            )

            mem_after_attn = get_allocated_mb()
            n_after_attn = count_large_blocks()
            print(f"    After get_attn_backend: {mem_after_attn:.1f} MB, {n_after_attn} large blocks")
            print(f"    Delta: {mem_after_attn - mem_before_attn:.1f} MB")
            if n_after_attn > n_before_attn:
                print(f"    *** NEW LARGE BLOCK(S) ALLOCATED BY ATTENTION BACKEND ***")

            # Multi-modal setup
            print("\n  Step 6: Multi-modal setup...")
            mem_before_mm = get_allocated_mb()
            n_before_mm = count_large_blocks()

            mm_registry = self.vllm_config.model_config.mm_processor_kwargs
            # Just check, don't actually initialize

            mem_after_mm = get_allocated_mb()
            n_after_mm = count_large_blocks()
            print(f"    After mm check: {mem_after_mm:.1f} MB, {n_after_mm} large blocks")

            # Now let the original init finish
            print("\n  Calling original __init__ to complete setup...")

        # Just call original init - we've traced enough
        return original_init(self, vllm_config, *args, **kwargs)

    # Don't actually use the instrumented version - it's too complex
    # Instead, let's just trace the key operations

    # Import what we need
    from vllm.attention import get_attn_backend
    from vllm.config import ModelConfig

    print("\n--- Testing attention backend allocation ---")
    mem_before = get_allocated_mb()
    n_before = count_large_blocks()
    print(f"  Before: {mem_before:.1f} MB, {n_before} large blocks")

    # Get attention backend
    try:
        attn_backend = get_attn_backend(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype='auto',
            block_size=16,
            is_attention_free=False,
            use_mla=False,
            num_kv_heads=8,
        )
        print(f"  Got attention backend: {attn_backend}")
    except Exception as e:
        print(f"  Attention backend error: {e}")

    mem_after = get_allocated_mb()
    n_after = count_large_blocks()
    print(f"  After: {mem_after:.1f} MB, {n_after} large blocks")
    print(f"  Delta: {mem_after - mem_before:.1f} MB")

    # Now try Flash Attention directly
    print("\n--- Testing Flash Attention allocation ---")
    mem_before = get_allocated_mb()
    n_before = count_large_blocks()
    print(f"  Before: {mem_before:.1f} MB, {n_before} large blocks")

    try:
        from flash_attn import flash_attn_func
        # Small attention to trigger initialization
        q = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 8, 16, 128, device='cuda', dtype=torch.bfloat16)

        out = flash_attn_func(q, k, v, causal=True)
        del q, k, v, out

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after = count_large_blocks()
        print(f"  After Flash Attention: {mem_after:.1f} MB, {n_after} large blocks")
        print(f"  Delta: {mem_after - mem_before:.1f} MB")
        if n_after > n_before:
            print(f"  *** Flash Attention created {n_after - n_before} large block(s) ***")
    except Exception as e:
        print(f"  Flash Attention error: {e}")

    # Test cuBLAS with large matrices
    print("\n--- Testing cuBLAS large matrix allocation ---")
    mem_before = get_allocated_mb()
    n_before = count_large_blocks()
    print(f"  Before: {mem_before:.1f} MB, {n_before} large blocks")

    # Large matmul similar to what happens in transformers
    a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    c = torch.mm(a, b)
    del a, b, c

    gc.collect()
    torch.cuda.empty_cache()

    mem_after = get_allocated_mb()
    n_after = count_large_blocks()
    print(f"  After large matmul: {mem_after:.1f} MB, {n_after} large blocks")
    print(f"  Delta: {mem_after - mem_before:.1f} MB")
    if n_after > n_before:
        print(f"  *** Large matmul created {n_after - n_before} large block(s) ***")

    # Try Triton kernel
    print("\n--- Testing Triton kernel allocation ---")
    mem_before = get_allocated_mb()
    n_before = count_large_blocks()
    print(f"  Before: {mem_before:.1f} MB, {n_before} large blocks")

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def simple_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x * 2, mask=mask)

        x = torch.randn(65536, device='cuda')
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        simple_kernel[grid](x, y, x.numel(), BLOCK_SIZE=1024)
        del x, y

        gc.collect()
        torch.cuda.empty_cache()

        mem_after = get_allocated_mb()
        n_after = count_large_blocks()
        print(f"  After Triton kernel: {mem_after:.1f} MB, {n_after} large blocks")
        print(f"  Delta: {mem_after - mem_before:.1f} MB")
    except Exception as e:
        print(f"  Triton error: {e}")

    torch.cuda.memory._record_memory_history(enabled=None)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    final = get_allocated_mb()
    n_final = count_large_blocks()
    print(f"\nFinal: {final:.1f} MB, {n_final} large blocks (> 100MB)")

    return 0


if __name__ == "__main__":
    sys.exit(trace_model_runner_init())
