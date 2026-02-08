#!/usr/bin/env python3
"""Test if using Flash Attention 2 vs 3 affects the 160MB blocks.

FA3 uses tiled intermediate buffers that might be the source of the leak.
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
# Force Flash Attention version 2
os.environ['VLLM_ATTENTION_CONFIG'] = 'flash_attn_version=2'

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


def test_fa_version():
    """Test model loading with FA2."""
    print("=" * 70)
    print("TESTING FLASH ATTENTION VERSION")
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

    # Load model with FA2
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print(f"\n--- Loading model (forcing FA2) ---")
    print(f"VLLM_ATTENTION_CONFIG = {os.environ.get('VLLM_ATTENTION_CONFIG', 'not set')}")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Generate to trigger attention
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    _ = out[0].outputs[0].text

    after_load = get_allocated_mb()
    n_after, _ = count_large_blocks()
    print(f"[After load/generate] {after_load:.1f} MB, {n_after} large blocks")

    # Cleanup
    from blitzinfer.engine.cleanup import full_cleanup
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    final = get_allocated_mb()
    n_final, blocks_final = count_large_blocks()
    print(f"\n[FINAL] {final:.1f} MB, {n_final} large blocks")

    if blocks_final:
        print("Remaining large blocks:")
        for b in blocks_final:
            print(f"  {b['size_mb']:.1f} MB at 0x{b['address']:x}")

    torch.cuda.memory._record_memory_history(enabled=None)

    return n_final


if __name__ == "__main__":
    n = test_fa_version()
    print(f"\nResult: {n} large blocks remaining")
    sys.exit(0 if n == 0 else 1)
