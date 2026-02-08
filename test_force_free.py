#!/usr/bin/env python3
"""Test using caching_allocator_delete to free phantom blocks.

We can get addresses from memory_snapshot() and try to free them directly.
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
import torch.cuda.memory as cuda_mem


def get_allocated_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def get_phantom_blocks(min_size_mb=100):
    """Get addresses of large blocks with no Python reference."""
    snapshot = torch.cuda.memory._snapshot()
    if not snapshot or 'segments' not in snapshot:
        return []

    phantom_blocks = []
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', block.get('requested_size', 0))
                if size > min_size_mb * 1024 * 1024:
                    # Get address - might be in block or segment
                    address = block.get('address', 0)
                    if address == 0:
                        # Some snapshots store address in segment
                        address = segment.get('address', 0)

                    phantom_blocks.append({
                        'address': address,
                        'size_mb': size / 1024**2,
                    })

    return phantom_blocks


def test_force_free():
    """Test freeing phantom blocks using caching_allocator_delete."""
    print("=" * 70)
    print("TEST: Force-free phantom blocks with caching_allocator_delete")
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
    print(f"\n[BASELINE] {baseline:.1f} MB allocated")

    # Load and cleanup a model to create phantom blocks
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

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

    # Cleanup
    print("\n--- Standard cleanup ---")
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after_cleanup = get_allocated_mb()
    phantom_blocks = get_phantom_blocks()
    print(f"[After cleanup] {after_cleanup:.1f} MB allocated")
    print(f"[Phantom blocks] {len(phantom_blocks)} large blocks")

    for b in phantom_blocks:
        print(f"  {b['size_mb']:.1f} MB at 0x{b['address']:x}")

    # Now try to force-free the phantom blocks
    print("\n--- Attempting force-free with caching_allocator_delete ---")

    for i, block in enumerate(phantom_blocks):
        addr = block['address']
        size = block['size_mb']

        if addr == 0:
            print(f"  Block {i}: Skipping (no address)")
            continue

        print(f"  Block {i}: Trying to free {size:.1f} MB at 0x{addr:x}...")

        try:
            # This is the key call - force delete the allocation
            cuda_mem.caching_allocator_delete(addr)
            print(f"    SUCCESS! Block freed")
        except Exception as e:
            print(f"    FAILED: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Check result
    final = get_allocated_mb()
    final_blocks = get_phantom_blocks()

    print(f"\n[FINAL] {final:.1f} MB allocated")
    print(f"[Remaining blocks] {len(final_blocks)} large blocks")

    if final_blocks:
        for b in final_blocks:
            print(f"  {b['size_mb']:.1f} MB at 0x{b['address']:x}")

    torch.cuda.memory._record_memory_history(enabled=None)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Before force-free: {after_cleanup:.1f} MB, {len(phantom_blocks)} phantom blocks")
    print(f"After force-free:  {final:.1f} MB, {len(final_blocks)} phantom blocks")
    print(f"Memory recovered:  {after_cleanup - final:.1f} MB")

    if len(final_blocks) == 0:
        print("\n*** SUCCESS: All phantom blocks freed! ***")
        return 0
    else:
        print(f"\n*** PARTIAL: {len(phantom_blocks) - len(final_blocks)} blocks freed ***")
        return 1


if __name__ == "__main__":
    sys.exit(test_force_free())
