#!/usr/bin/env python3
"""Surgical profiler to find exact source of memory drift per switch.

Traces every allocation that survives cleanup, with Python stack traces.
"""
import os
import gc
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# Enable CUDA memory history with Python stack traces
torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")


def snapshot_blocks():
    """Get all active allocated blocks with stack traces."""
    snapshot = torch.cuda.memory._snapshot()
    blocks = {}
    for seg in snapshot.get('segments', []):
        for block in seg.get('blocks', []):
            if block.get('state') == 'active_allocated':
                addr = block.get('addr', block.get('address', 0))
                size = block.get('size', 0)

                # Get Python stack frames
                traces = []
                for frame in block.get('frames', []):
                    fn = frame.get('filename', '')
                    line = frame.get('line', 0)
                    name = frame.get('name', '')
                    if fn and not any(skip in fn for skip in [
                        '/lib/python3.', 'importlib', '_bootstrap'
                    ]):
                        traces.append(f"{os.path.basename(fn)}:{line} {name}")

                blocks[addr] = {
                    'addr': addr,
                    'size': size,
                    'size_kb': size / 1024,
                    'size_mb': size / 1024**2,
                    'traces': traces[:5],
                }
    return blocks


def print_blocks(blocks, label=""):
    """Print block summary."""
    if not blocks:
        print(f"  [{label}] No allocated blocks")
        return

    total = sum(b['size'] for b in blocks.values())
    print(f"  [{label}] {len(blocks)} blocks, {total/1024**2:.2f} MB total")

    # Group by first trace
    by_source = {}
    for b in sorted(blocks.values(), key=lambda x: -x['size']):
        source = b['traces'][0] if b['traces'] else '<no trace>'
        if source not in by_source:
            by_source[source] = {'count': 0, 'total': 0, 'blocks': []}
        by_source[source]['count'] += 1
        by_source[source]['total'] += b['size']
        by_source[source]['blocks'].append(b)

    for source, info in sorted(by_source.items(), key=lambda x: -x[1]['total']):
        print(f"    {info['total']/1024**2:.3f} MB ({info['count']} blocks) - {source}")
        for b in info['blocks'][:3]:
            if len(b['traces']) > 1:
                print(f"      ^ {b['traces'][1]}")


def mem(label=""):
    """Print memory state."""
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    used = (total - free) / 1024**3
    print(f"[{label}] used={used:.3f}GB alloc={alloc/1024**3:.3f}GB free={free/1024**3:.3f}GB")
    return free, total


def main():
    print("=" * 70)
    print("DRIFT SOURCE PROFILER - Every byte accounted for")
    print("=" * 70)

    # Baseline
    _ = torch.zeros(1, device='cuda')
    del _
    gc.collect()
    torch.cuda.empty_cache()
    baseline_free, total = mem("0. Baseline")
    snap0 = snapshot_blocks()
    print_blocks(snap0, "baseline blocks")

    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import (
        cleanup_vllm_model, clear_vllm_caches, destroy_parallel_state,
        nuclear_cleanup, clear_fla_module_caches, force_free_all_allocated_blocks,
    )

    # ===== SWITCH 1: gpt-oss-120b =====
    print("\n\n" + "#" * 70)
    print("# SWITCH 1: gpt-oss-120b (MXFP4)")
    print("#" * 70)

    llm = LLM(model="openai/gpt-oss-120b", dtype="auto",
              gpu_memory_utilization=0.90, max_model_len=4096,
              trust_remote_code=True, enforce_eager=True)
    mem("1a. loaded")

    outputs = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=50))
    print(f"   Output: {outputs[0].outputs[0].text[:40]}")
    mem("1b. after inference")

    # Step-by-step cleanup
    print("\n--- Step-by-step cleanup ---")
    snap_pre = snapshot_blocks()

    cleanup_vllm_model(llm)
    llm = None
    gc.collect(); torch.cuda.empty_cache()
    free1c, _ = mem("1c. cleanup_vllm_model")
    snap1c = snapshot_blocks()

    clear_vllm_caches()
    gc.collect(); torch.cuda.empty_cache()
    free1d, _ = mem("1d. clear_vllm_caches")

    destroy_parallel_state()
    gc.collect(); torch.cuda.empty_cache()
    free1e, _ = mem("1e. destroy_parallel_state")

    nuclear_cleanup()
    gc.collect(); torch.cuda.empty_cache()
    free1f, _ = mem("1f. nuclear_cleanup")

    clear_fla_module_caches()
    gc.collect(); torch.cuda.empty_cache()
    free1g, _ = mem("1g. clear_fla_module_caches")

    # cuBLAS clear
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()
    free1h, _ = mem("1h. clearCublasWorkspaces")

    # Force free
    snap_before_ff = snapshot_blocks()
    print(f"\n--- Blocks before force_free: ---")
    print_blocks(snap_before_ff, "before force_free")

    nblocks, ngb = force_free_all_allocated_blocks()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    free1i, _ = mem(f"1i. force_free ({nblocks} blocks, {ngb:.2f}GB)")

    snap_after_ff = snapshot_blocks()
    print(f"\n--- Blocks SURVIVING force_free: ---")
    print_blocks(snap_after_ff, "after force_free")

    # Drift after switch 1
    drift1 = (total - free1i) / 1024**3 - (total - baseline_free) / 1024**3
    print(f"\n>>> DRIFT after switch 1: {drift1:+.3f} GB")

    # ===== SWITCH 2: qwen3-32b =====
    print("\n\n" + "#" * 70)
    print("# SWITCH 2: qwen3-32b")
    print("#" * 70)

    llm2 = LLM(model="Qwen/Qwen3-32B", dtype="bfloat16",
                gpu_memory_utilization=0.90, max_model_len=4096,
                trust_remote_code=True, enforce_eager=True)
    mem("2a. loaded")

    outputs = llm2.generate(["Capital of France?"], SamplingParams(max_tokens=30))
    print(f"   Output: {outputs[0].outputs[0].text[:40]}")
    mem("2b. after inference")

    # Full cleanup (no force_free)
    cleanup_vllm_model(llm2)
    llm2 = None
    gc.collect(); torch.cuda.empty_cache()
    mem("2c. cleanup_vllm_model")

    clear_vllm_caches()
    destroy_parallel_state()
    nuclear_cleanup()
    clear_fla_module_caches()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    free2d, _ = mem("2d. full cleanup (no force_free)")

    snap2d = snapshot_blocks()
    print(f"\n--- Blocks surviving switch 2 (no force_free): ---")
    print_blocks(snap2d, "after switch 2")

    drift2 = (total - free2d) / 1024**3 - (total - baseline_free) / 1024**3
    print(f"\n>>> CUMULATIVE DRIFT after switch 2: {drift2:+.3f} GB")

    # ===== SWITCH 3: gpt-oss again with force_free =====
    print("\n\n" + "#" * 70)
    print("# SWITCH 3: gpt-oss-120b again")
    print("#" * 70)

    llm3 = LLM(model="openai/gpt-oss-120b", dtype="auto",
                gpu_memory_utilization=0.90, max_model_len=4096,
                trust_remote_code=True, enforce_eager=True)
    mem("3a. loaded")

    outputs = llm3.generate(["Hello"], SamplingParams(max_tokens=20))
    print(f"   Output: {outputs[0].outputs[0].text[:40]}")
    mem("3b. after inference")

    # Full cleanup with force_free
    cleanup_vllm_model(llm3)
    llm3 = None
    gc.collect(); torch.cuda.empty_cache()
    mem("3c. cleanup_vllm_model")

    clear_vllm_caches()
    destroy_parallel_state()
    nuclear_cleanup()
    clear_fla_module_caches()
    gc.collect(); torch.cuda.empty_cache()
    mem("3d. pre-force_free")

    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass

    snap_before_ff3 = snapshot_blocks()
    print(f"\n--- Blocks before force_free (switch 3): ---")
    print_blocks(snap_before_ff3, "before force_free")

    nblocks3, ngb3 = force_free_all_allocated_blocks()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    free3e, _ = mem(f"3e. force_free ({nblocks3} blocks, {ngb3:.2f}GB)")

    snap_final = snapshot_blocks()
    print(f"\n--- FINAL surviving blocks: ---")
    print_blocks(snap_final, "final")

    drift3 = (total - free3e) / 1024**3 - (total - baseline_free) / 1024**3
    print(f"\n>>> CUMULATIVE DRIFT after switch 3: {drift3:+.3f} GB")

    # ===== SUMMARY =====
    baseline_used = (total - baseline_free) / 1024**3
    print("\n\n" + "=" * 70)
    print("DRIFT SUMMARY")
    print("=" * 70)
    print(f"Baseline:       {baseline_used:.3f} GB")
    print(f"After switch 1: {drift1:+.3f} GB (gpt-oss, force_free)")
    print(f"After switch 2: {drift2:+.3f} GB (qwen3-32b, no force_free)")
    print(f"After switch 3: {drift3:+.3f} GB (gpt-oss again, force_free)")
    print(f"Per-switch avg: {drift3/3:+.3f} GB")

    if snap_final:
        print(f"\nFinal blocks ({len(snap_final)}):")
        for addr in sorted(snap_final, key=lambda a: -snap_final[a]['size']):
            b = snap_final[addr]
            print(f"  {b['size_kb']:.1f} KB at 0x{addr:x}")
            for t in b['traces']:
                print(f"    {t}")

    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
