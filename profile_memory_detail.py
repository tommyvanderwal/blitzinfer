#!/usr/bin/env python3
"""Profile GPU memory in detail - find every MB."""
import os
import gc
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


def get_detailed_memory_info():
    """Get detailed breakdown of GPU memory."""
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    # Driver memory = total - free - what PyTorch knows about
    pytorch_using = reserved
    driver_mem = (total - free) - pytorch_using

    return {
        'total_gb': total / 1024**3,
        'free_gb': free / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'pytorch_allocated_gb': allocated / 1024**3,
        'pytorch_reserved_gb': reserved / 1024**3,
        'driver_overhead_gb': max(0, driver_mem) / 1024**3,
    }


def print_memory(label=""):
    """Print detailed memory state."""
    info = get_detailed_memory_info()
    print(f"\n[{label}]")
    print(f"  CUDA total:         {info['total_gb']:.3f} GB")
    print(f"  CUDA free:          {info['free_gb']:.3f} GB")
    print(f"  CUDA used:          {info['used_gb']:.3f} GB")
    print(f"  PyTorch allocated:  {info['pytorch_allocated_gb']:.3f} GB")
    print(f"  PyTorch reserved:   {info['pytorch_reserved_gb']:.3f} GB")
    print(f"  Driver overhead:    {info['driver_overhead_gb']:.3f} GB")
    return info


def get_allocated_blocks():
    """Get all allocated blocks from memory snapshot."""
    snapshot = torch.cuda.memory._snapshot()
    blocks = []

    for seg in snapshot.get('segments', []):
        for block in seg.get('blocks', []):
            if block.get('state') == 'active_allocated':
                addr = block.get('addr', block.get('address', 0))
                size = block.get('size', 0)
                frames = block.get('frames', [])

                # Get stack trace if available
                trace = []
                for f in frames[:3]:  # First 3 frames
                    filename = f.get('filename', '')
                    line = f.get('line', 0)
                    name = f.get('name', '')
                    if filename:
                        trace.append(f"{filename}:{line} {name}")

                blocks.append({
                    'addr': addr,
                    'size_mb': size / 1024**2,
                    'trace': trace,
                })

    return sorted(blocks, key=lambda x: -x['size_mb'])


def main():
    print("=" * 70)
    print("Detailed GPU Memory Profiler")
    print("=" * 70)

    # Enable memory tracking
    torch.cuda.memory._record_memory_history(max_entries=100000)

    print_memory("BASELINE - Before CUDA init")

    # Force CUDA context creation
    _ = torch.zeros(1, device='cuda')
    del _
    torch.cuda.empty_cache()

    baseline = print_memory("After CUDA context init")

    print("\n" + "=" * 70)
    print("Loading gpt-oss-120b (MXFP4)...")
    print("=" * 70)

    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    llm1 = LLM(
        model="openai/gpt-oss-120b",
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )

    after_load1 = print_memory("After gpt-oss-120b load")

    # Run inference
    outputs = llm1.generate(["What is 2+2?"], SamplingParams(max_tokens=50))
    print(f"Output: {outputs[0].outputs[0].text[:50]}")

    after_infer1 = print_memory("After gpt-oss-120b inference")

    # Cleanup
    print("\nCleaning up gpt-oss-120b with force_free=True...")
    freed1 = full_cleanup(llm1, nuclear=True, force_free=True)
    llm1 = None

    after_cleanup1 = print_memory("After gpt-oss-120b cleanup")

    # Show remaining blocks
    blocks = get_allocated_blocks()
    if blocks:
        print(f"\nRemaining blocks after cleanup: {len(blocks)}")
        for b in blocks[:10]:
            print(f"  {b['size_mb']:.2f} MB at 0x{b['addr']:x}")
            for t in b['trace']:
                print(f"    {t}")

    print("\n" + "=" * 70)
    print("Loading qwen3-coder-next (FLA)...")
    print("=" * 70)

    llm2 = LLM(
        model="Qwen/Qwen3-Coder-Next-FP8",
        dtype="auto",
        gpu_memory_utilization=0.94,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
    )

    after_load2 = print_memory("After qwen3-coder-next load")

    # Run inference
    outputs = llm2.generate(["Hello"], SamplingParams(max_tokens=50))
    print(f"Output: {outputs[0].outputs[0].text[:50]}")

    after_infer2 = print_memory("After qwen3-coder-next inference")

    # Cleanup
    print("\nCleaning up qwen3-coder-next with force_free=False...")
    freed2 = full_cleanup(llm2, nuclear=True, force_free=False)
    llm2 = None

    after_cleanup2 = print_memory("After qwen3-coder-next cleanup")

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY DRIFT ANALYSIS")
    print("=" * 70)

    drift_after_gptoss = after_cleanup1['used_gb'] - baseline['used_gb']
    drift_after_qwen = after_cleanup2['used_gb'] - baseline['used_gb']

    print(f"\nBaseline used:              {baseline['used_gb']:.3f} GB")
    print(f"After gpt-oss cleanup:      {after_cleanup1['used_gb']:.3f} GB (drift: {drift_after_gptoss:+.3f} GB)")
    print(f"After qwen3-coder cleanup:  {after_cleanup2['used_gb']:.3f} GB (drift: {drift_after_qwen:+.3f} GB)")

    # Final blocks
    blocks = get_allocated_blocks()
    if blocks:
        total_mb = sum(b['size_mb'] for b in blocks)
        print(f"\nFinal allocated blocks: {len(blocks)} totaling {total_mb:.2f} MB")
        for b in blocks[:5]:
            print(f"  {b['size_mb']:.2f} MB at 0x{b['addr']:x}")
            for t in b['trace'][:2]:
                print(f"    {t}")

    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
