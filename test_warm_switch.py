#!/usr/bin/env python3
"""
Test warm model switch: arena pre-loaded, then vLLM init.

This simulates the real use case where arena loading happens
in the background while serving another model.
"""

import os
import gc
import time
import subprocess

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
from pathlib import Path
from huggingface_hub import snapshot_download


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=" * 80)
    print("WARM MODEL SWITCH TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    _ = torch.randn(1000, device='cuda')
    cleanup()

    from blitzinfer.memory import (
        PinnedMemoryArena,
        load_model_to_arena,
        get_model_size,
        set_preloaded_weights,
    )

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))
    model_gb = model_size / 1e9

    print(f"\nModel: {model_name}")
    print(f"Size: {model_gb:.1f} GB")

    # PHASE 1: Pre-load into arena (would be background during inference)
    print("\n" + "=" * 80)
    print("PHASE 1: Pre-load into arena (background)")
    print("=" * 80)

    arena_size_gb = model_gb + 5
    print(f"Allocating {arena_size_gb:.0f}GB arena...")
    arena = PinnedMemoryArena(arena_size_gb)

    print("Loading model into arena...")
    t0 = time.perf_counter()
    load_model_to_arena(str(model_path), arena, model_name)
    arena_load_time = time.perf_counter() - t0
    print(f"Arena load: {arena_load_time:.2f}s ({model_gb / arena_load_time:.1f} GB/s)")

    # Get pinned tensors (fast, just views)
    print("Getting tensor views...")
    t0 = time.perf_counter()
    pinned_tensors = arena.get_all_tensors(model_name)
    view_time = time.perf_counter() - t0
    print(f"Tensor views: {view_time:.3f}s")

    # PHASE 2: User requests model switch - measure only this
    print("\n" + "=" * 80)
    print("PHASE 2: MODEL SWITCH (measured time)")
    print("=" * 80)

    set_preloaded_weights(pinned_tensors)

    from vllm import LLM, SamplingParams

    print("Creating vLLM...")
    switch_start = time.perf_counter()
    llm = LLM(
        model=model_name,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    switch_time = time.perf_counter() - switch_start
    print(f"SWITCH TIME: {switch_time:.2f}s")

    # PHASE 3: Verify with inference
    print("\n" + "=" * 80)
    print("PHASE 3: Verify")
    print("=" * 80)

    print("Running inference...")
    t0 = time.perf_counter()
    outputs = llm.generate(
        ["What is the capital of France? Answer in one word:"],
        SamplingParams(max_tokens=10, temperature=0.7),
    )
    infer_time = time.perf_counter() - t0
    print(f"Inference: {infer_time:.2f}s")
    print(f"Response: {outputs[0].outputs[0].text.strip()}")

    del llm
    del pinned_tensors
    arena.clear()
    del arena
    cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Background (can be hidden):
  - Arena pre-load: {arena_load_time:.2f}s

Switch time (user-facing):
  - vLLM init with pre-loaded weights: {switch_time:.2f}s

Comparison with baseline:
  - Baseline vLLM cold load: ~18s (from safetensors)
  - Effective switch time: {switch_time:.2f}s
  - Speedup factor: {18.27 / switch_time:.1f}x (if arena pre-loaded)
""")


if __name__ == '__main__':
    main()
