#!/usr/bin/env python3
"""
Test pinned arena loading with Qwen3-VL-32B-Thinking-FP8.

This model uses FP8 quantization which may have better weight compatibility.
"""

import os
import gc
import time
import subprocess
from pathlib import Path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
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
    print("PINNED ARENA TEST WITH QWEN3-VL-32B-FP8")
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

    # Phase 1: Load into arena
    print("\n" + "=" * 80)
    print("PHASE 1: Pre-load into arena")
    print("=" * 80)

    arena_size_gb = model_gb + 5
    print(f"Allocating {arena_size_gb:.0f}GB arena...")
    arena = PinnedMemoryArena(arena_size_gb)

    print(f"\nLoading model into arena...")
    t0 = time.perf_counter()
    load_model_to_arena(str(model_path), arena, model_name)
    load_time = time.perf_counter() - t0
    print(f"Arena load time: {load_time:.2f}s ({model_gb / load_time:.1f} GB/s)")

    # Phase 2: Get tensor views
    print("\n" + "=" * 80)
    print("PHASE 2: Get tensor views")
    print("=" * 80)

    pinned_tensors = arena.get_all_tensors(model_name)
    total_bytes = sum(t.numel() * t.element_size() for t in pinned_tensors.values())
    print(f"Got {len(pinned_tensors)} tensors ({total_bytes/1e9:.2f}GB)")

    # Print some tensor info
    print("\nSample tensors:")
    for i, (name, tensor) in enumerate(pinned_tensors.items()):
        print(f"  {name}: {tensor.shape} {tensor.dtype}")
        if i >= 5:
            print(f"  ... and {len(pinned_tensors)-6} more")
            break

    # Phase 3: Create vLLM
    print("\n" + "=" * 80)
    print("PHASE 3: Create vLLM with pinned_arena loader")
    print("=" * 80)

    set_preloaded_weights(pinned_tensors)

    from vllm import LLM, SamplingParams

    print("Creating LLM...")
    t0 = time.perf_counter()

    try:
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
        vllm_time = time.perf_counter() - t0
        print(f"vLLM init: {vllm_time:.2f}s")

        # Run inference
        print("\nRunning inference...")
        t0 = time.perf_counter()
        outputs = llm.generate(
            ["What is 2+2? Answer briefly:"],
            SamplingParams(max_tokens=20, temperature=0.7),
        )
        infer_time = time.perf_counter() - t0
        print(f"Inference: {infer_time:.2f}s")
        print(f"Response: {outputs[0].outputs[0].text}")

        del llm
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        vllm_time = 0

    del pinned_tensors
    arena.clear()
    del arena
    cleanup()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Arena load: {load_time:.2f}s (background)")
    print(f"vLLM init: {vllm_time:.2f}s (switch time)")


if __name__ == '__main__':
    main()
