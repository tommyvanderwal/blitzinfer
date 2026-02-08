#!/usr/bin/env python3
"""
Test: Full pinned arena → GPU → vLLM pipeline.

This test validates the fast model loading path:
1. Pre-load weights into pinned arena (can be hidden during inference)
2. Transfer to GPU at ~48 GB/s
3. Create vLLM model with load_format="pinned_arena"
4. Inject pre-loaded weights
5. Run inference to verify correctness

Target: ~6s model switch (1.4s transfer + 5s vLLM init)
"""

import os
import gc
import time
import subprocess
from pathlib import Path

# Configure for single-process mode
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


def get_ram_free_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024


def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=" * 80)
    print("PINNED ARENA → vLLM INTEGRATION TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")
    print(f"RAM free: {get_ram_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    # Import blitzinfer components
    from blitzinfer.memory import (
        PinnedMemoryArena,
        load_model_to_arena,
        get_model_size,
        set_preloaded_weights,
    )

    model_name = "openai/gpt-oss-120b"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))
    model_gb = model_size / 1e9

    print(f"\nModel: {model_name}")
    print(f"Size: {model_gb:.1f} GB")

    # PHASE 1: Pre-load into pinned arena
    print("\n" + "=" * 80)
    print("PHASE 1: Pre-load into pinned arena (would be background during inference)")
    print("=" * 80)

    arena_size_gb = model_gb + 5  # Some headroom
    print(f"Allocating {arena_size_gb:.0f}GB arena...")
    t0 = time.perf_counter()
    arena = PinnedMemoryArena(arena_size_gb)
    alloc_time = time.perf_counter() - t0
    print(f"Arena allocation: {alloc_time:.2f}s")

    print(f"\nLoading model into arena...")
    t0 = time.perf_counter()
    load_model_to_arena(str(model_path), arena, model_name)
    load_time = time.perf_counter() - t0
    load_bw = model_gb / load_time
    print(f"Arena load time: {load_time:.2f}s ({load_bw:.1f} GB/s)")

    # PHASE 2: Get pinned tensor views (NOT transferred to GPU yet)
    print("\n" + "=" * 80)
    print("PHASE 2: Get pinned tensor views")
    print("=" * 80)

    t0 = time.perf_counter()
    pinned_tensors = arena.get_all_tensors(model_name)
    view_time = time.perf_counter() - t0
    total_bytes = sum(t.numel() * t.element_size() for t in pinned_tensors.values())

    print(f"Got {len(pinned_tensors)} tensor views in {view_time:.3f}s")
    print(f"Total tensor data: {total_bytes / 1e9:.2f}GB")
    print(f"Tensors are on: {next(iter(pinned_tensors.values())).device}")
    print(f"GPU free (should be ~95GB): {nvidia_smi_free_gb():.1f} GB")

    # PHASE 3: Create vLLM with pinned_arena loader
    print("\n" + "=" * 80)
    print("PHASE 3: Create vLLM model with pre-loaded weights")
    print("=" * 80)

    # Set the pre-loaded PINNED weights for the custom loader
    # Transfer happens during load_weights, not before
    set_preloaded_weights(pinned_tensors)

    # Now create vLLM with our custom loader
    from vllm import LLM, SamplingParams

    print("Creating LLM with load_format='pinned_arena'...")
    t0 = time.perf_counter()

    try:
        llm = LLM(
            model=model_name,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=8192,  # Smaller for faster init
            gpu_memory_utilization=0.90,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            enforce_eager=True,
            trust_remote_code=True,
        )
        vllm_init_time = time.perf_counter() - t0
        print(f"vLLM init time: {vllm_init_time:.2f}s")

        # PHASE 4: Run inference to verify
        print("\n" + "=" * 80)
        print("PHASE 4: Verify with inference")
        print("=" * 80)

        print("Running inference...")
        t0 = time.perf_counter()
        outputs = llm.generate(
            ["Hello, how are you today?"],
            SamplingParams(max_tokens=50, temperature=0.7),
        )
        infer_time = time.perf_counter() - t0

        print(f"Inference time: {infer_time:.2f}s")
        print(f"Response: {outputs[0].outputs[0].text[:100]}...")

        # Cleanup
        del llm

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        vllm_init_time = 0
        infer_time = 0

    # Cleanup
    del pinned_tensors
    arena.clear()
    del arena
    cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # vLLM init includes the weight transfer (pinned → GPU)
    total_switch_time = vllm_init_time

    print(f"\n{'Phase':<40} {'Time':>10}")
    print("-" * 55)
    print(f"{'1. Pre-load to arena (background)':<40} {load_time:>9.2f}s")
    print(f"{'2. vLLM init + weight injection':<40} {vllm_init_time:>9.2f}s")
    print("-" * 55)
    print(f"{'TOTAL SWITCH TIME':<40} {total_switch_time:>9.2f}s")

    print(f"""
RESULTS:
- Pre-load to arena: {load_time:.1f}s (can be hidden during inference)
- vLLM init + injection: {vllm_init_time:.2f}s
- Total switch time: {total_switch_time:.2f}s

COMPARISON:
- Previous (page cache warm): ~14s
- This approach: {total_switch_time:.1f}s
- Speedup: {14/max(total_switch_time, 0.1):.1f}x (if it works!)
""")


if __name__ == '__main__':
    main()
