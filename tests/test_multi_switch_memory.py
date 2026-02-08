#!/usr/bin/env python3
"""Test memory stability with multiple model switches and standby prefetching.

This test simulates the comprehensive test scenario:
1. Pre-allocate 80GB arena
2. Load model, do inference
3. Switch to another model, prefetch previous
4. Repeat for multiple models
5. Verify no memory accumulation
"""

import os
import sys
import gc
import time
import ctypes

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.orchestrator.standby_manager import StandbyManager


def get_memory_info():
    """Get comprehensive memory state."""
    gpu_free, gpu_total = torch.cuda.mem_get_info()

    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024

    return {
        'gpu_used_gb': (gpu_total - gpu_free) / 1024**3,
        'gpu_total_gb': gpu_total / 1024**3,
        'ram_avail_gb': meminfo.get('MemAvailable', 0) / 1024**3,
        'ram_shared_gb': meminfo.get('Shmem', 0) / 1024**3,
    }


def log_mem(label, baseline_avail=None):
    """Log memory state."""
    mem = get_memory_info()
    delta_str = ""
    if baseline_avail is not None:
        delta = baseline_avail - mem['ram_avail_gb']
        delta_str = f" | RAM delta: {delta:+.1f}GB"
    print(f"[{label}] GPU: {mem['gpu_used_gb']:.1f}/{mem['gpu_total_gb']:.1f}GB | "
          f"RAM avail: {mem['ram_avail_gb']:.1f}GB, shared: {mem['ram_shared_gb']:.1f}GB{delta_str}")
    return mem


def malloc_trim():
    """Release freed memory to OS."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# Test models - use smaller subset to avoid hitting model limits
TEST_MODELS = [
    ("gpt-oss-120b", "openai/gpt-oss-120b"),
    ("qwen3-32b", "Qwen/Qwen3-32B-FP8"),
    ("mistral-small-24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
]


def main():
    from vllm import LLM, SamplingParams

    print("=" * 80)
    print("MULTI-MODEL SWITCH MEMORY TEST")
    print("=" * 80)

    # Initial state
    initial_mem = log_mem("INITIAL (no arena, no model)")
    baseline_avail = initial_mem['ram_avail_gb']

    # Create standby manager with 80GB arena
    print("\n" + "=" * 60)
    print("Creating 80GB pinned arena...")
    print("=" * 60)

    start = time.time()
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    print(f"Arena created in {time.time() - start:.1f}s")
    log_mem("After arena creation", baseline_avail)

    # Track memory at each switch
    switch_memory = []
    llm = None
    current_model = None

    # Do 4 rounds of switching through 3 models
    rounds = 4
    sampling_params = SamplingParams(max_tokens=5)

    for round_num in range(rounds):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num + 1}/{rounds}")
        print(f"{'=' * 60}")

        for model_name, model_path in TEST_MODELS:
            print(f"\n--- Switching to {model_name} ---")

            # Cleanup current model
            if llm is not None:
                print(f"Cleaning up {current_model}...")
                cleanup_start = time.time()
                freed = full_cleanup(llm)
                llm = None
                gc.collect()
                malloc_trim()
                print(f"Cleanup freed {freed:.1f}GB GPU in {time.time() - cleanup_start:.1f}s")
                log_mem(f"After cleanup {current_model}", baseline_avail)

            # Check if standby is ready for this model
            use_standby = standby.is_ready(model_path)
            if use_standby:
                print(f"Using standby for {model_name}!")
                premerged = standby.consume_standby()
                from blitzinfer.memory import set_preloaded_weights
                set_preloaded_weights(premerged)
                load_format = "pinned_arena"
            else:
                load_format = "auto"

            # Load model
            print(f"Loading {model_name} (format={load_format})...")
            load_start = time.time()
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.94,
                max_model_len=131072,
                trust_remote_code=True,
                enforce_eager=True,
                load_format=load_format,
            )
            current_model = model_name
            print(f"Loaded in {time.time() - load_start:.1f}s")
            mem = log_mem(f"After load {model_name}", baseline_avail)

            # Quick inference
            outputs = llm.generate(["Hello"], sampling_params)
            print(f"Inference: {outputs[0].outputs[0].text[:50]}")

            # Start prefetch of previous model (simulating server behavior)
            prev_idx = (TEST_MODELS.index((model_name, model_path)) - 1) % len(TEST_MODELS)
            prev_name, prev_path = TEST_MODELS[prev_idx]
            print(f"Starting prefetch of {prev_name}...")
            standby.start_prefetch(prev_path)

            # Record memory state
            switch_memory.append({
                'round': round_num + 1,
                'model': model_name,
                'ram_avail_gb': mem['ram_avail_gb'],
                'ram_delta_gb': baseline_avail - mem['ram_avail_gb'],
                'gpu_used_gb': mem['gpu_used_gb'],
            })

            # Wait a bit for prefetch to progress
            time.sleep(2)

    # Final cleanup
    print("\n" + "=" * 60)
    print("FINAL CLEANUP")
    print("=" * 60)

    if llm is not None:
        full_cleanup(llm)
        llm = None

    gc.collect()
    malloc_trim()
    log_mem("After final cleanup", baseline_avail)

    standby.shutdown()
    del standby
    gc.collect()
    malloc_trim()
    final_mem = log_mem("After arena shutdown", baseline_avail)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Round':<6} {'Model':<20} {'RAM Avail':<12} {'RAM Delta':<12} {'GPU Used':<10}")
    print("-" * 60)

    for entry in switch_memory:
        print(f"{entry['round']:<6} {entry['model']:<20} {entry['ram_avail_gb']:<12.1f} "
              f"{entry['ram_delta_gb']:<12.1f} {entry['gpu_used_gb']:<10.1f}")

    # Check for memory drift
    ram_deltas = [e['ram_delta_gb'] for e in switch_memory]
    min_delta = min(ram_deltas)
    max_delta = max(ram_deltas)
    drift = max_delta - min_delta

    print(f"\nMemory drift over {len(switch_memory)} switches: {drift:.1f}GB")
    print(f"  Min RAM delta: {min_delta:.1f}GB")
    print(f"  Max RAM delta: {max_delta:.1f}GB")

    if drift < 5.0:
        print("\n[PASS] Memory stable - drift < 5GB")
    else:
        print(f"\n[WARN] Memory drift detected: {drift:.1f}GB")

    print(f"\nFinal RAM available: {final_mem['ram_avail_gb']:.1f}GB")
    print(f"Expected (after arena release): ~{baseline_avail:.0f}GB")


if __name__ == "__main__":
    main()
