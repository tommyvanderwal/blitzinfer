#!/usr/bin/env python3
"""
Test caching effects on vLLM initialization.

This tests:
1. First load (cold Triton cache)
2. Second load (warm Triton cache, new engine)
3. Third load with engine reuse

Goal: Find what can be cached to speed up 2nd+ loads.
"""

import os
import gc
import time
import subprocess

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


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


def unload_vllm(llm):
    """Properly unload vLLM to release GPU memory."""
    try:
        # Shutdown engine core first
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'engine_core'):
                if hasattr(engine.engine_core, 'shutdown'):
                    engine.engine_core.shutdown()
    except Exception as e:
        print(f"Shutdown warning: {e}")

    # Clear model weights
    try:
        core = llm.llm_engine.engine_core
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                    if model_runner and hasattr(model_runner, 'model'):
                        for param in model_runner.model.parameters():
                            param.data = torch.empty(0, device='cpu')
                    if hasattr(model_runner, 'kv_caches'):
                        model_runner.kv_caches.clear()
    except Exception as e:
        print(f"Cleanup warning: {e}")

    del llm
    cleanup()


def time_vllm_init(model_name, label, use_pinned=False, premerged=None):
    """Time vLLM initialization."""
    from vllm import LLM, SamplingParams

    if use_pinned and premerged is not None:
        from blitzinfer.memory import set_preloaded_weights
        set_preloaded_weights(premerged)
        load_format = "pinned_arena"
    else:
        load_format = "auto"

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"GPU free before: {nvidia_smi_free_gb():.1f} GB")

    t0 = time.perf_counter()
    llm = LLM(
        model=model_name,
        load_format=load_format,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,  # Lower for sequential loads with imperfect cleanup
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    init_time = time.perf_counter() - t0
    print(f"Init time: {init_time:.2f}s")

    # Quick inference to verify
    t0 = time.perf_counter()
    outputs = llm.generate(
        ["What is 1+1?"],
        SamplingParams(max_tokens=5, temperature=0.1),
    )
    infer_time = time.perf_counter() - t0
    print(f"Inference: {infer_time:.2f}s")
    print(f"Output: {outputs[0].outputs[0].text.strip()}")

    return llm, init_time


def main():
    print("=" * 70)
    print("CACHE EFFECTS TEST")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    results = {}

    # Test 1: Standard vLLM load (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: Standard vLLM load (Triton cache should be warm from prior runs)")
    print("=" * 70)

    llm1, t1 = time_vllm_init(model_name, "Load 1 - Standard vLLM")
    results['standard_load'] = t1

    # Cleanup completely
    unload_vllm(llm1)
    print(f"\nGPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")

    # Test 2: Second standard load (same process, no engine reuse)
    print("\n" + "=" * 70)
    print("TEST 2: Second load (same process, fresh engine)")
    print("=" * 70)

    llm2, t2 = time_vllm_init(model_name, "Load 2 - Fresh engine, same process")
    results['second_load_fresh'] = t2

    unload_vllm(llm2)

    # Test 3: With pinned arena
    print("\n" + "=" * 70)
    print("TEST 3: Pinned arena load")
    print("=" * 70)

    from pathlib import Path
    from huggingface_hub import snapshot_download
    from blitzinfer.memory import (
        PinnedMemoryArena,
        load_model_to_arena,
        get_model_size,
        get_premerged_tensors_for_vllm,
    )

    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))

    arena = PinnedMemoryArena(model_size / 1e9 + 5)
    load_model_to_arena(str(model_path), arena, model_name)
    pinned_tensors = arena.get_all_tensors(model_name)
    premerged = get_premerged_tensors_for_vllm(pinned_tensors)

    llm3, t3 = time_vllm_init(model_name, "Load 3 - Pinned arena", use_pinned=True, premerged=premerged)
    results['pinned_arena'] = t3

    unload_vllm(llm3)
    print(f"\nGPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")

    # Test 4: Second pinned arena load (same premerged tensors)
    print("\n" + "=" * 70)
    print("TEST 4: Second pinned arena load (reusing premerged tensors)")
    print("=" * 70)

    llm4, t4 = time_vllm_init(model_name, "Load 4 - Pinned arena (2nd)", use_pinned=True, premerged=premerged)
    results['pinned_arena_2nd'] = t4

    unload_vllm(llm4)
    del premerged
    del pinned_tensors
    arena.clear()
    del arena
    cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Load Times:
  1. Standard vLLM:           {results['standard_load']:.2f}s
  2. Fresh engine (same proc): {results['second_load_fresh']:.2f}s
  3. Pinned arena (1st):       {results['pinned_arena']:.2f}s
  4. Pinned arena (2nd):       {results['pinned_arena_2nd']:.2f}s

Observations:
  - Standard vs 2nd load difference: {results['standard_load'] - results['second_load_fresh']:.2f}s
  - Pinned 1st vs 2nd difference:    {results['pinned_arena'] - results['pinned_arena_2nd']:.2f}s
""")


if __name__ == '__main__':
    main()
