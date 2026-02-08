#!/usr/bin/env python3
"""Test if setting kv_cache_memory_bytes skips any profiling.

Run each test in a separate process to ensure clean GPU state.
"""

import os
import sys
import gc
import time
import subprocess


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def test_without_kv_cache_bytes():
    """Test 1: Without kv_cache_memory_bytes."""
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'

    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from blitzinfer.memory import (
        PinnedMemoryArena,
        load_model_to_arena,
        get_model_size,
        get_premerged_tensors_for_vllm,
        set_preloaded_weights,
    )
    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))

    # Load into arena
    arena = PinnedMemoryArena(model_size / 1e9 + 5)
    load_model_to_arena(str(model_path), arena, model_name)
    pinned_tensors = arena.get_all_tensors(model_name)
    premerged = get_premerged_tensors_for_vllm(pinned_tensors)

    set_preloaded_weights(premerged)
    t0 = time.perf_counter()
    llm = LLM(
        model=model_name,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    t = time.perf_counter() - t0
    print(f"RESULT:without_kv_cache_bytes:{t:.2f}")

    # Verify
    outputs = llm.generate(
        ["What is 1+1?"],
        SamplingParams(max_tokens=5, temperature=0.1),
    )
    print(f"Output: {outputs[0].outputs[0].text.strip()}")


def test_with_kv_cache_bytes():
    """Test 2: With kv_cache_memory_bytes."""
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'

    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from blitzinfer.memory import (
        PinnedMemoryArena,
        load_model_to_arena,
        get_model_size,
        get_premerged_tensors_for_vllm,
        set_preloaded_weights,
    )
    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))

    # Load into arena
    arena = PinnedMemoryArena(model_size / 1e9 + 5)
    load_model_to_arena(str(model_path), arena, model_name)
    pinned_tensors = arena.get_all_tensors(model_name)
    premerged = get_premerged_tensors_for_vllm(pinned_tensors)

    # Use the same KV cache size as computed in test 1
    kv_cache_bytes = int(7.0 * 1024**3)  # 7 GB

    set_preloaded_weights(premerged)
    t0 = time.perf_counter()
    llm = LLM(
        model=model_name,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        kv_cache_memory_bytes=kv_cache_bytes,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    t = time.perf_counter() - t0
    print(f"RESULT:with_kv_cache_bytes:{t:.2f}")

    # Verify
    outputs = llm.generate(
        ["What is 1+1?"],
        SamplingParams(max_tokens=5, temperature=0.1),
    )
    print(f"Output: {outputs[0].outputs[0].text.strip()}")


def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("KV CACHE SKIP TEST")
        print("=" * 70)
        print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")
        print("\nRunning tests in separate processes...")

        # Run test 1
        print("\n" + "=" * 60)
        print("TEST 1: Without kv_cache_memory_bytes")
        print("=" * 60)
        result1 = subprocess.run(
            [sys.executable, __file__, "1"],
            capture_output=True, text=True
        )
        print(result1.stdout)
        if result1.returncode != 0:
            print(f"Error: {result1.stderr}")
            return

        # Run test 2
        print("\n" + "=" * 60)
        print("TEST 2: With kv_cache_memory_bytes")
        print("=" * 60)
        result2 = subprocess.run(
            [sys.executable, __file__, "2"],
            capture_output=True, text=True
        )
        print(result2.stdout)
        if result2.returncode != 0:
            print(f"Error: {result2.stderr}")
            return

        # Extract results
        t1 = None
        t2 = None
        for line in result1.stdout.split('\n'):
            if line.startswith("RESULT:without"):
                t1 = float(line.split(':')[2])
        for line in result2.stdout.split('\n'):
            if line.startswith("RESULT:with"):
                t2 = float(line.split(':')[2])

        if t1 and t2:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Without kv_cache_memory_bytes: {t1:.2f}s")
            print(f"With kv_cache_memory_bytes:    {t2:.2f}s")
            print(f"Difference:                    {t1 - t2:.2f}s")
    else:
        test_num = int(sys.argv[1])
        if test_num == 1:
            test_without_kv_cache_bytes()
        elif test_num == 2:
            test_with_kv_cache_bytes()


if __name__ == '__main__':
    main()
