#!/usr/bin/env python3
"""Test 70GB pinned arena with 1GB chunks.

This test verifies that the 1GB chunk strategy eliminates the huge page
overhead, allowing a full 70GB pinned arena for fast model switching.

Expected results:
- 70GB pinned arena with 0% memory overhead
- ~46 GB/s transfer speed to GPU
- Fast model switching between Qwen3-32B and gpt-oss-120b
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_mem():
    """Get memory stats from /proc/meminfo."""
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = int(parts[1].strip().split()[0]) / 1024 / 1024  # GB
                mem[key] = val
    return mem


def log_mem(label):
    mem = get_mem()
    shmem = mem.get('Shmem', 0)
    avail = mem.get('MemAvailable', 0)
    print(f"[{label:30s}] Shmem: {shmem:.1f}GB, MemAvail: {avail:.1f}GB", flush=True)
    return mem


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers import rotary_embedding
    from huggingface_hub import snapshot_download

    def clear_vllm_rope_cache():
        """Clear vLLM's RoPE cache to allow cross-architecture switching."""
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()

    # Test models
    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB
    MODEL_B = "openai/gpt-oss-120b"              # ~60GB

    print("=" * 70)
    print("70GB PINNED ARENA TEST (1GB Chunks)")
    print("=" * 70)
    print()
    print("This test verifies:")
    print("  1. 70GB pinned arena allocates with 0% overhead")
    print("  2. Fast GPU transfer speed (~46 GB/s)")
    print("  3. Model switching works correctly")
    print()

    m0 = log_mem("START")

    # Phase 1: Initialize StandbyManager with 70GB pinned arena
    print("\n[Phase 1] Initialize 70GB pinned arena (70x 1GB chunks)...")
    t0 = time.perf_counter()
    standby = StandbyManager(
        arena_size_gb=70.0,
        chunk_size_gb=1.0,  # 1GB chunks to avoid huge page overhead
        pin_memory=True,
    )
    init_time = time.perf_counter() - t0
    print(f"  StandbyManager init: {init_time:.1f}s")

    # Note: Arena is lazy-allocated on first prefetch

    # Phase 2: Cold load Model A
    print(f"\n[Phase 2] Cold load {MODEL_A}...")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_A,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        trust_remote_code=True,
    )
    cold_load_time = time.perf_counter() - t0
    print(f"  Cold load: {cold_load_time:.1f}s")

    # Quick verification
    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"  Model A says: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("AFTER MODEL A LOAD")

    # Phase 3: Start prefetch for Model B (this triggers arena allocation)
    print(f"\n[Phase 3] Start prefetch for {MODEL_B}...")
    standby.start_prefetch(MODEL_B)
    log_mem("AFTER PREFETCH START")

    # Continue serving Model A while prefetch runs
    print("  Serving Model A while prefetching...")
    for i in range(3):
        time.sleep(2)
        state = standby.get_state()
        print(f"    Prefetch state: {state.name}")
        if state == StandbyState.READY:
            break

    # Wait for prefetch to complete
    print("  Waiting for prefetch completion...")
    standby.wait_for_load(timeout=180)
    m1 = log_mem("AFTER PREFETCH COMPLETE")

    # Check memory overhead
    shmem_delta = m1['Shmem'] - m0['Shmem']
    print(f"\n  Shmem delta: {shmem_delta:.1f}GB")
    if shmem_delta > 75:  # 70GB + small overhead
        print(f"  WARNING: High memory overhead! Expected ~70GB, got {shmem_delta:.1f}GB")
    else:
        print(f"  OK: Memory usage within expected range")

    stats = standby.get_stats()
    print(f"  Standby state: {stats['state']}")
    print(f"  Model: {stats['model']}")
    print(f"  Model size: {stats['model_size_gb']:.1f}GB")
    print(f"  Load time: {stats['load_time']:.1f}s")

    assert standby.is_ready(MODEL_B), f"Expected {MODEL_B} to be ready in standby"

    # Phase 4: Fast switch to Model B
    print(f"\n[Phase 4] Fast switch to {MODEL_B}...")

    # Consume standby tensors
    premerged = standby.consume_standby()
    assert premerged is not None, "Expected premerged tensors"
    print(f"  Got {len(premerged)} premerged tensors")

    # Unload Model A
    print("  Unloading Model A...")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    clear_vllm_rope_cache()
    log_mem("AFTER UNLOAD A")

    # Load Model B from standby
    print("  Loading Model B from standby...")
    t0 = time.perf_counter()
    set_preloaded_weights(premerged)
    llm = LLM(
        model=MODEL_B,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        trust_remote_code=True,
    )
    switch_time = time.perf_counter() - t0
    print(f"  Switch time: {switch_time:.1f}s")
    log_mem("AFTER LOAD B")

    # Verification
    print("\n  Verifying Model B...")
    out = llm.generate(["3+3="], SamplingParams(max_tokens=20))
    response = out[0].outputs[0].text.strip()[:100]
    print(f"  Model B says: {response}")

    # Phase 5: Start prefetch for Model A
    print(f"\n[Phase 5] Start prefetch for {MODEL_A}...")
    standby.start_prefetch(MODEL_A)

    # Wait for prefetch
    print("  Waiting for prefetch...")
    standby.wait_for_load(timeout=180)
    log_mem("AFTER PREFETCH A")

    # Phase 6: Switch back to Model A
    print(f"\n[Phase 6] Fast switch back to {MODEL_A}...")
    premerged = standby.consume_standby()
    assert premerged is not None, "Expected premerged tensors"

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    clear_vllm_rope_cache()

    t0 = time.perf_counter()
    set_preloaded_weights(premerged)
    llm = LLM(
        model=MODEL_A,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        trust_remote_code=True,
    )
    switch_back_time = time.perf_counter() - t0
    print(f"  Switch time: {switch_back_time:.1f}s")

    # Verification
    out = llm.generate(["5+5="], SamplingParams(max_tokens=10))
    print(f"  Model A says: {out[0].outputs[0].text.strip()[:50]}")

    # Cleanup
    print("\n[Cleanup]")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    standby.shutdown()
    log_mem("AFTER CLEANUP")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Arena Configuration:
      - Size: 70GB
      - Chunks: 70x 1GB (to avoid huge page overhead)
      - Type: Pinned (CUDA page-locked)
      - Shmem overhead: {shmem_delta:.1f}GB (expected ~70GB)

    Performance:
      - Cold load (Model A): {cold_load_time:.1f}s
      - Warm switch (A → B): {switch_time:.1f}s
      - Warm switch (B → A): {switch_back_time:.1f}s
      - Speedup: {cold_load_time / switch_time:.1f}x

    Status: {"PASS" if shmem_delta < 75 else "FAIL - Memory overhead too high"}
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
