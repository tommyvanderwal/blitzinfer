#!/usr/bin/env python3
"""Test 40GB pinned arena with 1GB chunks - safe end-to-end test.

This test uses a smaller arena (40GB) to ensure we have enough headroom
for vLLM initialization during model switching.

Models:
- Qwen3-32B-FP8 (~33GB) - fits in 40GB arena
- gpt-oss-120b (~65GB) - cold load only (too big for arena)
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_mem():
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

    def clear_rope_cache():
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()

    MODEL_A = "openai/gpt-oss-120b"              # ~65GB - cold load only
    MODEL_B = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB - fits in 40GB arena

    print("=" * 70)
    print("40GB PINNED ARENA TEST (1GB Chunks)")
    print("=" * 70)
    print()
    print("Testing with 40GB arena (Qwen fits, gpt-oss cold only)")
    print()

    m0 = log_mem("START")

    # Initialize StandbyManager with 40GB pinned arena
    print("\n[Phase 1] Initialize 40GB pinned arena...")
    standby = StandbyManager(
        arena_size_gb=40.0,
        chunk_size_gb=1.0,
        pin_memory=True,
    )
    print("  Manager initialized (arena lazy-allocated)")

    # Cold load Model A (gpt-oss-120b)
    print(f"\n[Phase 2] Cold load {MODEL_A}...")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_A,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,  # 66GB model needs ~70% of 95GB GPU
        enforce_eager=True,
        trust_remote_code=True,
    )
    cold_load_a = time.perf_counter() - t0
    print(f"  Cold load: {cold_load_a:.1f}s")

    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"  Model A: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("AFTER LOAD A")

    # Start prefetch for Model B (Qwen - fits in 40GB)
    print(f"\n[Phase 3] Prefetch {MODEL_B} while serving A...")
    standby.start_prefetch(MODEL_B)

    # Serve requests while prefetch runs
    for i in range(3):
        time.sleep(2)
        state = standby.get_state()
        print(f"    [{i*2}s] State: {state.name}")
        if state == StandbyState.READY:
            break

    standby.wait_for_load(timeout=180)
    m1 = log_mem("AFTER PREFETCH B")

    stats = standby.get_stats()
    print(f"  Standby: {stats['model']} ({stats['model_size_gb']:.1f}GB) in {stats['load_time']:.1f}s")

    # Fast switch to Model B
    print(f"\n[Phase 4] Fast switch A → B...")
    premerged = standby.consume_standby()
    print(f"  Got {len(premerged)} premerged tensors")

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    clear_rope_cache()
    log_mem("AFTER UNLOAD A")

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
    switch_ab = time.perf_counter() - t0
    print(f"  Switch time: {switch_ab:.1f}s")

    out = llm.generate(["3+3="], SamplingParams(max_tokens=10))
    print(f"  Model B: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("AFTER LOAD B")

    # Prefetch Model A? No - too big for 40GB arena
    # Instead, cold switch back to A
    print(f"\n[Phase 5] Cold switch B → A...")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    clear_rope_cache()

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_A,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,  # 66GB model needs ~70% of 95GB GPU
        enforce_eager=True,
        trust_remote_code=True,
    )
    cold_switch_ba = time.perf_counter() - t0
    print(f"  Cold switch: {cold_switch_ba:.1f}s")

    out = llm.generate(["5+5="], SamplingParams(max_tokens=10))
    print(f"  Model A: {out[0].outputs[0].text.strip()[:50]}")

    # Start prefetch for Model B again
    print(f"\n[Phase 6] Prefetch B while serving A...")
    standby.start_prefetch(MODEL_B)
    standby.wait_for_load(timeout=180)

    # Fast switch A → B again
    print(f"\n[Phase 7] Fast switch A → B again...")
    premerged = standby.consume_standby()

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    clear_rope_cache()

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
    switch_ab2 = time.perf_counter() - t0
    print(f"  Switch time: {switch_ab2:.1f}s")

    out = llm.generate(["7+7="], SamplingParams(max_tokens=10))
    print(f"  Model B: {out[0].outputs[0].text.strip()[:50]}")

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
    shmem_overhead = (m1['Shmem'] - m0['Shmem']) - 40
    print(f"""
    Arena: 40GB pinned (1GB chunks)
    Shmem overhead: {shmem_overhead:.1f}GB (expected ~0)

    Cold load gpt-oss-120b:     {cold_load_a:.1f}s
    Fast switch (gpt → qwen):   {switch_ab:.1f}s
    Cold switch (qwen → gpt):   {cold_switch_ba:.1f}s
    Fast switch (gpt → qwen):   {switch_ab2:.1f}s

    Speedup: {cold_load_a / switch_ab:.1f}x (cold vs fast)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
