#!/usr/bin/env python3
"""Simple standby test with explicit memory tracking."""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_cpu_mem():
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
    cpu = get_cpu_mem()
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_used = (gpu_total - gpu_free) / 1024**3
    gpu_free_gb = gpu_free / 1024**3
    print(f"[{label:30s}] CPU Shmem: {cpu['Shmem']:.1f}GB, "
          f"MemAvail: {cpu['MemAvailable']:.1f}GB, "
          f"GPU: {gpu_used:.1f}/{gpu_total/1024**3:.1f}GB used", flush=True)
    return cpu, gpu_free_gb


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers import rotary_embedding

    def clear_rope_cache():
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()

    # Use smaller models for this test
    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB
    MODEL_B = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # Same model for simplicity

    print("=" * 70)
    print("SIMPLE STANDBY TEST")
    print("=" * 70)

    log_mem("START")

    # Initialize StandbyManager
    print("\n[1] Initialize 40GB pinned arena...")
    standby = StandbyManager(
        arena_size_gb=40.0,
        chunk_size_gb=1.0,
        pin_memory=True,
    )

    # Cold load Model A
    print(f"\n[2] Cold load {MODEL_A}...")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_A,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    cold_load = time.perf_counter() - t0
    print(f"  Cold load: {cold_load:.1f}s")

    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"  Output: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("AFTER LOAD A")

    # Start prefetch
    print(f"\n[3] Prefetch {MODEL_B}...")
    standby.start_prefetch(MODEL_B)

    # Wait for prefetch
    print("  Waiting for prefetch...")
    standby.wait_for_load(timeout=180)
    log_mem("AFTER PREFETCH")

    stats = standby.get_stats()
    print(f"  Standby: {stats['state']}, model_size: {stats['model_size_gb']:.1f}GB")

    # Get premerged tensors BEFORE unloading
    print("\n[4] Consume standby tensors...")
    premerged = standby.consume_standby()
    print(f"  Got {len(premerged)} tensors")
    log_mem("AFTER CONSUME")

    # Unload Model A
    print("\n[5] Unload Model A...")
    print("  Calling engine_core.shutdown()...")
    llm.llm_engine.engine_core.shutdown()
    log_mem("AFTER SHUTDOWN")

    print("  Deleting LLM object...")
    del llm
    log_mem("AFTER DEL LLM")

    print("  gc.collect()...")
    gc.collect()
    log_mem("AFTER GC")

    print("  torch.cuda.empty_cache()...")
    torch.cuda.empty_cache()
    log_mem("AFTER EMPTY_CACHE")

    print("  clear_rope_cache()...")
    clear_rope_cache()
    log_mem("AFTER CLEAR_ROPE")

    # Small delay
    print("  Waiting 2s for cleanup...")
    time.sleep(2)
    log_mem("AFTER WAIT")

    # Now load Model B from standby
    print(f"\n[6] Load {MODEL_B} from standby...")
    t0 = time.perf_counter()
    set_preloaded_weights(premerged)
    llm = LLM(
        model=MODEL_B,
        load_format="pinned_arena",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    switch_time = time.perf_counter() - t0
    print(f"  Switch time: {switch_time:.1f}s")

    out = llm.generate(["3+3="], SamplingParams(max_tokens=10))
    print(f"  Output: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("AFTER LOAD B")

    # Cleanup
    print("\n[7] Cleanup...")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    standby.shutdown()
    log_mem("FINAL")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Cold load:  {cold_load:.1f}s")
    print(f"  Warm switch: {switch_time:.1f}s")
    print(f"  Speedup: {cold_load / switch_time:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
