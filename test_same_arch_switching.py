#!/usr/bin/env python3
"""Test same-architecture model switching with static 80GB arena.

Tests multiple switches of the same model to verify the standby system works.
This avoids the vLLM V1 single-process memory leak by using same architecture.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

import torch


def get_mem():
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                mem[parts[0].strip()] = int(parts[1].strip().split()[0]) / 1024 / 1024
    return mem


def get_gpu():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'total_gb': total / 1024**3,
    }


def log(msg):
    gpu = get_gpu()
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (GPU: {gpu['free_gb']:.1f}GB free)", flush=True)


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from blitzinfer.engine.cleanup import full_cleanup, log_gpu_memory
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB

    print("=" * 70)
    print("SAME-ARCHITECTURE SWITCHING TEST (Static 80GB Arena)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Arena: 80GB (5x 16GB pinned chunks, pre-allocated)")
    print()

    mem = get_mem()
    log(f"START: Shmem={mem['Shmem']:.1f}GB")

    # Initialize StandbyManager with static 80GB arena
    log("Pre-allocating 80GB standby arena...")
    t0 = time.perf_counter()
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    arena_time = time.perf_counter() - t0
    log(f"Arena ready in {arena_time:.1f}s")

    results = []
    times = []
    llm = None
    cold_time = 0

    try:
        # === Cold load ===
        log("\n=== PHASE 1: Cold load ===")
        log_gpu_memory("before cold load")

        t0 = time.perf_counter()
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        cold_time = time.perf_counter() - t0
        log(f"Cold load: {cold_time:.1f}s")

        out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        log(f"Output: {text[:50]}...")
        results.append(('cold', len(text) > 2))

        # === 5 warm switches ===
        for i in range(5):
            log(f"\n=== PHASE {i+2}: Switch {i+1}/5 ===")

            # Start prefetch in background while serving
            standby.start_prefetch(MODEL)

            # Wait for prefetch
            standby.wait_for_load(timeout=120)
            log("Prefetch done")

            # Get premerged tensors
            premerged = standby.consume_standby()
            log(f"Got {len(premerged)} premerged tensors")

            # Cleanup old model
            log("Cleaning up...")
            log_gpu_memory("before cleanup")
            freed = full_cleanup(llm)
            llm = None
            log_gpu_memory("after cleanup")
            log(f"Freed {freed:.1f}GB")

            time.sleep(0.5)

            # Load from standby
            t0 = time.perf_counter()
            set_preloaded_weights(premerged)
            llm = LLM(
                model=MODEL,
                load_format="pinned_arena",
                dtype="bfloat16",
                max_model_len=4096,
                gpu_memory_utilization=0.50,
                enforce_eager=True,
                trust_remote_code=True,
            )
            switch_time = time.perf_counter() - t0
            times.append(switch_time)
            log(f"Switch {i+1}: {switch_time:.1f}s")

            # Test output
            prompts = [
                f"{i+1}+{i+1}=",
                "The sky is",
                "Hello, my name is",
                "Count to 5:",
                "Capital of France is",
            ]
            out = llm.generate([prompts[i]], SamplingParams(max_tokens=30))
            text = out[0].outputs[0].text.strip()[:80]
            log(f"Output: {text[:50]}...")

            ok = len(text) > 2 and '!!!!' not in text
            results.append((f'switch-{i+1}', ok))

            if not ok:
                log("WARNING: Suspicious output!")

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        log("\n=== CLEANUP ===")
        if llm is not None:
            full_cleanup(llm)
        gc.collect()
        torch.cuda.empty_cache()
        standby.shutdown()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTiming:")
    print(f"  Cold load:      {cold_time:.1f}s")
    if times:
        avg = sum(times) / len(times)
        print(f"  Warm switches:  {', '.join(f'{t:.1f}s' for t in times)}")
        print(f"  Average warm:   {avg:.1f}s")
        print(f"  Speedup:        {cold_time/avg:.1f}x")

    print(f"\nOutput verification:")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    for phase, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {phase}")

    mem = get_mem()
    print(f"\nFinal memory:")
    print(f"  Shmem: {mem['Shmem']:.1f}GB")

    print("\n" + "=" * 70)
    if passed == total and len(times) == 5:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
