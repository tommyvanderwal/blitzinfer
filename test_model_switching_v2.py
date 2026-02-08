#!/usr/bin/env python3
"""Comprehensive model switching test with static 80GB arena.

Tests multiple switches between Qwen-32B and gpt-oss-120b with aggressive
GPU cleanup between switches.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
# Help with GPU memory fragmentation
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
    alloc = torch.cuda.memory_allocated()
    return {
        'free_gb': free / 1024**3,
        'total_gb': total / 1024**3,
        'alloc_gb': alloc / 1024**3,
    }


def log(msg):
    gpu = get_gpu()
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (GPU: {gpu['free_gb']:.1f}GB free)", flush=True)


def verify_output(model_name, prompt, output):
    """Check if output is lucid (not garbage)."""
    if not output or len(output.strip()) < 2:
        return False, "Empty output"

    garbage_patterns = ['!!!!', '????', '####', '.....' * 3, '\x00']
    for pattern in garbage_patterns:
        if pattern in output:
            return False, f"Contains garbage pattern: {pattern}"

    return True, "OK"


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from blitzinfer.engine.cleanup import full_cleanup, clear_rope_cache, log_gpu_memory
    from vllm import LLM, SamplingParams

    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB
    MODEL_B = "openai/gpt-oss-120b"              # ~65GB

    print("=" * 70)
    print("MODEL SWITCHING TEST V2 (Static 80GB Arena)")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Arena: 80GB (5x 16GB pinned chunks, pre-allocated)")
    print()

    mem = get_mem()
    log(f"START: Shmem={mem['Shmem']:.1f}GB")

    # Initialize StandbyManager with static 80GB arena
    log("Pre-allocating 80GB standby arena (5x 16GB chunks)...")
    t0 = time.perf_counter()
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,  # Pre-allocate now
    )
    arena_time = time.perf_counter() - t0
    log(f"Arena ready in {arena_time:.1f}s")

    mem = get_mem()
    log(f"After arena: Shmem={mem['Shmem']:.1f}GB")

    results = []
    llm = None
    cold_a = switch_ab = switch_ba = 0

    try:
        # === PHASE 1: Cold load Model A ===
        log(f"\n=== PHASE 1: Cold load {MODEL_A} ===")
        log_gpu_memory("before cold load")

        t0 = time.perf_counter()
        llm = LLM(
            model=MODEL_A,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        cold_a = time.perf_counter() - t0
        log(f"Cold load: {cold_a:.1f}s")

        # Test output
        out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        ok, reason = verify_output(MODEL_A, "math", text)
        log(f"Output A: {text[:50]}... [{reason}]")
        results.append(('A-cold', ok))

        # === PHASE 2: Prefetch Model B, switch A→B ===
        log(f"\n=== PHASE 2: Prefetch {MODEL_B}, switch A→B ===")

        # Start prefetch in background
        standby.start_prefetch(MODEL_B)

        # Serve Model A while prefetching
        log("Serving Model A while prefetching B...")
        for i in range(5):
            time.sleep(2)
            state = standby.get_state()
            if state == StandbyState.READY:
                log(f"Prefetch complete!")
                break
            log(f"  Prefetch state: {state.name}")

        # Wait for prefetch to complete
        standby.wait_for_load(timeout=180)
        mem = get_mem()
        log(f"Prefetch done: Shmem={mem['Shmem']:.1f}GB")

        # Get premerged tensors
        premerged = standby.consume_standby()
        log(f"Got {len(premerged)} premerged tensors")

        # Aggressive cleanup of Model A
        log("Cleaning up Model A...")
        log_gpu_memory("before cleanup")
        freed = full_cleanup(llm)
        llm = None
        log_gpu_memory("after cleanup")
        log(f"Freed {freed:.1f}GB GPU memory")

        # Give GPU time to settle
        time.sleep(1)
        log_gpu_memory("after settle")

        # Load Model B from standby
        # Calculate max utilization based on available memory
        gpu_info = get_gpu()
        available = gpu_info['free_gb']
        total = gpu_info['total_gb']
        # gpt-oss-120b needs ~60GB for weights, leave some room for KV cache
        # Use slightly less than available to leave margin
        util_b = min(0.85, (available - 2) / total)  # Leave 2GB margin
        log(f"Using gpu_memory_utilization={util_b:.2f} for Model B ({available:.1f}GB available)")

        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_B,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=util_b,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ab = time.perf_counter() - t0
        log(f"Switch A→B: {switch_ab:.1f}s")

        # Test output
        out = llm.generate(["What is 3+3?"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        ok, reason = verify_output(MODEL_B, "math", text)
        log(f"Output B: {text[:50]}... [{reason}]")
        results.append(('B-warm', ok))

        # === PHASE 3: Prefetch Model A, switch B→A ===
        log(f"\n=== PHASE 3: Prefetch {MODEL_A}, switch B→A ===")

        standby.start_prefetch(MODEL_A)
        standby.wait_for_load(timeout=180)

        premerged = standby.consume_standby()
        log(f"Got {len(premerged)} premerged tensors")

        # Aggressive cleanup of Model B
        log("Cleaning up Model B...")
        freed = full_cleanup(llm)
        llm = None
        log(f"Freed {freed:.1f}GB GPU memory")
        time.sleep(1)

        # Load Model A from standby
        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_A,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ba = time.perf_counter() - t0
        log(f"Switch B→A: {switch_ba:.1f}s")

        # Test output
        out = llm.generate(["Count from 1 to 5:"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        ok, reason = verify_output(MODEL_A, "counting", text)
        log(f"Output A: {text[:50]}... [{reason}]")
        results.append(('A-warm', ok))

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
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
    print(f"  Cold load (Model A):    {cold_a:.1f}s")
    print(f"  Warm switch A→B:        {switch_ab:.1f}s")
    print(f"  Warm switch B→A:        {switch_ba:.1f}s")
    if switch_ab > 0 and switch_ba > 0:
        avg = (switch_ab + switch_ba) / 2
        print(f"  Average warm switch:    {avg:.1f}s")
        print(f"  Speedup vs cold:        {cold_a / avg:.1f}x")

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
    print(f"  Available: {mem['MemAvailable']:.1f}GB")

    print("\n" + "=" * 70)
    if passed == total and switch_ab > 0 and switch_ba > 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
