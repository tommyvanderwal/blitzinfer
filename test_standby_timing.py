#!/usr/bin/env python3
"""Standby timing test: Show pre-load vs switchover times.

Uses 2 models that work with pinned arena:
- Qwen/Qwen3-VL-32B-Thinking-FP8 (~35GB)
- mistralai/Mistral-Small-3.2-24B-Instruct-2506 (~48GB)

Key metrics:
- Pre-load time: Background loading into pinned RAM
- Switchover time: Cleanup + GPU injection (user-facing)
"""

import os
import gc
import sys
import time
from datetime import datetime

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'used_gb': (total - free) / 1024**3}


def verify_model(llm, name):
    """Quick verification that model produces sensible output."""
    from vllm import SamplingParams
    out = llm.generate(["What is 2+2? Answer with one number:"],
                       SamplingParams(max_tokens=10, temperature=0))
    answer = out[0].outputs[0].text.strip()
    passed = '4' in answer
    log(f"  Verify: '{answer}' [{'PASS' if passed else 'FAIL'}]")
    return passed


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager
    from blitzinfer.memory import set_preloaded_weights
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    MODEL_B = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    ARENA_SIZE = 50.0  # Smaller arena for both models
    NUM_SWITCHES = 4

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 32768,  # Smaller context for faster testing
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 70)
    log("STANDBY TIMING TEST")
    log("=" * 70)
    log(f"Model A: {MODEL_A}")
    log(f"Model B: {MODEL_B}")
    log(f"Arena: {ARENA_SIZE}GB")
    log(f"Switches: {NUM_SWITCHES}")
    log("")

    results = []

    # Initialize standby manager
    log("Allocating pinned arena...")
    arena_start = time.time()
    standby = StandbyManager(
        arena_size_gb=ARENA_SIZE,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    arena_time = time.time() - arena_start
    log(f"Arena allocated in {arena_time:.1f}s")

    llm = None
    current_model = None

    try:
        # Cold load first model
        log(f"\nCold loading {MODEL_A}...")
        cold_start = time.time()
        llm = LLM(model=MODEL_A, **VLLM_KWARGS)
        cold_time = time.time() - cold_start
        current_model = MODEL_A
        log(f"Cold load: {cold_time:.1f}s")
        verify_model(llm, MODEL_A)

        for i in range(NUM_SWITCHES):
            next_model = MODEL_B if current_model == MODEL_A else MODEL_A

            log(f"\n{'='*70}")
            log(f"SWITCH {i+1}: {current_model.split('/')[-1]} -> {next_model.split('/')[-1]}")
            log("=" * 70)

            # 1. Start preload (background)
            log(f"[PRELOAD] Starting background preload...")
            preload_start = time.time()
            standby.start_prefetch(next_model)

            # 2. Run inference while preloading
            log(f"[INFERENCE] Running 3 requests while preloading...")
            for j in range(3):
                out = llm.generate([f"Explain {j+1}: recursion briefly."],
                                   SamplingParams(max_tokens=30, temperature=0.7))

            # 3. Wait for preload to complete
            was_ready = standby.is_ready(next_model)
            if not was_ready:
                log("[PRELOAD] Waiting for completion...")
                standby.wait_for_load(timeout=120)
                was_ready = standby.is_ready(next_model)

            preload_time = time.time() - preload_start
            log(f"[PRELOAD] {'Done' if was_ready else 'FAILED'} in {preload_time:.1f}s")

            # 4. Get premerged weights
            premerged = standby.consume_standby() if was_ready else None
            if premerged:
                total_gb = sum(t.numel() * t.element_size() for t in premerged.values()) / 1e9
                log(f"[PRELOAD] Got {len(premerged)} tensors ({total_gb:.1f}GB)")

            # === SWITCHOVER (user-facing) ===
            switchover_start = time.time()

            # Cleanup
            cleanup_start = time.time()
            freed = full_cleanup(llm, nuclear=True)
            llm = None
            cleanup_time = time.time() - cleanup_start
            log(f"[CLEANUP] Freed {freed:.1f}GB in {cleanup_time:.2f}s")

            # Load
            load_start = time.time()
            if premerged:
                log(f"[INJECT] Loading from pinned memory...")
                set_preloaded_weights(premerged)
                llm = LLM(model=next_model, load_format="pinned_arena", **VLLM_KWARGS)
                standby.release_consumed(next_model)
            else:
                log(f"[COLD] Loading from disk...")
                llm = LLM(model=next_model, **VLLM_KWARGS)

            load_time = time.time() - load_start
            switchover_time = time.time() - switchover_start

            current_model = next_model
            mem = get_memory()

            # Verify
            passed = verify_model(llm, next_model)

            # Record
            results.append({
                'switch': i + 1,
                'to_model': next_model.split('/')[-1],
                'preload_time': preload_time,
                'cleanup_time': cleanup_time,
                'load_time': load_time,
                'switchover_time': switchover_time,
                'was_preloaded': was_ready and premerged is not None,
                'passed': passed,
            })

            log(f"\n[RESULT] Preload: {preload_time:.1f}s (background)")
            log(f"[RESULT] Switchover: {switchover_time:.1f}s (cleanup: {cleanup_time:.2f}s + load: {load_time:.1f}s)")

        # Summary
        log("\n" + "=" * 70)
        log("TIMING SUMMARY")
        log("=" * 70)
        log(f"{'#':>3} {'Model':<30} {'Preload':>10} {'Cleanup':>10} {'Load':>10} {'Switchover':>12}")
        log("-" * 70)

        for r in results:
            preload_str = f"{r['preload_time']:.1f}s" if r['was_preloaded'] else "cold"
            log(f"{r['switch']:3d} {r['to_model']:<30} {preload_str:>10} "
                f"{r['cleanup_time']:>9.2f}s {r['load_time']:>9.1f}s {r['switchover_time']:>11.1f}s")

        log("-" * 70)

        preloaded = [r for r in results if r['was_preloaded']]
        cold = [r for r in results if not r['was_preloaded']]

        if preloaded:
            avg_preload = sum(r['preload_time'] for r in preloaded) / len(preloaded)
            avg_switchover = sum(r['switchover_time'] for r in preloaded) / len(preloaded)
            log(f"\nPreloaded switches ({len(preloaded)}):")
            log(f"  Avg preload (background): {avg_preload:.1f}s")
            log(f"  Avg switchover (user):    {avg_switchover:.1f}s")

        if cold:
            avg_cold = sum(r['switchover_time'] for r in cold) / len(cold)
            log(f"\nCold switches ({len(cold)}): avg {avg_cold:.1f}s")

        if preloaded and cold:
            speedup = avg_cold / avg_switchover if avg_switchover > 0 else 0
            log(f"\nSpeedup from preloading: {speedup:.1f}x")

        passed = sum(1 for r in results if r['passed'])
        log(f"\nVerification: {passed}/{len(results)} passed")
        log("=" * 70)

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        log("\nCleaning up...")
        standby.shutdown()
        if llm is not None:
            try:
                full_cleanup(llm, nuclear=True)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
