#!/usr/bin/env python3
"""Long stress test: 30+ switches to find patterns and bugs.

Goals:
1. Find patterns in slow load times
2. Verify lucidity on every switch
3. Track memory drift over time
4. Identify any bugs in the standby setup
"""

import os
import gc
import sys
import time
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


@dataclass
class SwitchRecord:
    """Detailed record for a single switch."""
    switch_num: int
    from_model: str
    to_model: str
    timestamp: str

    # Preload phase (background)
    preload_start: float = 0.0
    preload_end: float = 0.0
    preload_time: float = 0.0
    preload_ready: bool = False
    preload_size_gb: float = 0.0

    # Cleanup phase
    cleanup_start: float = 0.0
    cleanup_end: float = 0.0
    cleanup_time: float = 0.0
    cleanup_freed_gb: float = 0.0

    # Load phase (broken down)
    load_start: float = 0.0
    inject_time: float = 0.0  # Weight transfer time
    inject_bandwidth_gbps: float = 0.0
    vllm_init_time: float = 0.0  # vLLM overhead
    load_end: float = 0.0
    load_time: float = 0.0

    # Total switchover (user-facing)
    switchover_time: float = 0.0

    # Memory tracking
    vram_before_cleanup_gb: float = 0.0
    vram_after_cleanup_gb: float = 0.0
    vram_after_load_gb: float = 0.0
    vram_drift_gb: float = 0.0  # Compared to baseline

    # Lucidity verification
    lucid_question: str = ""
    lucid_answer: str = ""
    lucid_expected: str = ""
    lucid_passed: bool = False
    lucid_time: float = 0.0

    # Anomaly flags
    is_slow_load: bool = False  # Load time > 2x average
    is_slow_preload: bool = False  # Preload > 2x average
    is_high_drift: bool = False  # Drift > 1GB from baseline
    notes: str = ""


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return {
        'used_gb': (total - free) / 1024**3,
        'free_gb': free / 1024**3,
        'allocated_gb': allocated / 1024**3,
        'total_gb': total / 1024**3,
    }


# Lucidity questions with expected answers
LUCIDITY_TESTS = [
    ("What is 2 + 2? Reply with just the number.", "4"),
    ("What is 3 + 5? Reply with just the number.", "8"),
    ("What is 10 - 3? Reply with just the number.", "7"),
    ("What is 6 * 2? Reply with just the number.", "12"),
    ("What is 15 / 3? Reply with just the number.", "5"),
    ("What is 100 - 1? Reply with just the number.", "99"),
    ("What is 7 + 8? Reply with just the number.", "15"),
    ("What is 20 / 4? Reply with just the number.", "5"),
]


def verify_lucidity(llm, switch_num):
    """Verify model is lucid with math question."""
    from vllm import SamplingParams

    question, expected = LUCIDITY_TESTS[switch_num % len(LUCIDITY_TESTS)]

    start = time.time()
    try:
        out = llm.generate([question], SamplingParams(max_tokens=10, temperature=0))
        answer = out[0].outputs[0].text.strip()
        elapsed = time.time() - start

        # Check if expected answer is in response
        # Handle various formats: "4", "4.", "The answer is 4", etc.
        passed = expected in answer.replace(',', '')

        return passed, question, answer, expected, elapsed
    except Exception as e:
        return False, question, str(e), expected, time.time() - start


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from blitzinfer.engine.cleanup import full_cleanup

    # Configuration
    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    MODEL_A_SHORT = "Qwen-32B-FP8"
    MODEL_B = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    MODEL_B_SHORT = "Mistral-24B"

    NUM_SWITCHES = 32
    ARENA_SIZE = 50.0

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 80)
    log("LONG STRESS TEST: 32 SWITCHES")
    log("=" * 80)
    log(f"Model A: {MODEL_A}")
    log(f"Model B: {MODEL_B}")
    log(f"Switches: {NUM_SWITCHES}")
    log(f"Arena: {ARENA_SIZE}GB")
    log("")

    records: List[SwitchRecord] = []

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
    current_short = None
    baseline_vram = None

    # Track running averages for anomaly detection
    load_times = []
    preload_times = []

    try:
        # Get baseline memory
        gc.collect()
        torch.cuda.empty_cache()
        baseline = get_memory()
        baseline_vram = baseline['used_gb']
        log(f"Baseline VRAM: {baseline_vram:.2f}GB")

        # Cold load first model
        log(f"\nCold loading {MODEL_A_SHORT}...")
        cold_start = time.time()
        llm = LLM(model=MODEL_A, **VLLM_KWARGS)
        cold_time = time.time() - cold_start
        current_model = MODEL_A
        current_short = MODEL_A_SHORT
        log(f"Cold load: {cold_time:.1f}s")

        # Verify first model
        passed, q, a, exp, t = verify_lucidity(llm, 0)
        log(f"Initial verify: '{a}' [{'PASS' if passed else 'FAIL'}]")
        if not passed:
            log(f"WARNING: Initial model not lucid! Expected '{exp}'")

        # Main switching loop
        for i in range(NUM_SWITCHES):
            # Determine next model
            if current_model == MODEL_A:
                next_model = MODEL_B
                next_short = MODEL_B_SHORT
            else:
                next_model = MODEL_A
                next_short = MODEL_A_SHORT

            log(f"\n{'='*80}")
            log(f"SWITCH {i+1}/{NUM_SWITCHES}: {current_short} -> {next_short}")
            log("=" * 80)

            record = SwitchRecord(
                switch_num=i + 1,
                from_model=current_short,
                to_model=next_short,
                timestamp=datetime.now().isoformat(),
            )

            # === PRELOAD PHASE (background) ===
            record.preload_start = time.time()
            log(f"[PRELOAD] Starting background preload of {next_short}...")
            standby.start_prefetch(next_model)

            # Run some inference while preloading
            for j in range(2):
                try:
                    out = llm.generate(
                        [f"Count from 1 to {j+3}:"],
                        SamplingParams(max_tokens=20, temperature=0)
                    )
                except Exception as e:
                    log(f"  Inference error during preload: {e}")

            # Wait for preload
            was_ready = standby.is_ready(next_model)
            if not was_ready:
                log("[PRELOAD] Waiting for completion...")
                standby.wait_for_load(timeout=180)
                was_ready = standby.is_ready(next_model)

            record.preload_end = time.time()
            record.preload_time = record.preload_end - record.preload_start
            record.preload_ready = was_ready

            preload_times.append(record.preload_time)
            avg_preload = sum(preload_times) / len(preload_times)
            record.is_slow_preload = record.preload_time > avg_preload * 2 and len(preload_times) > 3

            log(f"[PRELOAD] {'Done' if was_ready else 'FAILED'} in {record.preload_time:.1f}s")

            # Get premerged weights
            premerged = None
            if was_ready:
                premerged = standby.consume_standby()
                if premerged:
                    record.preload_size_gb = sum(
                        t.numel() * t.element_size() for t in premerged.values()
                    ) / 1e9
                    log(f"[PRELOAD] Got {len(premerged)} tensors ({record.preload_size_gb:.1f}GB)")

            # === SWITCHOVER STARTS (user-facing) ===
            record.vram_before_cleanup_gb = get_memory()['used_gb']

            # Cleanup
            record.cleanup_start = time.time()
            log(f"[CLEANUP] Freeing {current_short}...")
            freed = full_cleanup(llm, nuclear=True)
            llm = None
            record.cleanup_end = time.time()
            record.cleanup_time = record.cleanup_end - record.cleanup_start
            record.cleanup_freed_gb = freed
            record.vram_after_cleanup_gb = get_memory()['used_gb']
            log(f"[CLEANUP] Freed {freed:.1f}GB in {record.cleanup_time:.2f}s, VRAM: {record.vram_after_cleanup_gb:.2f}GB")

            # Load
            record.load_start = time.time()
            if premerged:
                log(f"[INJECT] Loading {next_short} from pinned memory...")
                set_preloaded_weights(premerged)

                inject_start = time.time()
                llm = LLM(model=next_model, load_format="pinned_arena", **VLLM_KWARGS)
                record.load_end = time.time()

                # Parse inject time from log (approximation)
                record.load_time = record.load_end - record.load_start
                record.inject_time = record.preload_size_gb / 25.0  # Approx at 25 GB/s
                record.inject_bandwidth_gbps = 25.0  # Nominal
                record.vllm_init_time = record.load_time - record.inject_time

                standby.release_consumed(next_model)
            else:
                log(f"[COLD] Loading {next_short} from disk...")
                llm = LLM(model=next_model, **VLLM_KWARGS)
                record.load_end = time.time()
                record.load_time = record.load_end - record.load_start
                record.notes = "COLD LOAD - preload failed"

            record.vram_after_load_gb = get_memory()['used_gb']
            record.vram_drift_gb = record.vram_after_load_gb - baseline_vram
            record.switchover_time = record.cleanup_time + record.load_time

            # Track for anomaly detection
            load_times.append(record.load_time)
            avg_load = sum(load_times) / len(load_times)
            record.is_slow_load = record.load_time > avg_load * 2 and len(load_times) > 3
            record.is_high_drift = abs(record.vram_drift_gb) > 1.0

            current_model = next_model
            current_short = next_short

            # === LUCIDITY CHECK ===
            passed, q, a, exp, t = verify_lucidity(llm, i + 1)
            record.lucid_question = q
            record.lucid_answer = a
            record.lucid_expected = exp
            record.lucid_passed = passed
            record.lucid_time = t

            # Flag issues
            if not passed:
                record.notes += f" LUCIDITY_FAIL"
                log(f"[VERIFY] FAIL - Q: '{q}' A: '{a}' (expected '{exp}')")
            else:
                log(f"[VERIFY] PASS - '{a}' ({t:.2f}s)")

            if record.is_slow_load:
                record.notes += f" SLOW_LOAD({record.load_time:.1f}s vs avg {avg_load:.1f}s)"
                log(f"[ANOMALY] Slow load: {record.load_time:.1f}s (avg: {avg_load:.1f}s)")

            if record.is_high_drift:
                record.notes += f" HIGH_DRIFT({record.vram_drift_gb:+.2f}GB)"
                log(f"[ANOMALY] High memory drift: {record.vram_drift_gb:+.2f}GB from baseline")

            records.append(record)

            # Summary for this switch
            log(f"\n[RESULT] Switch {i+1}: preload={record.preload_time:.1f}s, "
                f"switchover={record.switchover_time:.1f}s (cleanup={record.cleanup_time:.2f}s + load={record.load_time:.1f}s), "
                f"lucid={'PASS' if passed else 'FAIL'}")

        # === FINAL ANALYSIS ===
        log("\n" + "=" * 80)
        log("FINAL ANALYSIS")
        log("=" * 80)

        # Group by model
        qwen_records = [r for r in records if r.to_model == MODEL_A_SHORT]
        mistral_records = [r for r in records if r.to_model == MODEL_B_SHORT]

        log(f"\n{MODEL_A_SHORT} switches ({len(qwen_records)}):")
        if qwen_records:
            avg_preload = sum(r.preload_time for r in qwen_records) / len(qwen_records)
            avg_load = sum(r.load_time for r in qwen_records) / len(qwen_records)
            avg_switch = sum(r.switchover_time for r in qwen_records) / len(qwen_records)
            min_load = min(r.load_time for r in qwen_records)
            max_load = max(r.load_time for r in qwen_records)
            log(f"  Preload (bg): avg={avg_preload:.1f}s")
            log(f"  Load: avg={avg_load:.1f}s, min={min_load:.1f}s, max={max_load:.1f}s")
            log(f"  Switchover: avg={avg_switch:.1f}s")

        log(f"\n{MODEL_B_SHORT} switches ({len(mistral_records)}):")
        if mistral_records:
            avg_preload = sum(r.preload_time for r in mistral_records) / len(mistral_records)
            avg_load = sum(r.load_time for r in mistral_records) / len(mistral_records)
            avg_switch = sum(r.switchover_time for r in mistral_records) / len(mistral_records)
            min_load = min(r.load_time for r in mistral_records)
            max_load = max(r.load_time for r in mistral_records)
            log(f"  Preload (bg): avg={avg_preload:.1f}s")
            log(f"  Load: avg={avg_load:.1f}s, min={min_load:.1f}s, max={max_load:.1f}s")
            log(f"  Switchover: avg={avg_switch:.1f}s")

        # Anomalies
        slow_loads = [r for r in records if r.is_slow_load]
        slow_preloads = [r for r in records if r.is_slow_preload]
        high_drifts = [r for r in records if r.is_high_drift]
        lucidity_fails = [r for r in records if not r.lucid_passed]

        log(f"\nANOMALIES:")
        log(f"  Slow loads (>2x avg): {len(slow_loads)}")
        for r in slow_loads:
            log(f"    Switch {r.switch_num}: {r.to_model} - {r.load_time:.1f}s")

        log(f"  Slow preloads (>2x avg): {len(slow_preloads)}")
        for r in slow_preloads:
            log(f"    Switch {r.switch_num}: {r.to_model} - {r.preload_time:.1f}s")

        log(f"  High memory drift (>1GB): {len(high_drifts)}")
        for r in high_drifts:
            log(f"    Switch {r.switch_num}: {r.vram_drift_gb:+.2f}GB")

        log(f"  Lucidity failures: {len(lucidity_fails)}")
        for r in lucidity_fails:
            log(f"    Switch {r.switch_num}: {r.to_model} - Q: '{r.lucid_question}' A: '{r.lucid_answer}'")

        # Memory trend
        log(f"\nMEMORY TREND:")
        log(f"  Baseline VRAM: {baseline_vram:.2f}GB")
        log(f"  Final VRAM after cleanup: {records[-1].vram_after_cleanup_gb:.2f}GB")
        log(f"  Net drift: {records[-1].vram_after_cleanup_gb - baseline_vram:+.2f}GB")

        # Timing trend (check if load times increase over time)
        log(f"\nTIMING TREND (load times):")
        first_half = records[:len(records)//2]
        second_half = records[len(records)//2:]
        avg_first = sum(r.load_time for r in first_half) / len(first_half)
        avg_second = sum(r.load_time for r in second_half) / len(second_half)
        log(f"  First half avg: {avg_first:.1f}s")
        log(f"  Second half avg: {avg_second:.1f}s")
        if avg_second > avg_first * 1.2:
            log(f"  WARNING: Load times increasing over time (+{(avg_second/avg_first-1)*100:.0f}%)")
        else:
            log(f"  OK: No significant increase in load times")

        # Overall stats
        total_lucid_pass = sum(1 for r in records if r.lucid_passed)
        log(f"\nOVERALL:")
        log(f"  Total switches: {len(records)}")
        log(f"  Lucidity: {total_lucid_pass}/{len(records)} passed")
        log(f"  Avg switchover (user-facing): {sum(r.switchover_time for r in records)/len(records):.1f}s")
        log(f"  Avg preload (background): {sum(r.preload_time for r in records)/len(records):.1f}s")

        # Detailed timing table
        log("\n" + "=" * 80)
        log("DETAILED TIMING TABLE")
        log("=" * 80)
        log(f"{'#':>3} {'To Model':<15} {'Preload':>8} {'Cleanup':>8} {'Load':>8} {'Switch':>8} {'VRAM':>8} {'Lucid':>6} {'Notes'}")
        log("-" * 80)
        for r in records:
            lucid_str = "PASS" if r.lucid_passed else "FAIL"
            notes = r.notes.strip() if r.notes else ""
            log(f"{r.switch_num:3d} {r.to_model:<15} {r.preload_time:>7.1f}s {r.cleanup_time:>7.2f}s "
                f"{r.load_time:>7.1f}s {r.switchover_time:>7.1f}s {r.vram_after_load_gb:>7.1f}G {lucid_str:>6} {notes}")
        log("-" * 80)

        # Save detailed results to JSON
        results_file = "/tmp/long_stress_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in records], f, indent=2)
        log(f"\nDetailed results saved to: {results_file}")

        log("\n" + "=" * 80)
        if total_lucid_pass == len(records):
            log("*** ALL LUCIDITY TESTS PASSED ***")
        else:
            log(f"*** {len(records) - total_lucid_pass} LUCIDITY FAILURES ***")
        log("=" * 80)

        return 0 if total_lucid_pass == len(records) else 1

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
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


if __name__ == "__main__":
    sys.exit(main())
