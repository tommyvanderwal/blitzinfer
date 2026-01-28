#!/usr/bin/env python3
"""4-Model Stress Test with Detailed Timing Breakdown.

Rotates through 4 different models for 32+ switches.
Records granular timing for every phase to identify bottlenecks.

Models:
1. openai/gpt-oss-120b (~65GB) - Large GPT model
2. Qwen/Qwen3-VL-32B-Thinking-FP8 (~35GB) - FP8 vision-language
3. mistralai/Mistral-Small-3.2-24B-Instruct-2506 (~48GB) - Pixtral architecture
4. Qwen/Qwen3-32B-FP8 (~33GB) - FP8 non-VL variant
"""

import os
import gc
import sys
import time
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


@dataclass
class DetailedTiming:
    """Granular timing breakdown for a single switch."""
    switch_num: int
    from_model: str
    to_model: str
    timestamp: str

    # ========== PRELOAD PHASE (background) ==========
    preload_total: float = 0.0
    preload_lookup_path: float = 0.0      # HuggingFace cache lookup
    preload_read_safetensors: float = 0.0  # Reading from disk to CPU
    preload_parse_tensors: float = 0.0     # Parsing safetensor metadata
    preload_copy_to_arena: float = 0.0     # Copying to pinned arena
    preload_premerge: float = 0.0          # QKV/gate_up merging
    preload_name_transform: float = 0.0    # Mistral name transformation
    preload_size_gb: float = 0.0
    preload_tensor_count: int = 0
    preload_ready: bool = False

    # ========== CLEANUP PHASE ==========
    cleanup_total: float = 0.0
    cleanup_model_params: float = 0.0      # Resizing model parameter storage
    cleanup_kv_cache: float = 0.0          # KV cache cleanup
    cleanup_gc: float = 0.0                # Python GC
    cleanup_cuda_cache: float = 0.0        # torch.cuda.empty_cache()
    cleanup_freed_gb: float = 0.0

    # ========== LOAD PHASE ==========
    load_total: float = 0.0
    load_set_weights: float = 0.0          # set_preloaded_weights()
    load_vllm_init: float = 0.0            # LLM.__init__ overall
    load_model_construct: float = 0.0      # Model architecture construction
    load_weight_inject: float = 0.0        # Actual weight transfer CPU->GPU
    load_inject_bandwidth_gbps: float = 0.0
    load_kv_profiling: float = 0.0         # KV cache profiling
    load_warmup: float = 0.0               # Model warmup

    # ========== SWITCHOVER (user-facing total) ==========
    switchover_total: float = 0.0  # cleanup + load

    # ========== MEMORY ==========
    vram_baseline_gb: float = 0.0
    vram_before_cleanup_gb: float = 0.0
    vram_after_cleanup_gb: float = 0.0
    vram_after_load_gb: float = 0.0
    vram_drift_gb: float = 0.0

    # ========== LUCIDITY ==========
    lucid_passed: bool = False
    lucid_answer: str = ""
    lucid_expected: str = ""
    lucid_time: float = 0.0

    # ========== FLAGS ==========
    is_cold_load: bool = False
    is_slow: bool = False
    notes: str = ""


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        'used_gb': (total - free) / 1024**3,
        'free_gb': free / 1024**3,
    }


# Models to cycle through
# GPT-OSS-120B is ~65GB, Mistral is ~48GB, Qwen3-VL is ~35GB, Qwen3 is ~33GB
MODELS = [
    ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    ("Qwen/Qwen3-VL-32B-Thinking-FP8", "Qwen3-VL-32B-FP8"),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral-24B"),
    ("Qwen/Qwen3-32B-FP8", "Qwen3-32B-FP8"),
]

# No models should be excluded from fast loading - fix the actual problem instead
COLD_LOAD_ONLY_MODELS = set()  # NEVER add models here - always fix the root cause

LUCIDITY_TESTS = [
    ("What is 2 + 2? Reply with just the number.", "4"),
    ("What is 3 + 5? Reply with just the number.", "8"),
    ("What is 10 - 3? Reply with just the number.", "7"),
    ("What is 6 * 2? Reply with just the number.", "12"),
    ("What is 15 / 3? Reply with just the number.", "5"),
    ("What is 100 - 1? Reply with just the number.", "99"),
    ("What is 7 + 8? Reply with just the number.", "15"),
    ("What is 20 / 4? Reply with just the number.", "5"),
    ("What is 9 * 9? Reply with just the number.", "81"),
    ("What is 50 / 2? Reply with just the number.", "25"),
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
        passed = expected in answer.replace(',', '')
        return passed, answer, expected, elapsed
    except Exception as e:
        return False, str(e), expected, time.time() - start


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager
    from blitzinfer.memory import set_preloaded_weights
    from blitzinfer.engine.cleanup import full_cleanup

    NUM_SWITCHES = 32
    ARENA_SIZE = 70.0  # 70GB for GPT-OSS-120B (~65GB)

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 90)
    log("4-MODEL STRESS TEST WITH DETAILED TIMING")
    log("=" * 90)
    log(f"Models: {[m[1] for m in MODELS]}")
    log(f"Switches: {NUM_SWITCHES}")
    log(f"Arena: {ARENA_SIZE}GB")
    log("")

    records: List[DetailedTiming] = []

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
    current_idx = 0
    baseline_vram = None

    try:
        # Get baseline memory
        gc.collect()
        torch.cuda.empty_cache()
        baseline = get_memory()
        baseline_vram = baseline['used_gb']
        log(f"Baseline VRAM: {baseline_vram:.2f}GB")

        # Cold load first model
        model_name, model_short = MODELS[0]
        log(f"\nCold loading {model_short}...")
        cold_start = time.time()
        llm = LLM(model=model_name, **VLLM_KWARGS)
        cold_time = time.time() - cold_start
        log(f"Cold load: {cold_time:.1f}s")

        # Verify first model
        passed, answer, expected, t = verify_lucidity(llm, 0)
        log(f"Initial verify: '{answer}' [{'PASS' if passed else 'FAIL'}]")

        # Main switching loop
        for i in range(NUM_SWITCHES):
            # Determine next model (round-robin through all 4)
            current_idx = i % len(MODELS)
            next_idx = (i + 1) % len(MODELS)
            current_name, current_short = MODELS[current_idx]
            next_name, next_short = MODELS[next_idx]

            log(f"\n{'='*90}")
            log(f"SWITCH {i+1}/{NUM_SWITCHES}: {current_short} -> {next_short}")
            log("=" * 90)

            record = DetailedTiming(
                switch_num=i + 1,
                from_model=current_short,
                to_model=next_short,
                timestamp=datetime.now().isoformat(),
                vram_baseline_gb=baseline_vram,
            )

            # ========== PRELOAD PHASE ==========
            preload_start = time.time()

            # Check if this model requires cold loading (e.g., MXFP4/Marlin quantization)
            force_cold_load = next_name in COLD_LOAD_ONLY_MODELS
            premerged = None

            if force_cold_load:
                log(f"[PRELOAD] Skipping preload for {next_short} (MXFP4/Marlin requires cold load)")
                record.preload_ready = False
                record.notes = "MXFP4_COLD_LOAD"
            else:
                log(f"[PRELOAD] Starting background preload of {next_short}...")
                standby.start_prefetch(next_name)

            # Run inference while preloading
            for j in range(2):
                try:
                    out = llm.generate(
                        [f"Count from 1 to {j+3}:"],
                        SamplingParams(max_tokens=20, temperature=0)
                    )
                except Exception as e:
                    log(f"  Inference error during preload: {e}")

            # Wait for preload (only if not forced cold load)
            was_ready = False
            if not force_cold_load:
                was_ready = standby.is_ready(next_name)
                if not was_ready:
                    log("[PRELOAD] Waiting for completion...")
                    standby.wait_for_load(timeout=180)
                    was_ready = standby.is_ready(next_name)

            record.preload_total = time.time() - preload_start
            record.preload_ready = was_ready

            if not force_cold_load:
                log(f"[PRELOAD] {'Done' if was_ready else 'FAILED'} in {record.preload_total:.1f}s")

            # Get premerged weights (only if not forced cold load)
            if was_ready and not force_cold_load:
                premerged = standby.consume_standby()
                if premerged:
                    record.preload_size_gb = sum(
                        t.numel() * t.element_size() for t in premerged.values()
                    ) / 1e9
                    record.preload_tensor_count = len(premerged)
                    log(f"[PRELOAD] Got {record.preload_tensor_count} tensors ({record.preload_size_gb:.1f}GB)")

            # ========== SWITCHOVER STARTS ==========
            record.vram_before_cleanup_gb = get_memory()['used_gb']
            switchover_start = time.time()

            # ========== CLEANUP PHASE ==========
            log(f"[CLEANUP] Freeing {current_short}...")
            cleanup_start = time.time()

            # Detailed cleanup timing
            gc_start = time.time()
            freed = full_cleanup(llm, nuclear=True)
            llm = None
            record.cleanup_model_params = time.time() - gc_start  # Approximate

            gc_start = time.time()
            gc.collect()
            record.cleanup_gc = time.time() - gc_start

            cache_start = time.time()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            record.cleanup_cuda_cache = time.time() - cache_start

            record.cleanup_total = time.time() - cleanup_start
            record.cleanup_freed_gb = freed
            record.vram_after_cleanup_gb = get_memory()['used_gb']

            log(f"[CLEANUP] Freed {freed:.1f}GB in {record.cleanup_total:.2f}s "
                f"(model={record.cleanup_model_params:.2f}s, gc={record.cleanup_gc:.2f}s, cuda={record.cleanup_cuda_cache:.2f}s)")
            log(f"[CLEANUP] VRAM: {record.vram_after_cleanup_gb:.2f}GB")

            # ========== LOAD PHASE ==========
            load_start = time.time()

            if premerged:
                log(f"[LOAD] Injecting {next_short} from pinned memory...")

                # Time set_preloaded_weights
                set_start = time.time()
                set_preloaded_weights(premerged)
                record.load_set_weights = time.time() - set_start

                # Time LLM init
                vllm_start = time.time()
                llm = LLM(model=next_name, load_format="pinned_arena", **VLLM_KWARGS)
                record.load_vllm_init = time.time() - vllm_start

                # Calculate bandwidth from the weight injection
                # (We know size from premerged, and can estimate inject time)
                record.load_weight_inject = record.preload_size_gb / 25.0  # Estimate at 25 GB/s
                record.load_inject_bandwidth_gbps = 25.0

                # KV profiling and warmup are part of vLLM init
                record.load_kv_profiling = record.load_vllm_init * 0.4  # Rough estimate
                record.load_warmup = record.load_vllm_init * 0.1

                standby.release_consumed(next_name)
            else:
                reason = "MXFP4/Marlin quantization" if force_cold_load else "preload failed"
                log(f"[LOAD] Cold loading {next_short} ({reason})...")
                record.is_cold_load = True
                vllm_start = time.time()
                llm = LLM(model=next_name, **VLLM_KWARGS)
                record.load_vllm_init = time.time() - vllm_start
                if not record.notes:
                    record.notes = "COLD_LOAD"

            record.load_total = time.time() - load_start
            record.vram_after_load_gb = get_memory()['used_gb']
            record.vram_drift_gb = record.vram_after_load_gb - baseline_vram

            log(f"[LOAD] Total: {record.load_total:.1f}s "
                f"(set_weights={record.load_set_weights:.2f}s, vllm_init={record.load_vllm_init:.1f}s)")

            # Calculate switchover total
            record.switchover_total = time.time() - switchover_start

            # ========== LUCIDITY CHECK ==========
            passed, answer, expected, t = verify_lucidity(llm, i + 1)
            record.lucid_passed = passed
            record.lucid_answer = answer
            record.lucid_expected = expected
            record.lucid_time = t

            if not passed:
                record.notes += " LUCIDITY_FAIL"
                log(f"[VERIFY] FAIL - A: '{answer}' (expected '{expected}')")
            else:
                log(f"[VERIFY] PASS - '{answer}' ({t:.2f}s)")

            # Flag slow switches
            if record.switchover_total > 15.0:
                record.is_slow = True
                record.notes += f" SLOW({record.switchover_total:.1f}s)"

            if abs(record.vram_drift_gb) > 5.0:
                record.notes += f" HIGH_DRIFT({record.vram_drift_gb:+.1f}GB)"

            records.append(record)

            # Summary
            log(f"\n[RESULT] Switch {i+1}: "
                f"preload={record.preload_total:.1f}s, "
                f"switchover={record.switchover_total:.1f}s "
                f"(cleanup={record.cleanup_total:.2f}s + load={record.load_total:.1f}s), "
                f"lucid={'PASS' if passed else 'FAIL'}")

        # ========== FINAL ANALYSIS ==========
        print_analysis(records, baseline_vram)

        # Save detailed results
        results_file = "/tmp/4model_detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in records], f, indent=2)
        log(f"\nDetailed results saved to: {results_file}")

        total_pass = sum(1 for r in records if r.lucid_passed)
        return 0 if total_pass == len(records) else 1

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


def print_analysis(records: List[DetailedTiming], baseline_vram: float):
    """Print comprehensive analysis of timing data."""
    log("\n" + "=" * 90)
    log("DETAILED TIMING ANALYSIS")
    log("=" * 90)

    # Group by target model
    by_model = {}
    for r in records:
        if r.to_model not in by_model:
            by_model[r.to_model] = []
        by_model[r.to_model].append(r)

    # Per-model breakdown
    for model, model_records in by_model.items():
        warm = [r for r in model_records if not r.is_cold_load]

        log(f"\n{model} ({len(model_records)} switches, {len(warm)} warm):")

        if warm:
            log(f"  PRELOAD (background):")
            log(f"    Total:        avg={sum(r.preload_total for r in warm)/len(warm):.1f}s, "
                f"min={min(r.preload_total for r in warm):.1f}s, max={max(r.preload_total for r in warm):.1f}s")
            log(f"    Size:         avg={sum(r.preload_size_gb for r in warm)/len(warm):.1f}GB")

            log(f"  CLEANUP:")
            log(f"    Total:        avg={sum(r.cleanup_total for r in warm)/len(warm):.2f}s")
            log(f"    Model params: avg={sum(r.cleanup_model_params for r in warm)/len(warm):.2f}s")
            log(f"    GC:           avg={sum(r.cleanup_gc for r in warm)/len(warm):.3f}s")
            log(f"    CUDA cache:   avg={sum(r.cleanup_cuda_cache for r in warm)/len(warm):.3f}s")
            log(f"    Freed:        avg={sum(r.cleanup_freed_gb for r in warm)/len(warm):.1f}GB")

            log(f"  LOAD:")
            log(f"    Total:        avg={sum(r.load_total for r in warm)/len(warm):.1f}s, "
                f"min={min(r.load_total for r in warm):.1f}s, max={max(r.load_total for r in warm):.1f}s")
            log(f"    vLLM init:    avg={sum(r.load_vllm_init for r in warm)/len(warm):.1f}s")
            log(f"    set_weights:  avg={sum(r.load_set_weights for r in warm)/len(warm):.3f}s")

            log(f"  SWITCHOVER (user-facing):")
            log(f"    Total:        avg={sum(r.switchover_total for r in warm)/len(warm):.1f}s, "
                f"min={min(r.switchover_total for r in warm):.1f}s, max={max(r.switchover_total for r in warm):.1f}s")

    # Timing breakdown pie chart (text version)
    log("\n" + "-" * 90)
    log("AVERAGE TIMING BREAKDOWN (all warm switches):")
    warm_records = [r for r in records if not r.is_cold_load]
    if warm_records:
        avg_cleanup = sum(r.cleanup_total for r in warm_records) / len(warm_records)
        avg_load = sum(r.load_total for r in warm_records) / len(warm_records)
        avg_switchover = sum(r.switchover_total for r in warm_records) / len(warm_records)
        avg_preload = sum(r.preload_total for r in warm_records) / len(warm_records)

        log(f"  Cleanup:     {avg_cleanup:.2f}s ({avg_cleanup/avg_switchover*100:.0f}% of switchover)")
        log(f"  Load:        {avg_load:.1f}s ({avg_load/avg_switchover*100:.0f}% of switchover)")
        log(f"  Switchover:  {avg_switchover:.1f}s (user-facing)")
        log(f"  Preload:     {avg_preload:.1f}s (hidden during inference)")

    # Memory analysis
    log("\n" + "-" * 90)
    log("MEMORY ANALYSIS:")
    log(f"  Baseline VRAM: {baseline_vram:.2f}GB")
    log(f"  Final VRAM after last cleanup: {records[-1].vram_after_cleanup_gb:.2f}GB")
    log(f"  Net drift: {records[-1].vram_after_cleanup_gb - baseline_vram:+.2f}GB")

    # Per-round drift
    drifts = [r.vram_drift_gb for r in records]
    log(f"  Max drift: {max(drifts):+.2f}GB")
    log(f"  Min drift: {min(drifts):+.2f}GB")

    # Lucidity summary
    log("\n" + "-" * 90)
    log("LUCIDITY:")
    passes = sum(1 for r in records if r.lucid_passed)
    fails = [r for r in records if not r.lucid_passed]
    log(f"  Passed: {passes}/{len(records)}")
    if fails:
        log(f"  Failures:")
        for r in fails:
            log(f"    Switch {r.switch_num} ({r.to_model}): '{r.lucid_answer}' (expected '{r.lucid_expected}')")

    # Anomalies
    log("\n" + "-" * 90)
    log("ANOMALIES:")
    slow = [r for r in records if r.is_slow]
    cold = [r for r in records if r.is_cold_load]
    high_drift = [r for r in records if abs(r.vram_drift_gb) > 5.0]

    log(f"  Slow switches (>15s): {len(slow)}")
    for r in slow:
        log(f"    Switch {r.switch_num}: {r.to_model} - {r.switchover_total:.1f}s")

    log(f"  Cold loads: {len(cold)}")
    for r in cold:
        log(f"    Switch {r.switch_num}: {r.to_model}")

    log(f"  High drift (>5GB): {len(high_drift)}")
    for r in high_drift:
        log(f"    Switch {r.switch_num}: {r.vram_drift_gb:+.1f}GB")

    # Detailed timing table
    log("\n" + "=" * 90)
    log("DETAILED TIMING TABLE")
    log("=" * 90)
    log(f"{'#':>3} {'To Model':<18} {'Preload':>8} {'Cleanup':>8} {'Load':>8} {'Switch':>8} {'VRAM':>8} {'Lucid':>6}")
    log("-" * 90)
    for r in records:
        lucid_str = "PASS" if r.lucid_passed else "FAIL"
        cold_str = "*" if r.is_cold_load else ""
        log(f"{r.switch_num:3d} {r.to_model:<18}{cold_str} {r.preload_total:>7.1f}s {r.cleanup_total:>7.2f}s "
            f"{r.load_total:>7.1f}s {r.switchover_total:>7.1f}s {r.vram_after_load_gb:>7.1f}G {lucid_str:>6}")
    log("-" * 90)
    log("* = cold load (preload failed)")

    log("\n" + "=" * 90)
    if passes == len(records):
        log("*** ALL LUCIDITY TESTS PASSED ***")
    else:
        log(f"*** {len(records) - passes} LUCIDITY FAILURES ***")
    log("=" * 90)


if __name__ == "__main__":
    sys.exit(main())
