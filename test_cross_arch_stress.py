#!/usr/bin/env python3
"""Stress test cross-architecture switching with multiple rounds.

This verifies the cleanup fix holds up over multiple switches and
that there's no memory accumulation.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
    }


def log_mem(label):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")
    return m


def check_model_lucidity(llm, model_name):
    """Verify model is lucid with basic tests.

    Returns (is_lucid, test_results_string)
    """
    from vllm import SamplingParams

    tests_passed = 0
    total_tests = 3
    details = []

    # Test 1: Basic arithmetic
    out1 = llm.generate(["What is 7 + 8? Answer with just the number:"],
                        SamplingParams(max_tokens=20, temperature=0.0))
    answer1 = out1[0].outputs[0].text.strip()
    if "15" in answer1:
        tests_passed += 1
        details.append(f"math=OK")
    else:
        details.append(f"math=FAIL({answer1[:20]})")

    # Test 2: Knowledge
    out2 = llm.generate(["Complete: The capital of France is"],
                        SamplingParams(max_tokens=20, temperature=0.0))
    answer2 = out2[0].outputs[0].text.strip()
    if "paris" in answer2.lower():
        tests_passed += 1
        details.append(f"knowledge=OK")
    else:
        details.append(f"knowledge=FAIL({answer2[:20]})")

    # Test 3: Self-awareness
    out3 = llm.generate(["Are you an AI? Answer yes or no:"],
                        SamplingParams(max_tokens=20, temperature=0.0))
    answer3 = out3[0].outputs[0].text.strip()
    if "yes" in answer3.lower():
        tests_passed += 1
        details.append(f"self=OK")
    else:
        details.append(f"self=FAIL({answer3[:20]})")

    is_lucid = tests_passed == total_tests
    return is_lucid, f"{tests_passed}/{total_tests} [{', '.join(details)}]"


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup, nuclear_cleanup

    # Two completely different architectures
    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # Vision-language model, FP8
    MODEL_B = "openai/gpt-oss-120b"              # Pure LLM, MXFP4

    NUM_ROUNDS = 5

    print("=" * 70)
    print(f"CROSS-ARCHITECTURE STRESS TEST ({NUM_ROUNDS} rounds)")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print("Using full_cleanup with nuclear=True for minimal drift")
    print()

    baseline = log_mem("baseline")
    baseline_used = baseline['used_gb']

    results = []
    llm = None

    try:
        for round_num in range(NUM_ROUNDS):
            print(f"\n{'='*70}")
            print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
            print("=" * 70)

            # Load Model A
            print(f"\n--- Loading Model A (Qwen-32B) ---")
            t0 = time.perf_counter()
            llm = LLM(
                model=MODEL_A,
                dtype="bfloat16",
                max_model_len=131072,  # Full 128K context (Qwen3-VL supports 128K)
                gpu_memory_utilization=0.95,  # Use 90+ GB of VRAM
                enforce_eager=True,
                trust_remote_code=True,
            )
            load_a_time = time.perf_counter() - t0
            print(f"Load time: {load_a_time:.1f}s")
            log_mem("after A load")

            # Warmup inference
            t_warmup = time.perf_counter()
            out = llm.generate([f"Round {round_num+1}: 2+2="], SamplingParams(max_tokens=20))
            warmup_a_time = time.perf_counter() - t_warmup
            text_a = out[0].outputs[0].text.strip()[:40]
            print(f"Warmup time: {warmup_a_time:.2f}s")
            print(f"Output: {text_a}")

            # Lucidity check
            t_lucid = time.perf_counter()
            is_lucid_a, lucid_details_a = check_model_lucidity(llm, MODEL_A)
            lucid_a_time = time.perf_counter() - t_lucid
            print(f"Lucidity: {'PASS' if is_lucid_a else 'FAIL'} {lucid_details_a} ({lucid_a_time:.2f}s)")

            # Cleanup A - use nuclear=True for minimal drift
            t_cleanup = time.perf_counter()
            mem_before_cleanup = get_gpu_memory()
            freed_a = full_cleanup(llm, nuclear=True)
            llm = None

            # Extra sync to ensure memory is fully released
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.5)  # Brief delay for CUDA to finalize
            cleanup_a_time = time.perf_counter() - t_cleanup

            mem_after_a = log_mem("after A cleanup")
            drift_a = mem_after_a['used_gb'] - baseline_used
            print(f"Cleanup time: {cleanup_a_time:.2f}s, Freed: {freed_a:.1f}GB, Drift: {drift_a:+.2f}GB")

            # Load Model B
            # Use conservative utilization - Model B (gpt-oss-120b) uses 66GB
            # Need headroom for KV cache profiling which requires extra memory
            mem = get_gpu_memory()
            available = mem['free_gb']
            # gpt-oss-120b needs 66GB for weights + temp memory during loading (~8GB)
            # Use high utilization (0.90) to ensure enough room for KV cache
            max_util = 0.90
            print(f"\n--- Loading Model B (gpt-oss-120b) ---")
            print(f"Available: {available:.1f}GB, using utilization: {max_util:.2f}")

            if available < 75:
                print(f"WARNING: Not enough memory! Available {available:.1f}GB < 75GB needed")
                print("Skipping Model B this round")
                results.append({
                    'round': round_num + 1,
                    # Timing
                    'load_a': load_a_time,
                    'load_b': 0,
                    'warmup_a': warmup_a_time,
                    'warmup_b': 0,
                    'cleanup_a': cleanup_a_time,
                    'cleanup_b': 0,
                    'lucid_a_time': lucid_a_time,
                    'lucid_b_time': 0,
                    # Memory
                    'freed_a': freed_a,
                    'freed_b': 0,
                    'used_after_a': mem_after_a['used_gb'],
                    'used_after_b': mem_after_a['used_gb'],
                    'drift_a': drift_a,
                    'drift_b': drift_a,  # Same as A since B was skipped
                    # Lucidity
                    'lucid_a': is_lucid_a,
                    'lucid_b': False,
                    'ok_a': is_lucid_a,
                    'ok_b': False,
                })
                continue

            t0 = time.perf_counter()
            llm = LLM(
                model=MODEL_B,
                dtype="bfloat16",
                max_model_len=131072,  # Full 128K+ context (gpt-oss-120b supports 130K)
                gpu_memory_utilization=max_util,
                enforce_eager=True,
                trust_remote_code=True,
            )
            load_b_time = time.perf_counter() - t0
            print(f"Load time: {load_b_time:.1f}s")
            log_mem("after B load")

            # Warmup inference
            t_warmup = time.perf_counter()
            out = llm.generate([f"Round {round_num+1}: 3+3="], SamplingParams(max_tokens=20))
            warmup_b_time = time.perf_counter() - t_warmup
            text_b = out[0].outputs[0].text.strip()[:40]
            print(f"Warmup time: {warmup_b_time:.2f}s")
            print(f"Output: {text_b}")

            # Lucidity check
            t_lucid = time.perf_counter()
            is_lucid_b, lucid_details_b = check_model_lucidity(llm, MODEL_B)
            lucid_b_time = time.perf_counter() - t_lucid
            print(f"Lucidity: {'PASS' if is_lucid_b else 'FAIL'} {lucid_details_b} ({lucid_b_time:.2f}s)")

            # Cleanup B - use nuclear=True for minimal drift
            t_cleanup = time.perf_counter()
            freed_b = full_cleanup(llm, nuclear=True)
            llm = None

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.5)
            cleanup_b_time = time.perf_counter() - t_cleanup

            mem_after_b = log_mem("after B cleanup")
            drift_b = mem_after_b['used_gb'] - baseline_used
            print(f"Cleanup time: {cleanup_b_time:.2f}s, Freed: {freed_b:.1f}GB, Drift: {drift_b:+.2f}GB")

            results.append({
                'round': round_num + 1,
                # Timing
                'load_a': load_a_time,
                'load_b': load_b_time,
                'warmup_a': warmup_a_time,
                'warmup_b': warmup_b_time,
                'cleanup_a': cleanup_a_time,
                'cleanup_b': cleanup_b_time,
                'lucid_a_time': lucid_a_time,
                'lucid_b_time': lucid_b_time,
                # Memory
                'freed_a': freed_a,
                'freed_b': freed_b,
                'used_after_a': mem_after_a['used_gb'],
                'used_after_b': mem_after_b['used_gb'],
                'drift_a': drift_a,
                'drift_b': drift_b,
                # Lucidity
                'lucid_a': is_lucid_a,
                'lucid_b': is_lucid_b,
                'ok_a': is_lucid_a,
                'ok_b': is_lucid_b,
            })

    except Exception as e:
        print(f"\nERROR in round {round_num + 1}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if llm is not None:
            full_cleanup(llm)
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    # Detailed timing table
    print("\n--- TIMING SUMMARY ---")
    print(f"{'Round':<6} {'Load A':<9} {'Load B':<9} {'Clean A':<9} {'Clean B':<9} {'Total':<9}")
    print("-" * 70)
    for r in results:
        total = r['load_a'] + r['load_b'] + r['cleanup_a'] + r['cleanup_b']
        print(f"{r['round']:<6} {r['load_a']:.1f}s{'':<5} {r['load_b']:.1f}s{'':<5} "
              f"{r['cleanup_a']:.1f}s{'':<5} {r['cleanup_b']:.1f}s{'':<5} {total:.1f}s")

    # Averages
    if results:
        avg_load_a = sum(r['load_a'] for r in results) / len(results)
        avg_load_b = sum(r['load_b'] for r in results) / len(results)
        avg_clean_a = sum(r['cleanup_a'] for r in results) / len(results)
        avg_clean_b = sum(r['cleanup_b'] for r in results) / len(results)
        avg_total = avg_load_a + avg_load_b + avg_clean_a + avg_clean_b
        print("-" * 70)
        print(f"{'AVG':<6} {avg_load_a:.1f}s{'':<5} {avg_load_b:.1f}s{'':<5} "
              f"{avg_clean_a:.1f}s{'':<5} {avg_clean_b:.1f}s{'':<5} {avg_total:.1f}s")

    # Lucidity summary
    print("\n--- LUCIDITY SUMMARY ---")
    print(f"{'Round':<6} {'Model A':<12} {'Model B':<12} {'Status'}")
    print("-" * 70)
    for r in results:
        a_status = "LUCID" if r['lucid_a'] else "NOT LUCID"
        b_status = "LUCID" if r['lucid_b'] else "NOT LUCID"
        overall = "OK" if (r['lucid_a'] and r['lucid_b']) else "FAIL"
        print(f"{r['round']:<6} {a_status:<12} {b_status:<12} {overall}")

    lucid_a_count = sum(1 for r in results if r['lucid_a'])
    lucid_b_count = sum(1 for r in results if r['lucid_b'])
    print(f"\nModel A lucidity: {lucid_a_count}/{len(results)} rounds")
    print(f"Model B lucidity: {lucid_b_count}/{len(results)} rounds")

    # Memory drift table
    print("\n--- MEMORY DRIFT ---")
    print(f"{'Round':<6} {'Drift A':<12} {'Drift B':<12} {'Status'}")
    print("-" * 70)
    for r in results:
        status = "OK" if (r['ok_a'] and r['ok_b']) else "FAIL"
        print(f"{r['round']:<6} {r['drift_a']:+.2f}GB{'':<5} {r['drift_b']:+.2f}GB{'':<5} {status}")

    # Check for memory drift
    final = log_mem("final")
    total_drift = final['used_gb'] - baseline_used

    print(f"\n{'='*70}")
    print("MEMORY DRIFT ANALYSIS")
    print("=" * 70)
    print(f"Baseline: {baseline_used:.2f}GB")
    print(f"Final:    {final['used_gb']:.2f}GB")
    print(f"Total drift: {total_drift:+.2f}GB over {len(results)} rounds")

    if results:
        # Calculate per-round drift (change from previous round)
        per_round_drifts = []
        for i, r in enumerate(results):
            if i == 0:
                per_round_drifts.append(r['drift_b'])
            else:
                per_round_drifts.append(r['drift_b'] - results[i-1]['drift_b'])

        avg_drift_per_round = sum(per_round_drifts) / len(per_round_drifts) if per_round_drifts else 0
        print(f"Avg drift per round: {avg_drift_per_round:+.3f}GB")

        # Extrapolate to 100 switches
        drift_100 = avg_drift_per_round * 100
        print(f"Projected drift over 100 switches: {drift_100:+.1f}GB")

    if total_drift < 1.0:
        print("\nMEMORY DRIFT: EXCELLENT (< 1GB total)")
    elif total_drift < 3.0:
        print("\nMEMORY DRIFT: GOOD (< 3GB total)")
    elif total_drift < 5.0:
        print("\nMEMORY DRIFT: ACCEPTABLE (< 5GB total)")
    else:
        print("\nMEMORY DRIFT: WARNING - significant drift detected!")

    # Overall stats
    passed = sum(1 for r in results if r['ok_a'] and r['ok_b'])
    total = len(results)

    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"\n  Rounds passed (lucid): {passed}/{total}")

    if results:
        avg_load_a = sum(r['load_a'] for r in results) / len(results)
        avg_load_b = sum(r['load_b'] for r in results if r['load_b'] > 0) / max(1, sum(1 for r in results if r['load_b'] > 0))
        avg_clean_a = sum(r['cleanup_a'] for r in results) / len(results)
        avg_clean_b = sum(r['cleanup_b'] for r in results if r['cleanup_b'] > 0) / max(1, sum(1 for r in results if r['cleanup_b'] > 0))

        print(f"\n  Average load time Model A: {avg_load_a:.1f}s")
        print(f"  Average load time Model B: {avg_load_b:.1f}s")
        print(f"  Average cleanup time Model A: {avg_clean_a:.1f}s")
        print(f"  Average cleanup time Model B: {avg_clean_b:.1f}s")

        # Total switch time (cleanup A + load B) represents time to switch from A to B
        switch_times = [(r['cleanup_a'] + r['load_b']) for r in results if r['load_b'] > 0]
        if switch_times:
            avg_switch = sum(switch_times) / len(switch_times)
            print(f"\n  Average switch time (A->B): {avg_switch:.1f}s (cleanup A + load B)")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
