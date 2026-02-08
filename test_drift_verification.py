#!/usr/bin/env python3
"""Verify memory drift stays bounded over many switches.

Test 8 rounds and check drift at rounds 4 and 8 to confirm
the phantom block fix is working correctly.
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return {
        'used_gb': (total - free) / 1024**3,
        'free_gb': free / 1024**3,
        'allocated_gb': allocated / 1024**3,
    }


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    ROUNDS = 8

    print("=" * 70)
    print(f"DRIFT VERIFICATION TEST ({ROUNDS} rounds)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Target: Drift stays under 1.3GB total")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = get_memory()
    print(f"\n[BASELINE] Used: {baseline['used_gb']:.2f}GB, Allocated: {baseline['allocated_gb']:.3f}GB")

    drift_history = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}/{ROUNDS}")
        print("=" * 70)

        # Load model
        print(f"\n--- Loading model ---")
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True,
        )

        # Generate
        out = llm.generate(["Count to 5"], SamplingParams(max_tokens=20))
        _ = out[0].outputs[0].text

        after_load = get_memory()
        print(f"[after load] Used: {after_load['used_gb']:.2f}GB")

        # Cleanup
        print(f"\n--- Cleanup ---")
        freed = full_cleanup(llm, nuclear=True)
        llm = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after_cleanup = get_memory()
        drift = after_cleanup['used_gb'] - baseline['used_gb']
        drift_history.append(drift)

        print(f"[after cleanup] Used: {after_cleanup['used_gb']:.2f}GB, Allocated: {after_cleanup['allocated_gb']:.3f}GB")
        print(f"[drift from baseline] +{drift:.2f}GB")

        # Check at key points
        if round_num == 4:
            print(f"\n*** CHECKPOINT at round 4: Drift = +{drift:.2f}GB ***")
            if drift > 1.3:
                print(f"*** WARNING: Drift exceeds 1.3GB target! ***")

        if round_num == 8:
            print(f"\n*** CHECKPOINT at round 8: Drift = +{drift:.2f}GB ***")
            if drift > 1.3:
                print(f"*** WARNING: Drift exceeds 1.3GB target! ***")

    # Summary
    print("\n" + "=" * 70)
    print("DRIFT VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\nBaseline: {baseline['used_gb']:.2f}GB")
    print(f"\nDrift per round:")
    for i, d in enumerate(drift_history, 1):
        status = "OK" if d <= 1.3 else "EXCEEDS TARGET"
        print(f"  Round {i}: +{d:.2f}GB [{status}]")

    final = get_memory()
    total_drift = final['used_gb'] - baseline['used_gb']

    print(f"\nFinal state:")
    print(f"  Used: {final['used_gb']:.2f}GB")
    print(f"  Allocated: {final['allocated_gb']:.3f}GB")
    print(f"  Total drift: +{total_drift:.2f}GB")

    # Check if drift is bounded or accumulating
    drift_round_1 = drift_history[0]
    drift_round_8 = drift_history[-1]
    accumulation = drift_round_8 - drift_round_1

    print(f"\nDrift analysis:")
    print(f"  Round 1 drift: +{drift_round_1:.2f}GB")
    print(f"  Round 8 drift: +{drift_round_8:.2f}GB")
    print(f"  Accumulation (R8 - R1): +{accumulation:.2f}GB")

    if accumulation < 0.1:
        print(f"\n*** PASS: Drift is bounded (not accumulating) ***")
        if total_drift <= 1.3:
            print(f"*** PASS: Total drift ({total_drift:.2f}GB) under 1.3GB target ***")
            return 0
        else:
            print(f"*** FAIL: Total drift ({total_drift:.2f}GB) exceeds 1.3GB target ***")
            return 1
    else:
        per_round_accumulation = accumulation / 7  # rounds 2-8
        projected_100 = drift_round_1 + (per_round_accumulation * 99)
        print(f"\n*** WARNING: Drift is accumulating! ***")
        print(f"  Per-round accumulation: +{per_round_accumulation:.3f}GB")
        print(f"  Projected 100-switch drift: +{projected_100:.1f}GB")
        return 1


if __name__ == "__main__":
    sys.exit(main())
