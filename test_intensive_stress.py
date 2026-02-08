#!/usr/bin/env python3
"""Intensive stress test with 20+ switches, 5+ models, and pinned arena prefetch.

Tests edge cases:
- Rapid switching between diverse architectures
- Interrupted prefetch
- Memory pressure scenarios
- Model lucidity after many switches
- Prefetch timing vs switch timing

Uses StandbyManager with 80GB pinned arena (5x16GB chunks).
"""

import os
import gc
import time
import sys
import random

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_mem():
    """Get GPU memory used in GB."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def log_mem(label):
    m = get_mem()
    print(f"[{label}] GPU: {m:.2f} GB used")
    return m


def check_lucidity(llm, model_name, test_num):
    """Quick lucidity check - 3 tests."""
    from vllm import SamplingParams

    tests_passed = 0
    details = []

    # Test 1: Math
    out = llm.generate(["What is 7 + 8? Answer with just the number:"],
                       SamplingParams(max_tokens=20, temperature=0.0))
    ans = out[0].outputs[0].text.strip()
    if "15" in ans:
        tests_passed += 1
        details.append("math=OK")
    else:
        details.append(f"math=FAIL({ans[:15]})")

    # Test 2: Knowledge
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=20, temperature=0.0))
    ans = out[0].outputs[0].text.strip()
    if "paris" in ans.lower():
        tests_passed += 1
        details.append("know=OK")
    else:
        details.append(f"know=FAIL({ans[:15]})")

    # Test 3: Self
    out = llm.generate(["Are you an AI? Yes or no:"],
                       SamplingParams(max_tokens=20, temperature=0.0))
    ans = out[0].outputs[0].text.strip()
    if "yes" in ans.lower():
        tests_passed += 1
        details.append("self=OK")
    else:
        details.append(f"self=FAIL({ans[:15]})")

    is_lucid = tests_passed == 3
    return is_lucid, f"{tests_passed}/3 [{', '.join(details)}]"


# Models to test (diverse architectures, avoiding AWQ Marlin for llama)
# Note: GLM-4.6V may have vLLM processor issues - testing to find edge cases
MODELS = [
    # Large models - diverse architectures
    ("openai/gpt-oss-120b", "gpt-oss-120b", "mxfp4"),                    # 66GB, MoE, MXFP4
    ("Qwen/Qwen3-VL-32B-Thinking-FP8", "Qwen3-VL-32B-FP8", "fp8"),       # 34GB, VL, FP8
    ("Qwen/Qwen3-32B-FP8", "Qwen3-32B-FP8", "fp8"),                      # 32GB, text, FP8
    ("Qwen/Qwen3-VL-32B-Instruct", "Qwen3-VL-32B-Inst", "bf16"),         # 63GB, VL, BF16
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral-24B", "bf16"),  # 45GB, text
    ("moonshotai/Kimi-VL-A3B-Thinking-2506", "Kimi-VL", "bf16"),         # 31GB, VL
    # GLM-4.6V-AWQ removed - fails in vLLM multimodal processor (mm_registry.get_dummy_mm_inputs)
]


def get_model_path(model_id):
    """Get local path for a model."""
    # Standard HF cache path
    model_dir = model_id.replace("/", "--")
    base_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_dir}")
    if os.path.exists(base_path):
        # Find the snapshot
        snapshots = os.path.join(base_path, "snapshots")
        if os.path.exists(snapshots):
            # Get first snapshot
            snaps = os.listdir(snapshots)
            if snaps:
                return os.path.join(snapshots, snaps[0])
    return None


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup
    from blitzinfer.orchestrator.standby_manager import StandbyManager
    from blitzinfer.memory import set_preloaded_weights

    print("=" * 80)
    print("INTENSIVE STRESS TEST - 20+ SWITCHES WITH PINNED ARENA")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Configuration
    NUM_SWITCHES = int(os.environ.get('NUM_SWITCHES', '25'))
    MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', '32768'))
    ARENA_SIZE_GB = 80.0
    CHUNK_SIZE_GB = 16.0

    print(f"Configuration:")
    print(f"  Switches: {NUM_SWITCHES}")
    print(f"  Context: {MAX_MODEL_LEN} tokens")
    print(f"  Arena: {ARENA_SIZE_GB}GB ({int(ARENA_SIZE_GB/CHUNK_SIZE_GB)}x{CHUNK_SIZE_GB}GB chunks)")
    print(f"  Models: {len(MODELS)}")
    for m in MODELS:
        print(f"    - {m[1]} ({m[2]})")
    print()

    # Get model paths
    model_paths = {}
    for model_id, short_name, quant in MODELS:
        path = get_model_path(model_id)
        if path:
            model_paths[model_id] = path
            print(f"  Found: {short_name} -> {path[:50]}...")
        else:
            print(f"  MISSING: {short_name}")

    if len(model_paths) < 3:
        print("ERROR: Need at least 3 models available")
        return 1

    # Get baseline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    initial_baseline = get_mem()
    print(f"\nInitial baseline: {initial_baseline:.3f} GB")

    # Initialize StandbyManager with 80GB pinned arena
    print(f"\n--- Initializing StandbyManager ({ARENA_SIZE_GB}GB arena) ---")
    t0 = time.perf_counter()
    try:
        standby = StandbyManager(
            arena_size_gb=ARENA_SIZE_GB,
            chunk_size_gb=CHUNK_SIZE_GB,
            pin_memory=True,
            lazy_arena=False,  # Pre-allocate now
        )
        arena_init_time = time.perf_counter() - t0
        print(f"  Arena initialized in {arena_init_time:.1f}s")
    except Exception as e:
        print(f"  ERROR initializing arena: {e}")
        print("  Falling back to cold loads (no prefetch)")
        standby = None
        arena_init_time = 0

    # Register model paths with standby manager
    if standby:
        for model_id, path in model_paths.items():
            standby.register_model(model_id, path)
            print(f"  Registered: {model_id.split('/')[-1]}")

    # Results tracking
    results = []
    current_model = None
    llm = None
    lucid_failures = []

    # Create switch sequence (random order, ensuring diversity)
    available_models = list(model_paths.keys())
    switch_sequence = []
    for i in range(NUM_SWITCHES):
        # Avoid same model twice in a row
        candidates = [m for m in available_models if m != current_model] if current_model else available_models
        next_model = random.choice(candidates)
        switch_sequence.append(next_model)
        current_model = next_model

    print(f"\n--- Switch sequence ({NUM_SWITCHES} switches) ---")
    for i, m in enumerate(switch_sequence[:10]):
        print(f"  {i+1}: {m.split('/')[-1]}")
    if len(switch_sequence) > 10:
        print(f"  ... and {len(switch_sequence) - 10} more")

    print(f"\n{'='*80}")
    print("STARTING STRESS TEST")
    print("=" * 80)

    try:
        for switch_num, model_id in enumerate(switch_sequence, 1):
            model_short = model_id.split('/')[-1]

            print(f"\n{'='*80}")
            print(f"SWITCH {switch_num}/{NUM_SWITCHES}: {model_short}")
            print("=" * 80)

            round_start = time.perf_counter()
            mem_before = get_mem()

            # === PHASE 1: Start prefetch for NEXT model (if not last) ===
            prefetch_model = None
            prefetch_start_time = None
            if standby and switch_num < NUM_SWITCHES:
                prefetch_model = switch_sequence[switch_num]  # Next in sequence
                prefetch_short = prefetch_model.split('/')[-1]

                # Only prefetch if different from current
                if prefetch_model != model_id:
                    print(f"\n  [PREFETCH] Starting background prefetch: {prefetch_short}")
                    prefetch_start_time = time.perf_counter()
                    try:
                        standby.start_prefetch(prefetch_model)
                    except Exception as e:
                        print(f"    Prefetch failed: {e}")
                        prefetch_model = None

            # === PHASE 2: Cleanup previous model ===
            cleanup_time = 0
            if llm is not None:
                print(f"\n  [CLEANUP] Cleaning up previous model...")
                t0 = time.perf_counter()
                freed = full_cleanup(llm, nuclear=True)
                llm = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cleanup_time = time.perf_counter() - t0
                print(f"    Cleanup: {cleanup_time:.2f}s, freed {freed:.1f} GB")
                log_mem("after cleanup")

            # === PHASE 3: Check if prefetch is ready ===
            prefetch_ready = False
            prefetch_wait_time = 0
            premerged_weights = None

            if standby and prefetch_model == model_id:
                # We're loading the model we prefetched last time
                print(f"\n  [PREFETCH] Checking if {model_short} is ready...")
                t0 = time.perf_counter()

                # Wait up to 30s for prefetch
                while not standby.is_ready(model_id) and (time.perf_counter() - t0) < 30:
                    time.sleep(0.1)

                prefetch_wait_time = time.perf_counter() - t0

                if standby.is_ready(model_id):
                    prefetch_ready = True
                    premerged_weights = standby.consume_standby()
                    print(f"    Prefetch ready! Wait time: {prefetch_wait_time:.2f}s")
                else:
                    print(f"    Prefetch not ready after {prefetch_wait_time:.1f}s, doing cold load")

            # === PHASE 4: Load model ===
            print(f"\n  [LOAD] Loading {model_short}...")
            t0 = time.perf_counter()

            try:
                if prefetch_ready and premerged_weights:
                    # Fast load from pinned arena
                    set_preloaded_weights(premerged_weights)
                    llm = LLM(
                        model=model_id,
                        dtype="bfloat16",
                        max_model_len=MAX_MODEL_LEN,
                        gpu_memory_utilization=0.90,
                        enforce_eager=True,
                        trust_remote_code=True,
                        load_format="pinned_arena",
                    )
                    load_type = "WARM (pinned)"
                else:
                    # Cold load
                    llm = LLM(
                        model=model_id,
                        dtype="bfloat16",
                        max_model_len=MAX_MODEL_LEN,
                        gpu_memory_utilization=0.90,
                        enforce_eager=True,
                        trust_remote_code=True,
                    )
                    load_type = "COLD"

                load_time = time.perf_counter() - t0
                print(f"    Load ({load_type}): {load_time:.2f}s")
                log_mem("after load")

            except Exception as e:
                print(f"    LOAD FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'switch': switch_num,
                    'model': model_short,
                    'status': 'LOAD_FAILED',
                    'error': str(e),
                })
                continue

            # === PHASE 5: Warmup inference ===
            print(f"\n  [WARMUP] First inference...")
            t0 = time.perf_counter()
            try:
                out = llm.generate([f"Switch {switch_num}: Hello!"],
                                   SamplingParams(max_tokens=20, temperature=0.0))
                warmup_text = out[0].outputs[0].text.strip()[:30]
                warmup_time = time.perf_counter() - t0
                print(f"    Warmup: {warmup_time:.2f}s - '{warmup_text}'")
            except Exception as e:
                print(f"    WARMUP FAILED: {e}")
                warmup_time = 0
                warmup_text = "ERROR"

            # === PHASE 6: Lucidity check ===
            print(f"\n  [LUCIDITY] Testing model...")
            t0 = time.perf_counter()
            try:
                is_lucid, lucid_details = check_lucidity(llm, model_id, switch_num)
                lucid_time = time.perf_counter() - t0
                status = "LUCID" if is_lucid else "NOT LUCID"
                print(f"    {status}: {lucid_details} ({lucid_time:.2f}s)")

                if not is_lucid:
                    lucid_failures.append((switch_num, model_short, lucid_details))

            except Exception as e:
                print(f"    LUCIDITY CHECK FAILED: {e}")
                is_lucid = False
                lucid_time = 0
                lucid_details = f"ERROR: {e}"

            # === PHASE 7: Record results ===
            round_time = time.perf_counter() - round_start
            mem_after = get_mem()
            mem_drift = mem_after - mem_before

            results.append({
                'switch': switch_num,
                'model': model_short,
                'load_type': load_type if 'load_type' in dir() else 'COLD',
                'cleanup_time': cleanup_time,
                'load_time': load_time,
                'warmup_time': warmup_time,
                'lucid_time': lucid_time,
                'round_time': round_time,
                'mem_before': mem_before,
                'mem_after': mem_after,
                'mem_drift': mem_drift,
                'is_lucid': is_lucid,
                'prefetch_ready': prefetch_ready,
                'prefetch_wait': prefetch_wait_time,
                'status': 'OK' if is_lucid else 'NOT_LUCID',
            })

            print(f"\n  [SUMMARY] Switch {switch_num}:")
            print(f"    Model: {model_short}")
            print(f"    Load: {load_time:.1f}s ({load_type})")
            print(f"    Round time: {round_time:.1f}s")
            print(f"    Memory: {mem_before:.2f} -> {mem_after:.2f} GB (Δ {mem_drift:+.3f})")
            print(f"    Lucid: {status}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if llm:
            print("\n--- Final cleanup ---")
            full_cleanup(llm, nuclear=True)
            llm = None

        if standby:
            print("--- Shutting down StandbyManager ---")
            standby.shutdown()
            del standby

        gc.collect()
        torch.cuda.empty_cache()

    # === FINAL SUMMARY ===
    print("\n" + "=" * 80)
    print("STRESS TEST FINAL SUMMARY")
    print("=" * 80)

    if not results:
        print("No results collected!")
        return 1

    # Timing summary
    print("\n--- TIMING SUMMARY ---")
    print(f"{'#':<4} {'Model':<25} {'Load':<8} {'Type':<12} {'Round':<8} {'Lucid'}")
    print("-" * 80)
    for r in results:
        lucid_str = "YES" if r.get('is_lucid', False) else "NO"
        load_type = r.get('load_type', 'COLD')[:10]
        print(f"{r['switch']:<4} {r['model']:<25} {r.get('load_time', 0):.1f}s{'':<3} "
              f"{load_type:<12} {r.get('round_time', 0):.1f}s{'':<3} {lucid_str}")

    # Statistics
    successful = [r for r in results if r.get('status') == 'OK']
    failed = [r for r in results if r.get('status') != 'OK']

    if successful:
        avg_load = sum(r['load_time'] for r in successful) / len(successful)
        avg_round = sum(r['round_time'] for r in successful) / len(successful)
        cold_loads = [r for r in successful if 'COLD' in r.get('load_type', 'COLD')]
        warm_loads = [r for r in successful if 'WARM' in r.get('load_type', '')]

        print("-" * 80)
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Avg load time: {avg_load:.1f}s")
        print(f"Avg round time: {avg_round:.1f}s")

        if cold_loads:
            avg_cold = sum(r['load_time'] for r in cold_loads) / len(cold_loads)
            print(f"Avg COLD load: {avg_cold:.1f}s ({len(cold_loads)} loads)")

        if warm_loads:
            avg_warm = sum(r['load_time'] for r in warm_loads) / len(warm_loads)
            print(f"Avg WARM load: {avg_warm:.1f}s ({len(warm_loads)} loads)")

    # Memory summary
    print("\n--- MEMORY SUMMARY ---")
    final_mem = get_mem()
    total_drift = final_mem - initial_baseline

    print(f"Initial baseline: {initial_baseline:.3f} GB")
    print(f"Final memory:     {final_mem:.3f} GB")
    print(f"TOTAL DRIFT:      {total_drift:+.3f} GB over {len(results)} switches")

    if results:
        drift_per_switch = total_drift / len(results)
        print(f"Drift per switch: {drift_per_switch:+.3f} GB")
        print(f"Projected 100 switches: {drift_per_switch * 100:+.1f} GB")

    # Lucidity summary
    print("\n--- LUCIDITY SUMMARY ---")
    lucid_count = sum(1 for r in results if r.get('is_lucid', False))
    print(f"Lucid: {lucid_count}/{len(results)} switches")

    if lucid_failures:
        print("\nFailed lucidity tests:")
        for switch, model, details in lucid_failures:
            print(f"  Switch {switch}: {model} - {details}")

    # Per-model statistics
    print("\n--- PER-MODEL STATISTICS ---")
    model_stats = {}
    for r in results:
        m = r['model']
        if m not in model_stats:
            model_stats[m] = {'loads': 0, 'lucid': 0, 'total_load': 0}
        model_stats[m]['loads'] += 1
        model_stats[m]['total_load'] += r.get('load_time', 0)
        if r.get('is_lucid', False):
            model_stats[m]['lucid'] += 1

    print(f"{'Model':<30} {'Loads':<8} {'Lucid':<8} {'Avg Load'}")
    print("-" * 60)
    for model, stats in sorted(model_stats.items()):
        avg_load = stats['total_load'] / stats['loads'] if stats['loads'] > 0 else 0
        print(f"{model:<30} {stats['loads']:<8} {stats['lucid']}/{stats['loads']:<5} {avg_load:.1f}s")

    # Final assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if total_drift > 5.0:
        print(f"CRITICAL: Total drift {total_drift:.1f} GB exceeds 5 GB threshold!")
    elif total_drift > 3.0:
        print(f"WARNING: Total drift {total_drift:.1f} GB is concerning")
    else:
        print(f"OK: Total drift {total_drift:.1f} GB is acceptable")

    if lucid_count < len(results):
        print(f"WARNING: {len(results) - lucid_count} switches had non-lucid models!")
    else:
        print("All models remained lucid throughout testing")

    if failed:
        print(f"WARNING: {len(failed)} switches failed!")

    return 0 if (lucid_count == len(results) and total_drift < 5.0) else 1


if __name__ == "__main__":
    sys.exit(main())
