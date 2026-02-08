#!/usr/bin/env python3
"""Test 70GB chunked arena for full standby support of both models.

Uses chunked allocation (10GB x 7) to potentially avoid kernel issues
with large pinned memory allocations.
"""

import os
import sys
import time
import gc
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

LOG_FILE = f"/home/tommy/pythonprojects/blitzinfer/standby_70gb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()


def get_memory_status():
    result = {}
    try:
        result['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        result['gpu_mb'] = 0

    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = int(parts[1].strip().split()[0]) / 1024
                    if key in ['MemTotal', 'MemFree', 'MemAvailable', 'Cached']:
                        result[key.lower() + '_mb'] = val
    except Exception:
        pass
    return result


def log_memory(label):
    mem = get_memory_status()
    log(f"  MEM [{label}]: GPU={mem.get('gpu_mb', 0):.0f}MB, "
        f"RAM avail={mem.get('memavailable_mb', 0)/1024:.1f}GB, "
        f"cached={mem.get('cached_mb', 0)/1024:.1f}GB")
    return mem


def unload_llm(llm):
    """Cleanup and unload LLM."""
    try:
        engine_core = llm.llm_engine.engine_core
        core = engine_core.engine_core if hasattr(engine_core, 'engine_core') else engine_core

        model_runner = None
        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                else:
                    model_runner = getattr(worker, 'model_runner', None)

        if model_runner and hasattr(model_runner, 'model') and model_runner.model:
            for param in model_runner.model.parameters():
                param.data = torch.empty(0, device='cpu')
            for buf in model_runner.model.buffers():
                buf.data = torch.empty(0, device='cpu')

            if hasattr(model_runner, 'kv_caches') and model_runner.kv_caches:
                model_runner.kv_caches.clear()

            if hasattr(model_runner, 'compilation_config'):
                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                if sfc:
                    for layer in sfc.values():
                        if hasattr(layer, 'kv_cache'):
                            layer.kv_cache = []

            model_runner.model = None

        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        log(f"    Cleanup warning: {e}")

    del llm
    gc.collect()

    # vLLM state cleanup
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'cleanup_dist_env_and_memory'):
            parallel_state.cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception:
        pass

    try:
        import torch._dynamo as dynamo
        dynamo.reset()
    except Exception:
        pass

    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()
    except Exception:
        pass

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()


def verify(llm):
    from vllm import SamplingParams
    outputs = llm.generate(
        ["What is the capital of France? Answer with just the city name:"],
        SamplingParams(max_tokens=20, temperature=0)
    )
    response = outputs[0].outputs[0].text.strip().lower()
    passed = 'paris' in response
    log(f"    Verify: {'PASS' if passed else 'FAIL'} ({response[:30]})")
    return passed


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager
    from blitzinfer.memory import set_preloaded_weights

    MODEL_QWEN = "Qwen/Qwen3-32B-FP8"   # ~33GB
    MODEL_GPT = "openai/gpt-oss-120b"    # ~65GB

    # 70GB arena with 10GB chunks - should fit both models
    STANDBY_ARENA_GB = 70.0
    CHUNK_SIZE_GB = 10.0  # 7 chunks
    NUM_SWITCHES = 6

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 70)
    log("70GB CHUNKED ARENA TEST")
    log("=" * 70)
    log(f"Log file: {LOG_FILE}")
    log(f"Arena: {STANDBY_ARENA_GB}GB ({int(STANDBY_ARENA_GB/CHUNK_SIZE_GB)} x {CHUNK_SIZE_GB}GB chunks)")
    log(f"Models: {MODEL_QWEN} (~33GB), {MODEL_GPT} (~65GB)")
    log(f"Switches: {NUM_SWITCHES}")
    log("")

    log_memory("START")

    results = []
    standby = None
    llm = None

    try:
        # Initialize standby manager - this allocates the 70GB arena
        log("Allocating 70GB chunked arena...")
        standby = StandbyManager(
            arena_size_gb=STANDBY_ARENA_GB,
            chunk_size_gb=CHUNK_SIZE_GB
        )
        log_memory("AFTER ARENA ALLOC")

        # Initial cold load
        log("\n[INIT] Cold loading Qwen...")
        t0 = time.perf_counter()
        llm = LLM(model=MODEL_QWEN, **VLLM_KWARGS)
        log(f"  Cold load: {time.perf_counter() - t0:.1f}s")
        log_memory("AFTER INIT")

        if not verify(llm):
            log("FATAL: Initial verification failed!")
            sys.exit(1)

        # Switching loop
        for switch_num in range(1, NUM_SWITCHES + 1):
            next_model = MODEL_GPT if (switch_num % 2 == 1) else MODEL_QWEN

            log("")
            log("=" * 70)
            log(f"[SWITCH {switch_num}/{NUM_SWITCHES}] -> {next_model}")
            log("=" * 70)

            mem_before = log_memory("BEFORE")

            # Start prefetch (both models now fit in arena)
            log("\n  Starting prefetch...")
            standby.start_prefetch(next_model)

            # Run inference while prefetching
            log("  Running inference while prefetching...")
            outputs = llm.generate(
                ["Explain machine learning briefly:"],
                SamplingParams(max_tokens=100, temperature=0.7)
            )
            log(f"    Output: {outputs[0].outputs[0].text.strip()[:60]}...")

            # Wait for prefetch
            log("  Waiting for prefetch...")
            standby.wait_for_load(timeout=180)
            standby_ready = standby.is_ready(next_model)
            log(f"    Prefetch: {'READY' if standby_ready else 'FAILED'}")

            log_memory("BEFORE UNLOAD")

            # Switch
            switch_start = time.perf_counter()
            premerged = standby.consume_standby() if standby_ready else None

            log("  Unloading current model...")
            unload_llm(llm)
            llm = None
            log_memory("AFTER UNLOAD")

            log("  Loading new model...")
            load_start = time.perf_counter()
            if premerged:
                log(f"    From standby ({len(premerged)} tensors)...")
                set_preloaded_weights(premerged)
                llm = LLM(model=next_model, load_format="pinned_arena", **VLLM_KWARGS)
            else:
                log("    Cold load from SSD...")
                llm = LLM(model=next_model, **VLLM_KWARGS)

            load_time = time.perf_counter() - load_start
            total_time = time.perf_counter() - switch_start
            log(f"    Load: {load_time:.1f}s, Total: {total_time:.1f}s")

            if standby_ready and premerged:
                standby.release_consumed(next_model)

            log_memory("AFTER LOAD")

            passed = verify(llm)
            results.append({
                'switch': switch_num,
                'model': next_model,
                'standby': standby_ready,
                'time': total_time,
                'passed': passed
            })

        # Summary
        log("\n" + "=" * 70)
        log("RESULTS")
        log("=" * 70)

        cold = [r for r in results if not r['standby']]
        warm = [r for r in results if r['standby']]

        if cold:
            log(f"Cold: {len(cold)}, avg {sum(r['time'] for r in cold)/len(cold):.1f}s")
        if warm:
            log(f"Warm: {len(warm)}, avg {sum(r['time'] for r in warm)/len(warm):.1f}s")
            if cold:
                log(f"Speedup: {(sum(r['time'] for r in cold)/len(cold))/(sum(r['time'] for r in warm)/len(warm)):.1f}x")

        passed = sum(1 for r in results if r['passed'])
        log(f"Verification: {passed}/{len(results)} passed")

        for r in results:
            mode = "WARM" if r['standby'] else "COLD"
            status = "OK" if r['passed'] else "FAIL"
            log(f"  {r['switch']}. {r['model'][:25]:25s} [{mode:4s}] {r['time']:5.1f}s [{status}]")

        log("=" * 70)
        log("TEST COMPLETE")

    except Exception as e:
        log(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        log_memory("AT CRASH")
        sys.exit(1)

    finally:
        log("\nCleaning up...")
        if standby:
            standby.shutdown()
        if llm:
            try:
                unload_llm(llm)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
