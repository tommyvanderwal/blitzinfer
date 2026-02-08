#!/usr/bin/env python3
"""Safe stress test for standby switching with memory monitoring.

Key changes from test_standby_stress.py:
1. Smaller arena (40GB) - only prefetch Qwen (~33GB), load gpt-oss cold
2. Memory monitoring at each step
3. Logs saved to file for crash diagnosis
4. Clear page cache option
"""

import os
import sys
import time
import gc
import logging
import threading
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure environment before importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Create log file with timestamp
LOG_FILE = f"/home/tommy/pythonprojects/blitzinfer/standby_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    """Print with immediate flush and save to file."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()


def get_memory_status() -> dict:
    """Get comprehensive memory status."""
    result = {}

    # GPU memory
    try:
        result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    except Exception:
        result['gpu_allocated_mb'] = 0
        result['gpu_reserved_mb'] = 0

    # System memory from /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    # Extract numeric value (in kB)
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val) / 1024  # Convert to MB

            result['ram_total_mb'] = meminfo.get('MemTotal', 0)
            result['ram_free_mb'] = meminfo.get('MemFree', 0)
            result['ram_available_mb'] = meminfo.get('MemAvailable', 0)
            result['buffers_mb'] = meminfo.get('Buffers', 0)
            result['cached_mb'] = meminfo.get('Cached', 0)
    except Exception as e:
        log(f"Warning: Could not read meminfo: {e}")

    return result


def log_memory(label: str):
    """Log current memory status."""
    mem = get_memory_status()
    log(f"  MEMORY [{label}]:")
    log(f"    GPU: {mem.get('gpu_allocated_mb', 0):.0f}MB allocated, {mem.get('gpu_reserved_mb', 0):.0f}MB reserved")
    log(f"    RAM: {mem.get('ram_available_mb', 0):.0f}MB available, "
        f"{mem.get('cached_mb', 0):.0f}MB cached, "
        f"{mem.get('ram_total_mb', 0):.0f}MB total")
    return mem


def clear_page_cache():
    """Try to drop page cache to free RAM."""
    try:
        # This needs root, will silently fail if not available
        subprocess.run(['sudo', '-n', 'sh', '-c', 'sync; echo 3 > /proc/sys/vm/drop_caches'],
                      capture_output=True, timeout=5)
        log("  Dropped page caches")
        return True
    except Exception:
        log("  Note: Could not drop page cache (needs sudo)")
        return False


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        return 0.0


def unload_llm_properly(llm):
    """Properly unload LLM and free GPU memory."""
    log("    Clearing model weights from GPU...")
    try:
        # Navigate to model runner - handle V1 engine structure
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            core = engine_core.engine_core
        else:
            core = engine_core

        model_runner = None
        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker is not None:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                else:
                    model_runner = getattr(worker, 'model_runner', None)

        if model_runner is not None:
            if hasattr(model_runner, 'model') and model_runner.model is not None:
                model = model_runner.model
                param_count = 0
                for param in model.parameters():
                    param.data = torch.empty(0, device='cpu')
                    param_count += 1
                buf_count = 0
                for buf in model.buffers():
                    buf.data = torch.empty(0, device='cpu')
                    buf_count += 1
                log(f"    Cleared {param_count} params, {buf_count} buffers")

            if hasattr(model_runner, 'kv_caches') and model_runner.kv_caches:
                cache_count = len(model_runner.kv_caches)
                for i in range(len(model_runner.kv_caches)):
                    model_runner.kv_caches[i] = None
                model_runner.kv_caches.clear()
                log(f"    Cleared {cache_count} KV caches")

            if hasattr(model_runner, 'compilation_config'):
                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                if sfc:
                    for layer in sfc.values():
                        if hasattr(layer, 'kv_cache'):
                            layer.kv_cache = []

            model_runner.model = None
    except Exception as e:
        log(f"    Warning: cleanup error: {e}")

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass

    del llm
    gc.collect()

    # Reset vLLM state
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

    # CRITICAL: Clear rotary embedding cache
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rope_dict = rotary_embedding._ROPE_DICT
            log(f"    Clearing _ROPE_DICT ({len(rope_dict)} entries)")
            rope_dict.clear()
    except Exception:
        pass

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()


def verify_inference(llm, model_name) -> bool:
    """Verify model produces correct output."""
    from vllm import SamplingParams

    prompt = "What is the capital city of France? Answer with just the city name:"
    outputs = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
    response = outputs[0].outputs[0].text.strip().lower()

    passed = 'paris' in response
    status = "PASS" if passed else "FAIL"
    log(f"    Verify [{status}]: {response[:50]}")

    if not passed:
        log(f"    WARNING: Unexpected response from {model_name}!")

    return passed


@dataclass
class SwitchMetrics:
    switch_num: int
    from_model: str
    to_model: str
    was_standby_ready: bool
    prefetch_time: float
    unload_time: float
    load_time: float
    total_switch_time: float
    verify_passed: bool
    ram_before_mb: float
    ram_after_mb: float


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager
    from blitzinfer.memory import set_preloaded_weights

    # Configuration - smaller arena, only prefetch the smaller model
    MODEL_QWEN = "Qwen/Qwen3-32B-FP8"   # ~33GB - fits in 40GB arena
    MODEL_GPT = "openai/gpt-oss-120b"    # ~65GB - too big for arena, cold load only

    # 40GB arena fits Qwen but not gpt-oss-120b
    # This keeps total memory reasonable: 40GB arena + 80GB GPU = 120GB < 124GB RAM
    STANDBY_ARENA_GB = 40.0
    NUM_SWITCHES = 6

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 70)
    log("SAFE STANDBY SWITCHING TEST")
    log("=" * 70)
    log(f"Log file: {LOG_FILE}")
    log(f"Model A (prefetchable): {MODEL_QWEN} (~33GB)")
    log(f"Model B (cold only): {MODEL_GPT} (~65GB)")
    log(f"Arena size: {STANDBY_ARENA_GB}GB")
    log(f"Switches: {NUM_SWITCHES}")
    log("")

    log_memory("START")

    results = []
    standby = StandbyManager(arena_size_gb=STANDBY_ARENA_GB)
    llm = None
    current_model = None

    try:
        # Clear page cache to start fresh
        clear_page_cache()
        log_memory("AFTER CACHE CLEAR")

        # Initial cold load
        log("-" * 50)
        log("[INIT] Cold loading Qwen...")
        log("-" * 50)

        t0 = time.perf_counter()
        llm = LLM(model=MODEL_QWEN, **VLLM_KWARGS)
        cold_load_time = time.perf_counter() - t0
        current_model = MODEL_QWEN
        log(f"Cold load: {cold_load_time:.1f}s")

        log_memory("AFTER INIT LOAD")

        # Verify
        if not verify_inference(llm, MODEL_QWEN):
            log("FATAL: Initial model verification failed!")
            sys.exit(1)

        # Switching loop
        for switch_num in range(1, NUM_SWITCHES + 1):
            # Alternate between models
            next_model = MODEL_GPT if current_model == MODEL_QWEN else MODEL_QWEN
            can_prefetch = (next_model == MODEL_QWEN)  # Only Qwen fits in arena

            log("")
            log("=" * 70)
            log(f"[SWITCH {switch_num}/{NUM_SWITCHES}] {current_model} -> {next_model}")
            log("=" * 70)

            mem_before = log_memory("BEFORE SWITCH")

            # Check available RAM
            ram_avail = mem_before.get('ram_available_mb', 0)
            if ram_avail < 20000:  # Less than 20GB available
                log(f"WARNING: Low RAM ({ram_avail:.0f}MB available)")
                clear_page_cache()
                log_memory("AFTER EMERGENCY CACHE CLEAR")

            prefetch_time = 0.0
            standby_ready = False

            if can_prefetch:
                # Start prefetch for Qwen
                log(f"\n[1] Starting prefetch for {next_model}...")
                prefetch_start = time.perf_counter()
                standby.start_prefetch(next_model)

                # Do some inference while prefetch runs
                log("\n[2] Running inference while prefetching...")
                outputs = llm.generate(
                    ["Explain machine learning briefly:"],
                    SamplingParams(max_tokens=100, temperature=0.7)
                )
                log(f"    Output: {outputs[0].outputs[0].text.strip()[:80]}...")

                # Wait for prefetch
                log("\n[3] Checking prefetch status...")
                standby.wait_for_load(timeout=120)
                prefetch_time = time.perf_counter() - prefetch_start
                standby_ready = standby.is_ready(next_model)
                log(f"    Prefetch: {'READY' if standby_ready else 'FAILED'} ({prefetch_time:.1f}s)")
            else:
                log(f"\n[1-3] Model too large for arena, will cold load...")

            log_memory("BEFORE UNLOAD")

            # Switch
            log(f"\n[4] Switching to {next_model}...")
            switch_start = time.perf_counter()

            premerged = None
            if standby_ready:
                premerged = standby.consume_standby()
                if premerged is None:
                    log("    WARNING: Standby was evicted!")
                    standby_ready = False

            # Unload current
            unload_start = time.perf_counter()
            log("    Unloading current model...")
            unload_llm_properly(llm)
            llm = None
            unload_time = time.perf_counter() - unload_start
            log(f"    Unload: {unload_time:.1f}s")

            log_memory("AFTER UNLOAD")

            # Load new model
            load_start = time.perf_counter()
            if standby_ready and premerged is not None:
                log(f"    Loading from standby ({len(premerged)} tensors)...")
                set_preloaded_weights(premerged)
                llm = LLM(model=next_model, load_format="pinned_arena", **VLLM_KWARGS)
            else:
                log("    Cold loading from SSD...")
                llm = LLM(model=next_model, **VLLM_KWARGS)

            load_time = time.perf_counter() - load_start
            total_switch_time = time.perf_counter() - switch_start
            current_model = next_model
            log(f"    Load: {load_time:.1f}s")
            log(f"    Total switch: {total_switch_time:.1f}s")

            # Release arena memory
            if standby_ready and premerged is not None:
                standby.release_consumed(next_model)

            mem_after = log_memory("AFTER LOAD")

            # Verify
            verify_passed = verify_inference(llm, next_model)

            results.append(SwitchMetrics(
                switch_num=switch_num,
                from_model=MODEL_QWEN if next_model == MODEL_GPT else MODEL_GPT,
                to_model=next_model,
                was_standby_ready=standby_ready,
                prefetch_time=prefetch_time,
                unload_time=unload_time,
                load_time=load_time,
                total_switch_time=total_switch_time,
                verify_passed=verify_passed,
                ram_before_mb=mem_before.get('ram_available_mb', 0),
                ram_after_mb=mem_after.get('ram_available_mb', 0),
            ))

        # Summary
        log("\n" + "=" * 70)
        log("RESULTS")
        log("=" * 70)

        cold_switches = [r for r in results if not r.was_standby_ready]
        warm_switches = [r for r in results if r.was_standby_ready]

        if cold_switches:
            avg_cold = sum(r.total_switch_time for r in cold_switches) / len(cold_switches)
            log(f"Cold switches: {len(cold_switches)}, avg {avg_cold:.1f}s")

        if warm_switches:
            avg_warm = sum(r.total_switch_time for r in warm_switches) / len(warm_switches)
            log(f"Warm switches: {len(warm_switches)}, avg {avg_warm:.1f}s")
            if cold_switches:
                log(f"Speedup: {avg_cold/avg_warm:.1f}x")

        passed = sum(1 for r in results if r.verify_passed)
        log(f"Verification: {passed}/{len(results)} passed")

        log("\nPer-switch:")
        for r in results:
            mode = "WARM" if r.was_standby_ready else "COLD"
            status = "OK" if r.verify_passed else "FAIL"
            log(f"  {r.switch_num}. {r.to_model[:20]:20s} [{mode:4s}] {r.total_switch_time:5.1f}s [{status}] "
                f"RAM: {r.ram_after_mb/1024:.1f}GB avail")

        log("=" * 70)
        log("TEST COMPLETE")
        log(f"Log saved to: {LOG_FILE}")

    except Exception as e:
        log(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        log_memory("AT CRASH")
        sys.exit(1)

    finally:
        log("\nCleaning up...")
        standby.shutdown()
        if llm is not None:
            try:
                unload_llm_properly(llm)
            except Exception as e:
                log(f"Cleanup error: {e}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
