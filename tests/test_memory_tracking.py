#!/usr/bin/env python3
"""Track memory at every step to find the leak source.

Run all models that were tested before the crash and track memory meticulously.
"""

import os
import sys
import gc
import time
import ctypes

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.orchestrator.standby_manager import StandbyManager


def get_memory_detailed():
    """Get very detailed memory state."""
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_allocated = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()

    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024

    return {
        'gpu_used_gb': (gpu_total - gpu_free) / 1024**3,
        'gpu_free_gb': gpu_free / 1024**3,
        'gpu_alloc_gb': gpu_allocated / 1024**3,
        'gpu_rsv_gb': gpu_reserved / 1024**3,
        'ram_total_gb': meminfo.get('MemTotal', 0) / 1024**3,
        'ram_free_gb': meminfo.get('MemFree', 0) / 1024**3,
        'ram_avail_gb': meminfo.get('MemAvailable', 0) / 1024**3,
        'ram_buffers_gb': meminfo.get('Buffers', 0) / 1024**3,
        'ram_cached_gb': meminfo.get('Cached', 0) / 1024**3,
        'ram_shared_gb': meminfo.get('Shmem', 0) / 1024**3,
        'ram_slab_gb': meminfo.get('Slab', 0) / 1024**3,
    }


def log_mem(label, prev_mem=None):
    """Log memory with delta from previous."""
    mem = get_memory_detailed()

    delta_gpu = ""
    delta_ram = ""
    if prev_mem:
        d_gpu = mem['gpu_used_gb'] - prev_mem['gpu_used_gb']
        d_ram = prev_mem['ram_avail_gb'] - mem['ram_avail_gb']  # Available decreases = usage increases
        delta_gpu = f" (Δ{d_gpu:+.2f}GB)"
        delta_ram = f" (Δ{d_ram:+.2f}GB)"

    print(f"[{label}]")
    print(f"  GPU: {mem['gpu_used_gb']:.2f}GB used{delta_gpu}, alloc={mem['gpu_alloc_gb']:.2f}, rsv={mem['gpu_rsv_gb']:.2f}")
    print(f"  RAM: {mem['ram_avail_gb']:.2f}GB avail{delta_ram}, shared={mem['ram_shared_gb']:.2f}, cached={mem['ram_cached_gb']:.2f}")
    sys.stdout.flush()
    return mem


def malloc_trim():
    """Release memory to OS."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# Models in order they were tested before crash
TEST_MODELS = [
    ("gpt-oss-120b", "openai/gpt-oss-120b"),
    ("qwen3-32b", "Qwen/Qwen3-32B-FP8"),
    ("mistral-small-24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("llama-3.1-70b", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"),  # CRASH POINT
    # ("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct"),  # Skip - 144GB won't fit
    # ("qwen3-vl-32b-thinking", "Qwen/Qwen3-VL-32B-Thinking-FP8"),
    # ("kimi-vl", "moonshotai/Kimi-VL-A3B-Thinking-2506"),
]


def main():
    from vllm import LLM, SamplingParams

    print("=" * 80)
    print("DETAILED MEMORY TRACKING TEST")
    print("=" * 80)

    prev_mem = log_mem("INITIAL (no arena, no model)")

    # Create arena
    print("\n" + "=" * 60)
    print("STEP 1: Create 80GB pinned arena")
    print("=" * 60)

    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    prev_mem = log_mem("After arena creation", prev_mem)

    llm = None
    current_model = None
    sampling_params = SamplingParams(max_tokens=10)

    memory_history = []

    for model_idx, (model_name, model_path) in enumerate(TEST_MODELS):
        print(f"\n{'=' * 60}")
        print(f"MODEL {model_idx + 1}/{len(TEST_MODELS)}: {model_name}")
        print(f"{'=' * 60}")

        # === CLEANUP PHASE ===
        if llm is not None:
            print(f"\n--- Cleanup {current_model} ---")
            prev_mem = log_mem("Before cleanup", prev_mem)

            freed = full_cleanup(llm)
            llm = None
            prev_mem = log_mem("After full_cleanup", prev_mem)

            gc.collect()
            prev_mem = log_mem("After gc.collect", prev_mem)

            torch.cuda.empty_cache()
            prev_mem = log_mem("After empty_cache", prev_mem)

            malloc_trim()
            prev_mem = log_mem("After malloc_trim", prev_mem)

            print(f"Freed {freed:.1f}GB GPU")

        # === LOAD PHASE ===
        print(f"\n--- Load {model_name} ---")
        prev_mem = log_mem("Before load", prev_mem)

        # Check standby
        use_standby = standby.is_ready(model_path)
        if use_standby:
            print("Using standby!")
            premerged = standby.consume_standby()
            from blitzinfer.memory import set_preloaded_weights
            set_preloaded_weights(premerged)
            load_format = "pinned_arena"
        else:
            load_format = "auto"

        print(f"Loading with format={load_format}...")
        load_start = time.time()

        try:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=0.94,
                max_model_len=131072,
                trust_remote_code=True,
                enforce_eager=True,
                load_format=load_format,
            )
            current_model = model_name
            load_time = time.time() - load_start
            print(f"Loaded in {load_time:.1f}s")
            prev_mem = log_mem("After model load", prev_mem)

        except Exception as e:
            print(f"LOAD FAILED: {e}")
            prev_mem = log_mem("After load failure", prev_mem)
            continue

        # === INFERENCE PHASE ===
        print(f"\n--- Inference on {model_name} ---")
        try:
            outputs = llm.generate(["Hello"], sampling_params)
            print(f"Output: {outputs[0].outputs[0].text[:50]}")
            prev_mem = log_mem("After inference", prev_mem)
        except Exception as e:
            print(f"INFERENCE FAILED: {e}")
            prev_mem = log_mem("After inference failure", prev_mem)

        # === PREFETCH PHASE ===
        if model_idx < len(TEST_MODELS) - 1:
            next_model_name, next_model_path = TEST_MODELS[model_idx + 1]
            print(f"\n--- Start prefetch {next_model_name} ---")
            standby.start_prefetch(next_model_path)
            time.sleep(3)  # Let prefetch start
            prev_mem = log_mem("After prefetch started", prev_mem)

        # Record for summary
        memory_history.append({
            'model': model_name,
            'ram_avail_gb': prev_mem['ram_avail_gb'],
            'ram_shared_gb': prev_mem['ram_shared_gb'],
            'gpu_used_gb': prev_mem['gpu_used_gb'],
        })

        print(f"\n[CHECKPOINT] After {model_name}: RAM avail={prev_mem['ram_avail_gb']:.1f}GB, "
              f"shared={prev_mem['ram_shared_gb']:.1f}GB, GPU={prev_mem['gpu_used_gb']:.1f}GB")

    # Final summary
    print("\n" + "=" * 80)
    print("MEMORY HISTORY")
    print("=" * 80)
    print(f"{'Model':<25} {'RAM Avail':<12} {'RAM Shared':<12} {'GPU Used':<12}")
    print("-" * 60)
    for entry in memory_history:
        print(f"{entry['model']:<25} {entry['ram_avail_gb']:<12.1f} {entry['ram_shared_gb']:<12.1f} {entry['gpu_used_gb']:<12.1f}")

    if len(memory_history) >= 2:
        ram_drift = memory_history[0]['ram_avail_gb'] - memory_history[-1]['ram_avail_gb']
        print(f"\nRAM drift from first to last model: {ram_drift:.1f}GB")

    # Cleanup
    print("\n" + "=" * 60)
    print("FINAL CLEANUP")
    print("=" * 60)

    if llm:
        full_cleanup(llm)
        llm = None
    standby.shutdown()
    gc.collect()
    malloc_trim()

    log_mem("FINAL STATE", prev_mem)


if __name__ == "__main__":
    main()
