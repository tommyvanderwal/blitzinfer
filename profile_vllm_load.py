#!/usr/bin/env python3
"""
Profile where vLLM spends time during model loading.
Target: Find the bottleneck beyond raw memory transfer.
"""

import os
import sys
import gc
import time
import types
import cProfile
import pstats
from io import StringIO

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def timed(name):
    """Context manager for timing."""
    class Timer:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = (time.perf_counter() - self.start) * 1000
            print(f"  {self.name}: {self.elapsed:.1f}ms")
    return Timer(name)


def profile_vllm_load():
    print("=" * 70)
    print("PROFILING VLLM MODEL LOADING")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    initial = get_mem()
    print(f"Initial: {initial:.2f} GB")

    # Profile the entire load
    print("\n>>> Loading model with timing hooks...")

    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load_time = (time.perf_counter() - t0) * 1000

    profiler.disable()

    print(f"\nTotal load: {load_time:.0f}ms")
    print(f"GPU: {get_mem():.2f} GB")

    # Analyze profile
    print("\n>>> Top 20 time consumers:")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Quick test
    out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
    print(f"\nOutput: {out[0].outputs[0].text}")

    del llm, out
    gc.collect()
    torch.cuda.empty_cache()


def trace_load_phases():
    """Trace the major phases of model loading."""
    print("\n" + "=" * 70)
    print("TRACING LOAD PHASES")
    print("=" * 70)

    import vllm.model_executor.model_loader.loader as loader_module

    # Store original functions
    original_load = None
    load_times = {}

    # Monkey-patch key functions
    from vllm import LLM, SamplingParams

    # Simple timing approach - just measure the whole thing
    initial = get_mem()
    print(f"Initial: {initial:.2f} GB")

    # Phase 1: Import and config parsing
    with timed("LLM init (config parsing)"):
        from vllm.config import LLMConfig, ModelConfig
        from vllm.engine.llm_engine import LLMEngine

    # Phase 2: Full load with sections
    print("\n>>> Loading with section timing...")

    t_total = time.perf_counter()

    # The LLM constructor does:
    # 1. Parse config
    # 2. Initialize engine
    # 3. Load model weights
    # 4. Initialize KV cache
    # 5. Warmup (if enabled)

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    total_time = (time.perf_counter() - t_total) * 1000
    print(f"\nTotal: {total_time:.0f}ms")
    print(f"GPU: {get_mem():.2f} GB (used {initial - get_mem():.2f} GB)")

    out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
    print(f"Output: {out[0].outputs[0].text}")

    del llm, out
    gc.collect()
    torch.cuda.empty_cache()


def measure_weight_load_only():
    """Measure just the weight loading portion."""
    print("\n" + "=" * 70)
    print("MEASURING WEIGHT LOAD ONLY")
    print("=" * 70)

    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = snapshot_download(model_name)
    safetensor_files = list(Path(model_path).glob("*.safetensors"))

    print(f"Model: {model_name}")
    print(f"Files: {len(safetensor_files)}")

    initial = get_mem()
    print(f"Initial GPU: {initial:.2f} GB")

    # Load weights file by file
    total_bytes = 0
    load_times = []

    for sf in safetensor_files:
        gc.collect()
        torch.cuda.empty_cache()
        before = get_mem()

        t0 = time.perf_counter()
        weights = load_file(str(sf), device="cuda")
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        after = get_mem()
        file_bytes = sum(t.numel() * t.element_size() for t in weights.values())
        total_bytes += file_bytes

        print(f"  {sf.name}: {elapsed:.0f}ms, {file_bytes / (1024**3):.2f} GB")
        load_times.append(elapsed)

        del weights

    total_time = sum(load_times)
    total_gb = total_bytes / (1024**3)
    bw = total_gb / (total_time / 1000)

    print(f"\nTotal: {total_time:.0f}ms for {total_gb:.2f} GB ({bw:.2f} GB/s)")

    # Compare to theoretical minimum
    theoretical_min = total_gb * 1000 / 20  # 20 GB/s max
    print(f"Theoretical minimum (20 GB/s): {theoretical_min:.0f}ms")
    print(f"Overhead: {total_time - theoretical_min:.0f}ms ({(total_time - theoretical_min) / total_time * 100:.0f}%)")


if __name__ == '__main__':
    measure_weight_load_only()
    trace_load_phases()
