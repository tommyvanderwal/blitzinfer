#!/usr/bin/env python3
"""
Deep profiling of vLLM initialization to find optimization opportunities.

Goal: Break down the ~3.2s vLLM overhead to find what can be cached/reused.
"""

import gc
import os
import sys
import time
import types
import functools
from contextlib import contextmanager
from typing import Dict, List, Tuple

# ROCm setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch

# Timing data structure
TIMINGS: Dict[str, List[float]] = {}
CALL_STACK: List[Tuple[str, float]] = []

def record_timing(name: str, duration_ms: float):
    """Record a timing measurement."""
    if name not in TIMINGS:
        TIMINGS[name] = []
    TIMINGS[name].append(duration_ms)

@contextmanager
def timed_section(name: str):
    """Context manager for timing a code section."""
    t0 = time.perf_counter()
    CALL_STACK.append((name, t0))
    try:
        yield
    finally:
        duration = (time.perf_counter() - t0) * 1000
        CALL_STACK.pop()
        record_timing(name, duration)
        indent = "  " * len(CALL_STACK)
        print(f"{indent}[{duration:7.1f}ms] {name}")


def patch_for_profiling():
    """Patch vLLM functions to add timing instrumentation."""
    import vllm
    from vllm import LLM
    from vllm.engine.llm_engine import LLMEngine
    from vllm.model_executor.model_loader import loader as model_loader

    # Patch LLMEngine.__init__
    original_engine_init = LLMEngine.__init__
    @functools.wraps(original_engine_init)
    def profiled_engine_init(self, *args, **kwargs):
        with timed_section("LLMEngine.__init__"):
            return original_engine_init(self, *args, **kwargs)
    LLMEngine.__init__ = profiled_engine_init

    # Patch model loading
    if hasattr(model_loader, 'get_model'):
        original_get_model = model_loader.get_model
        @functools.wraps(original_get_model)
        def profiled_get_model(*args, **kwargs):
            with timed_section("model_loader.get_model"):
                return original_get_model(*args, **kwargs)
        model_loader.get_model = profiled_get_model

    # Patch tokenizer loading
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer
        import vllm.transformers_utils.tokenizer as tokenizer_module

        original_get_tokenizer = get_tokenizer
        @functools.wraps(original_get_tokenizer)
        def profiled_get_tokenizer(*args, **kwargs):
            with timed_section("get_tokenizer"):
                return original_get_tokenizer(*args, **kwargs)
        tokenizer_module.get_tokenizer = profiled_get_tokenizer
    except ImportError:
        pass

    print("[Profiler] vLLM patched for detailed timing")


def profile_llm_init():
    """Profile a single LLM initialization in detail."""
    from vllm import LLM, SamplingParams

    # Apply profiling patches
    patch_for_profiling()

    # Apply Blitz patch
    import blitz_vllm_patch
    blitz_vllm_patch.patch_vllm()

    print("=" * 70)
    print("VLLM INITIALIZATION PROFILING")
    print("=" * 70)

    # Warmup GPU
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"\nInitial GPU memory: {free_mem:.1f} GB free\n")

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # === First Load (includes all initialization) ===
    print("-" * 70)
    print("FIRST LOAD (includes buffer allocation)")
    print("-" * 70)

    TIMINGS.clear()
    t0 = time.perf_counter()

    with timed_section("LLM()"):
        llm = LLM(model=model_name, **config)

    total1 = (time.perf_counter() - t0) * 1000
    print(f"\nTotal first load: {total1:.0f}ms")

    # Verify
    params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello"], params)
    print(f"Verify: {outputs[0].outputs[0].text[:40]}...")

    # === Cleanup ===
    print("\n" + "-" * 70)
    print("CLEANUP")
    print("-" * 70)

    del llm
    gc.collect()

    t0 = time.perf_counter()
    cleanup_time, cleared = blitz_vllm_patch.fast_cleanup()
    print(f"Cleanup: {cleanup_time:.0f}ms (cleared {cleared} params)")

    # === Second Load (warmed up) ===
    print("\n" + "-" * 70)
    print("SECOND LOAD (warmed up)")
    print("-" * 70)

    TIMINGS.clear()
    t0 = time.perf_counter()

    with timed_section("LLM()"):
        llm2 = LLM(model=model_name, **config)

    total2 = (time.perf_counter() - t0) * 1000
    print(f"\nTotal second load: {total2:.0f}ms")

    # Verify
    outputs = llm2.generate(["World"], params)
    print(f"Verify: {outputs[0].outputs[0].text[:40]}...")

    # === Summary ===
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    print(f"\nFirst load:  {total1:.0f}ms")
    print(f"Second load: {total2:.0f}ms")
    print(f"Difference:  {total1 - total2:.0f}ms (buffer pre-allocation)")

    # Detailed breakdown
    print("\nDetailed timings (second load):")
    for name, times in sorted(TIMINGS.items(), key=lambda x: -max(x[1])):
        avg = sum(times) / len(times)
        print(f"  {name}: {avg:.1f}ms (n={len(times)})")

    # Cleanup
    del llm2
    gc.collect()
    blitz_vllm_patch.fast_cleanup()


def profile_component_timing():
    """Profile individual vLLM components to understand the overhead."""
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig
    from vllm.engine.arg_utils import EngineArgs
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    import blitz_vllm_patch
    blitz_vllm_patch.patch_vllm()

    print("=" * 70)
    print("COMPONENT-LEVEL PROFILING")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    timings = {}

    # 1. HuggingFace config loading
    print("\n>>> HuggingFace Config Loading...")
    t0 = time.perf_counter()
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    timings['hf_config'] = (time.perf_counter() - t0) * 1000
    print(f"  HF config: {timings['hf_config']:.0f}ms")

    # 2. Tokenizer loading
    print("\n>>> Tokenizer Loading...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    timings['tokenizer'] = (time.perf_counter() - t0) * 1000
    print(f"  Tokenizer: {timings['tokenizer']:.0f}ms")

    # 3. Model download/cache check
    print("\n>>> Model Path Resolution...")
    t0 = time.perf_counter()
    local_path = snapshot_download(model_name)
    timings['snapshot_download'] = (time.perf_counter() - t0) * 1000
    print(f"  Snapshot download: {timings['snapshot_download']:.0f}ms")

    # 4. EngineArgs parsing
    print("\n>>> EngineArgs Parsing...")
    t0 = time.perf_counter()
    engine_args = EngineArgs(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        enforce_eager=True,
    )
    timings['engine_args'] = (time.perf_counter() - t0) * 1000
    print(f"  EngineArgs: {timings['engine_args']:.0f}ms")

    # 5. Full LLM load for comparison
    print("\n>>> Full LLM Load...")

    # Warmup GPU first
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    timings['full_llm'] = (time.perf_counter() - t0) * 1000
    print(f"  Full LLM: {timings['full_llm']:.0f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("COMPONENT TIMING SUMMARY")
    print("=" * 70)

    overhead = timings['full_llm'] - 1600  # Subtract ~1600ms for Blitz weight loading

    print(f"\nComponent times:")
    print(f"  HF config:      {timings['hf_config']:>6.0f}ms")
    print(f"  Tokenizer:      {timings['tokenizer']:>6.0f}ms")
    print(f"  Snapshot check: {timings['snapshot_download']:>6.0f}ms")
    print(f"  EngineArgs:     {timings['engine_args']:>6.0f}ms")

    known = timings['hf_config'] + timings['tokenizer'] + timings['snapshot_download'] + timings['engine_args']
    print(f"\n  Known overhead: {known:>6.0f}ms")
    print(f"  Full LLM:       {timings['full_llm']:>6.0f}ms")
    print(f"  Weight loading: ~1600ms (Blitz)")
    print(f"  Unaccounted:    {overhead - known:>6.0f}ms")

    print("\n>>> Cacheable components:")
    print(f"  - Tokenizer: {timings['tokenizer']:.0f}ms (can cache for same model family)")
    print(f"  - HF config: {timings['hf_config']:.0f}ms (can cache)")

    # Cleanup
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()


def deep_profile_with_hooks():
    """Use Python hooks to trace all vLLM function calls."""
    from vllm import LLM, SamplingParams
    import blitz_vllm_patch

    # Track function call timings
    call_times: Dict[str, List[float]] = {}
    call_starts: Dict[int, Tuple[str, float]] = {}

    def trace_calls(frame, event, arg):
        """Trace function calls and measure time."""
        if event == 'call':
            code = frame.f_code
            filename = code.co_filename

            # Only trace vllm and our code
            if 'vllm' in filename or 'blitz' in filename:
                name = f"{code.co_filename.split('/')[-1]}:{code.co_name}"
                call_starts[id(frame)] = (name, time.perf_counter())

        elif event == 'return':
            frame_id = id(frame)
            if frame_id in call_starts:
                name, start = call_starts.pop(frame_id)
                duration = (time.perf_counter() - start) * 1000
                if duration > 10:  # Only track calls > 10ms
                    if name not in call_times:
                        call_times[name] = []
                    call_times[name].append(duration)

        return trace_calls

    print("=" * 70)
    print("DEEP FUNCTION-LEVEL PROFILING")
    print("=" * 70)

    blitz_vllm_patch.patch_vllm()

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Enable tracing
    print("\n>>> Loading with function tracing enabled...")
    print("    (Only showing functions taking >10ms)")

    sys.settrace(trace_calls)
    t0 = time.perf_counter()

    llm = LLM(model=model_name, **config)

    total = (time.perf_counter() - t0) * 1000
    sys.settrace(None)

    print(f"\nTotal load time: {total:.0f}ms")

    # Show slowest functions
    print("\n" + "-" * 70)
    print("SLOWEST FUNCTIONS (>50ms)")
    print("-" * 70)

    sorted_calls = sorted(call_times.items(), key=lambda x: sum(x[1]), reverse=True)

    for name, times in sorted_calls[:30]:
        total_time = sum(times)
        if total_time > 50:
            avg = total_time / len(times)
            print(f"  {total_time:>7.0f}ms ({len(times):>3}x, avg {avg:>6.1f}ms) {name}")

    # Cleanup
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()

    return call_times


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'components':
            profile_component_timing()
        elif mode == 'deep':
            deep_profile_with_hooks()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python profile_vllm_init.py [components|deep]")
    else:
        profile_llm_init()
