#!/usr/bin/env python3
"""
Profile where time is spent during model loading

This helps identify optimization opportunities.
"""

import time
import os
import sys

# Pre-configure environment
IS_ROCM = os.path.exists("/opt/rocm")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if not IS_ROCM:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
else:
    os.environ["VLLM_SKIP_WARMUP"] = "1"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

PLATFORM = "780M" if IS_ROCM else "RTX PRO 6000"

def timed(name):
    """Decorator to time a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"  [{name}] {elapsed:.2f}s")
            return result, elapsed
        return wrapper
    return decorator


def profile_model_load(model_name: str, dtype: str):
    """Profile the stages of model loading"""
    print(f"\n{'='*60}")
    print(f"Profiling: {model_name}")
    print(f"Platform: {PLATFORM}")
    print(f"{'='*60}")

    timings = {}

    # Stage 1: Import vLLM
    print("\n[1/5] Importing vLLM...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    import torch
    timings["import"] = time.time() - t0
    print(f"  Import time: {timings['import']:.2f}s")

    # Stage 2: Parse engine args
    print("\n[2/5] Parsing engine arguments...")
    t0 = time.time()
    kwargs = {
        "model": model_name,
        "dtype": dtype,
        "max_model_len": 512,
        "max_num_seqs": 2,
        "disable_log_stats": True,
        "enforce_eager": True,
    }
    if IS_ROCM:
        kwargs["compilation_config"] = {"custom_ops": ["none"]}
        kwargs["kv_cache_memory_bytes"] = 10 * 1024**3

    engine_args = EngineArgs(**kwargs)
    timings["args_parse"] = time.time() - t0
    print(f"  Args parse time: {timings['args_parse']:.2f}s")

    # Stage 3: Create vllm_config
    print("\n[3/5] Creating VllmConfig...")
    t0 = time.time()
    vllm_config = engine_args.create_engine_config()
    timings["config_create"] = time.time() - t0
    print(f"  Config create time: {timings['config_create']:.2f}s")

    # Stage 4: Full LLM initialization
    print("\n[4/5] Creating LLM (full init)...")
    t0 = time.time()
    llm = LLM(**kwargs)
    timings["llm_init"] = time.time() - t0
    print(f"  LLM init time: {timings['llm_init']:.2f}s")

    # Stage 5: First inference
    print("\n[5/5] First inference...")
    t0 = time.time()
    output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    timings["first_inference"] = time.time() - t0
    print(f"  First inference time: {timings['first_inference']:.2f}s")
    print(f"  Output: {output[0].outputs[0].text[:40]}")

    # Summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    total = sum(timings.values())
    for stage, t in timings.items():
        pct = (t / total) * 100
        print(f"  {stage:20s}: {t:6.2f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:6.2f}s")

    # Memory info
    free, total_mem = torch.cuda.mem_get_info()
    print(f"\nGPU Memory: {free/1e9:.1f}/{total_mem/1e9:.1f} GiB free")

    return timings, llm


def profile_internal_stages():
    """Profile internal vLLM initialization stages"""
    print(f"\n{'='*60}")
    print("DETAILED INTERNAL PROFILING")
    print(f"{'='*60}")

    import time
    from vllm import LLM, SamplingParams
    import torch

    # Monkey-patch to add timing to internal functions
    from vllm.v1.engine import llm_engine
    from vllm.model_executor.model_loader import loader

    original_init = llm_engine.LLMEngine.__init__

    stage_times = {}

    def timed_init(self, vllm_config, *args, **kwargs):
        print("\n--- LLMEngine.__init__ internal timing ---")

        # Time each major step
        t0 = time.time()

        # Store vllm_config timing
        self.vllm_config = vllm_config
        stage_times["vllm_config_store"] = time.time() - t0

        # Continue with original init but we can't easily break it up
        # So just time the whole thing
        t0 = time.time()
        original_init(self, vllm_config, *args, **kwargs)
        stage_times["engine_init_total"] = time.time() - t0

    llm_engine.LLMEngine.__init__ = timed_init

    # Now load a model
    kwargs = {
        "model": "openai/gpt-oss-120b",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "max_num_seqs": 2,
        "disable_log_stats": True,
        "enforce_eager": True,
    }
    if IS_ROCM:
        kwargs["compilation_config"] = {"custom_ops": ["none"]}
        kwargs["kv_cache_memory_bytes"] = 10 * 1024**3

    print("\nLoading model with internal timing...")
    t0 = time.time()
    llm = LLM(**kwargs)
    total = time.time() - t0

    print(f"\nInternal stage times: {stage_times}")
    print(f"Total: {total:.2f}s")

    return llm, stage_times


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt"

    if model == "gpt":
        timings, llm = profile_model_load("openai/gpt-oss-120b", "bfloat16")
    elif model == "qwen":
        timings, llm = profile_model_load("Qwen/Qwen3-VL-32B-Instruct", "float16")
    elif model == "internal":
        llm, stage_times = profile_internal_stages()
    else:
        print(f"Unknown model: {model}")
        print("Usage: python blitz_profile_load.py [gpt|qwen|internal]")
        sys.exit(1)


if __name__ == "__main__":
    main()
