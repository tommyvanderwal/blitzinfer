#!/usr/bin/env python3
"""Test whether triton cache affects gpt-oss-120b garbled output.

Hypothesis: During reload from Qwen->gpt-oss, a wrongly-cached triton kernel
is being used (from Qwen's compilation), causing garbled output.

Test plan:
1. Clear triton cache
2. Load Qwen (compiles Qwen kernels)
3. Reload to gpt-oss
4. See if it crashes (same PTX error as fresh load) or produces garbled output

If it crashes: the kernel was being wrongly reused from cache
If garbled: the issue is elsewhere
"""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import shutil
import time
import torch

TRITON_CACHE = os.path.expanduser("~/.triton/cache")

def main():
    # Step 0: Report triton cache status
    if os.path.exists(TRITON_CACHE):
        n_dirs = len(os.listdir(TRITON_CACHE))
        print(f"Triton cache: {TRITON_CACHE} ({n_dirs} entries)")
        print("Clearing triton cache...")
        shutil.rmtree(TRITON_CACHE)
        os.makedirs(TRITON_CACHE, exist_ok=True)
        print("Cache cleared.")
    else:
        print("No triton cache found.")

    from sglang.srt.entrypoints.engine import Engine

    # Step 1: Load Qwen
    print("\n=== STEP 1: Load Qwen (will compile triton kernels) ===")
    t0 = time.perf_counter()
    engine = Engine(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
    )
    print(f"  Qwen loaded in {time.perf_counter()-t0:.1f}s")

    # Verify Qwen works
    result = engine.generate(
        prompt="What is 2+3?",
        sampling_params={"temperature": 0, "max_new_tokens": 50},
    )
    print(f"  Qwen output: {result.get('text', '')[:80]!r}")

    # Step 2: Reload to gpt-oss (with cleared triton cache)
    print("\n=== STEP 2: Reload to gpt-oss (no triton cache) ===")
    t0 = time.perf_counter()
    try:
        ok, msg = engine.reload_model(
            "openai/gpt-oss-120b",
            server_args_overrides={
                "context_length": 131072,
                "moe_runner_backend": "triton_kernel",
                "dtype": "bfloat16",
            },
        )
        print(f"  Reload: ok={ok}, time={time.perf_counter()-t0:.1f}s")
        if not ok:
            print(f"  Error: {msg[:300]}")
            engine.shutdown()
            return
    except Exception as e:
        print(f"  Reload CRASHED: {e}")
        engine.shutdown()
        return

    # Step 3: Test generation
    print("\n=== STEP 3: Test gpt-oss generation ===")
    result = engine.generate(
        prompt="What is 2+3? Answer with just the number.",
        sampling_params={"temperature": 0, "max_new_tokens": 50},
    )
    text = result.get("text", "")
    print(f"  Output: {text[:120]!r}")

    # Check if output is coherent
    has_5 = "5" in text
    is_short = len(text.strip()) < 20
    print(f"  Contains '5': {has_5}")
    print(f"  Short answer: {is_short}")

    if has_5 and is_short:
        print("  RESULT: Output looks CORRECT!")
    else:
        print("  RESULT: Output still garbled or wrong")

    # Check how many triton cache entries were created
    if os.path.exists(TRITON_CACHE):
        n_dirs = len(os.listdir(TRITON_CACHE))
        print(f"\n  Triton cache now has {n_dirs} entries")

    engine.shutdown()
    print("\nDone")


if __name__ == "__main__":
    main()
