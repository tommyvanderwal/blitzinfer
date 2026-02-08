#!/usr/bin/env python3
"""Direct test of llama loading after gpt-oss.

This mimics exactly what the server does, but without FastAPI/uvicorn,
to isolate if the crash is related to the async context.
"""

import os
import sys
import gc
import time
from datetime import datetime

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CRASH_LOG = os.path.expanduser('~/llama_direct_crash.log')


def crash_log(msg: str):
    """Write to crash log with immediate sync."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(CRASH_LOG, 'a') as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup
    from blitzinfer.orchestrator.standby_manager import StandbyManager

    # Clear log
    with open(CRASH_LOG, 'w') as f:
        f.write(f"=== LLAMA DIRECT TEST ===\n")
        f.write(f"Started: {datetime.now().isoformat()}\n\n")

    crash_log("Creating 80GB arena...")
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    crash_log("Arena ready")

    # Load gpt-oss (same as server initial model)
    crash_log("Loading gpt-oss-120b...")
    crash_log("LLM() constructor starting for gpt-oss")
    llm = LLM(
        model="openai/gpt-oss-120b",
        gpu_memory_utilization=0.94,
        max_model_len=131072,
        trust_remote_code=True,
        enforce_eager=True,
    )
    crash_log("LLM() constructor done for gpt-oss")

    # Quick inference
    crash_log("Inference on gpt-oss...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    crash_log(f"Output: {outputs[0].outputs[0].text}")

    # Start prefetch like server does (this is what server does after switch)
    crash_log("Starting prefetch of gpt-oss (simulating server behavior)...")
    standby.start_prefetch("openai/gpt-oss-120b")
    time.sleep(3)
    crash_log(f"Prefetch state: {standby.get_state().name}")

    # Cleanup gpt-oss
    crash_log("Cleanup gpt-oss starting...")
    crash_log("torch.cuda.synchronize() starting")
    torch.cuda.synchronize()
    crash_log("torch.cuda.synchronize() done")

    crash_log("full_cleanup() starting")
    freed = full_cleanup(llm)
    crash_log(f"full_cleanup() done, freed={freed:.1f}GB")
    llm = None

    crash_log("gc.collect() x3")
    for _ in range(3):
        gc.collect()

    crash_log("torch.cuda.empty_cache()")
    torch.cuda.empty_cache()

    # Wait for prefetch if running (like server does)
    if standby.get_state().name == "LOADING":
        crash_log("Waiting for prefetch to complete...")
        standby.wait_for_load(timeout=120)
        crash_log("Prefetch done")

    # Now load llama - THIS IS WHERE SERVER CRASHES
    crash_log("=" * 50)
    crash_log("LOADING LLAMA - this is where server crashes")
    crash_log("=" * 50)

    crash_log("LLM() constructor starting for llama")
    llm = LLM(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        gpu_memory_utilization=0.94,
        max_model_len=131072,
        trust_remote_code=True,
        enforce_eager=True,
    )
    crash_log("LLM() constructor done for llama")

    # Quick inference
    crash_log("Inference on llama...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    crash_log(f"Output: {outputs[0].outputs[0].text}")

    crash_log("SUCCESS - llama loaded without crash!")

    # Cleanup
    full_cleanup(llm)
    standby.shutdown()
    crash_log("Test complete")


if __name__ == "__main__":
    main()
