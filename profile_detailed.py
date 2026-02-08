#!/usr/bin/env python3
"""Detailed profiling to find single-threaded bottlenecks."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time

def profile():
    timings = {}

    # Phase 1: Import torch
    t0 = time.time()
    import torch
    timings['import_torch'] = time.time() - t0

    # Phase 2: Import vllm base
    t0 = time.time()
    import vllm
    timings['import_vllm'] = time.time() - t0

    # Phase 3: Import LLM class
    t0 = time.time()
    from vllm import LLM, SamplingParams
    timings['import_llm_class'] = time.time() - t0

    # Phase 4: Parse engine args (no model loading yet)
    t0 = time.time()
    from vllm.engine.arg_utils import EngineArgs
    engine_args = EngineArgs(
        model='Qwen/Qwen2.5-7B-Instruct',
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    timings['parse_engine_args'] = time.time() - t0

    # Phase 5: Create LLM (main loading)
    print("Creating LLM instance (watch CPU usage)...")
    t0 = time.time()
    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    timings['create_llm'] = time.time() - t0

    print("\n" + "=" * 50)
    print("DETAILED TIMING")
    print("=" * 50)
    for phase, elapsed in timings.items():
        print(f"{phase:25s}: {elapsed:.3f}s")
    print(f"{'TOTAL':25s}: {sum(timings.values()):.3f}s")


if __name__ == '__main__':
    profile()
