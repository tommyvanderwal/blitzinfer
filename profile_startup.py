#!/usr/bin/env python3
"""Profile vLLM startup to identify optimization opportunities."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time


def profile_startup():
    """Profile each phase of model startup."""
    timings = {}

    # Phase 1: Import time
    t0 = time.time()
    from vllm import LLM, SamplingParams
    timings['import_vllm'] = time.time() - t0

    # Phase 2: Model instantiation
    t0 = time.time()
    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    timings['model_init'] = time.time() - t0

    # Phase 3: First inference (warmup)
    t0 = time.time()
    out = llm.generate(['Hi'], SamplingParams(max_tokens=1, temperature=0))
    timings['first_inference'] = time.time() - t0

    # Phase 4: Second inference (hot)
    t0 = time.time()
    out = llm.generate(['Hi'], SamplingParams(max_tokens=1, temperature=0))
    timings['second_inference'] = time.time() - t0

    print("\n" + "=" * 50)
    print("STARTUP TIMING BREAKDOWN")
    print("=" * 50)
    for phase, elapsed in timings.items():
        print(f"{phase:20s}: {elapsed:.2f}s")
    print(f"{'TOTAL':20s}: {sum(timings.values()):.2f}s")
    print("=" * 50)

    return timings


if __name__ == '__main__':
    profile_startup()
