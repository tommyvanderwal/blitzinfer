#!/usr/bin/env python3
"""Test the effect of OS page cache on model loading."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import subprocess
import sys
import time


def load_model_timing():
    """Load model and return timing."""
    start = time.time()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )

    load_time = time.time() - start

    # Quick inference test
    out = llm.generate(['Hi'], SamplingParams(max_tokens=1, temperature=0))

    # Cleanup
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    return load_time


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'inner':
        # Run the actual test
        load_time = load_model_timing()
        print(f"LOAD_TIME: {load_time:.2f}")
    else:
        # Outer script - run twice to test caching
        print("=" * 50)
        print("Testing OS page cache effect on model loading")
        print("=" * 50)

        # First run - cold
        print("\n[Run 1] Cold cache...")
        result1 = subprocess.run(
            [sys.executable, __file__, 'inner'],
            capture_output=True, text=True, timeout=300
        )
        time1 = float([l for l in result1.stdout.split('\n') if 'LOAD_TIME:' in l][0].split(':')[1])
        print(f"Cold load time: {time1:.2f}s")

        # Second run - warm cache
        print("\n[Run 2] Warm cache (OS page cache)...")
        result2 = subprocess.run(
            [sys.executable, __file__, 'inner'],
            capture_output=True, text=True, timeout=300
        )
        time2 = float([l for l in result2.stdout.split('\n') if 'LOAD_TIME:' in l][0].split(':')[1])
        print(f"Warm load time: {time2:.2f}s")

        print("\n" + "=" * 50)
        print(f"Improvement: {time1 - time2:.2f}s ({(1 - time2/time1)*100:.1f}%)")
        print("=" * 50)
