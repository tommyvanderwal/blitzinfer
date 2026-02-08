#!/usr/bin/env python3
"""Baseline vLLM load time without pinned arena."""

import os
import gc
import time
import subprocess

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024

def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def main():
    print("=" * 80)
    print("BASELINE vLLM LOAD (NO PINNED ARENA)")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    _ = torch.randn(1000, device='cuda')
    cleanup()

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print(f"\nModel: {model_name}")
    print("Loading with standard safetensor loader...")

    t0 = time.perf_counter()
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    vllm_time = time.perf_counter() - t0
    print(f"vLLM init: {vllm_time:.2f}s")

    print("\nRunning inference...")
    t0 = time.perf_counter()
    outputs = llm.generate(
        ["What is 2+2? Answer briefly:"],
        SamplingParams(max_tokens=20, temperature=0.7),
    )
    infer_time = time.perf_counter() - t0
    print(f"Inference: {infer_time:.2f}s")
    print(f"Response: {outputs[0].outputs[0].text}")

    del llm
    cleanup()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline vLLM init: {vllm_time:.2f}s")

if __name__ == '__main__':
    main()
