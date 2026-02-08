#!/usr/bin/env python3
"""Test sequential loading of two large models to verify cleanup works."""
import os
import gc
import time
import subprocess
import multiprocessing

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup

import torch

MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Instruct"

VLLM_CONFIG = {
    "dtype": "bfloat16",
    "max_model_len": 100000,
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}


def gpu_free():
    return (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3


def nvidia_smi_gpu_processes():
    """Get GPU processes from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def unload(llm):
    """Unload model - multiprocessing handles cleanup via subprocess termination."""
    print(f"  Before del: GPU processes:\n    {nvidia_smi_gpu_processes()}")

    del llm

    # Wait for subprocess to fully terminate with monitoring
    for i in range(30):  # Up to 15 seconds
        children = multiprocessing.active_children()
        procs = nvidia_smi_gpu_processes()
        print(f"  Tick {i}: children={len(children)}, GPU procs: {procs or '(none)'}")

        if not children and not procs:
            break
        if not children and 'EngineCore' not in procs:
            break

        time.sleep(0.5)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"  After cleanup: GPU processes: {nvidia_smi_gpu_processes() or '(none)'}")


def main():
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("TWO MODEL SWITCH TEST")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Initial free: {gpu_free():.1f} GB")

    sampling = SamplingParams(max_tokens=20, temperature=0.7)

    # Load Model A
    print(f"\n=== Loading {MODEL_A} ===")
    t0 = time.time()
    llm_a = LLM(model=MODEL_A, **VLLM_CONFIG)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"GPU free: {gpu_free():.1f} GB")

    # Inference
    out = llm_a.generate(["Hello, I am"], sampling)
    print(f"Output A: {out[0].outputs[0].text[:40]}...")

    # Unload Model A
    print(f"\n=== Unloading Model A ===")
    t0 = time.time()
    unload(llm_a)
    print(f"Unloaded in {time.time() - t0:.1f}s")
    print(f"GPU free: {gpu_free():.1f} GB")

    # Load Model B
    print(f"\n=== Loading {MODEL_B} ===")
    t0 = time.time()
    llm_b = LLM(model=MODEL_B, **VLLM_CONFIG)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"GPU free: {gpu_free():.1f} GB")

    # Inference
    out = llm_b.generate(["Hello, I am"], sampling)
    print(f"Output B: {out[0].outputs[0].text[:40]}...")

    # Cleanup
    print(f"\n=== Cleanup ===")
    unload(llm_b)
    print(f"Final GPU free: {gpu_free():.1f} GB")

    print("\n" + "=" * 60)
    print("SUCCESS: Both models loaded and ran inference")
    print("=" * 60)


if __name__ == '__main__':
    main()
