#!/usr/bin/env python3
"""Test sequential loading of two large models - avoid CUDA in main process."""
import os
import gc
import time
import subprocess as sp
import multiprocessing

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup

# DO NOT import torch until after model loading - avoid CUDA init in main process

MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Instruct"

# max_model_len: not specified, let model use its default max
VLLM_CONFIG = {
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}


def nvidia_smi_free_gb():
    """Get free GPU memory from nvidia-smi (doesn't touch CUDA)."""
    result = sp.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024  # Convert MiB to GiB


def nvidia_smi_processes():
    """Get GPU processes from nvidia-smi."""
    result = sp.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip() or "(none)"


def unload(llm):
    """Unload model - wait for subprocess to terminate."""
    print(f"  GPU processes before del: {nvidia_smi_processes()}")

    del llm

    # Wait for subprocess to fully terminate
    for i in range(60):  # Up to 30 seconds
        time.sleep(0.5)
        procs = nvidia_smi_processes()
        children = multiprocessing.active_children()

        # Check if EngineCore process is gone
        if 'EngineCore' not in procs and len(children) == 0:
            print(f"  Subprocess terminated after {(i+1)*0.5:.1f}s")
            break

        if i % 4 == 0:  # Print every 2 seconds
            print(f"  Waiting... children={len(children)}, procs={procs}")

    gc.collect()
    time.sleep(1)

    print(f"  GPU free after unload: {nvidia_smi_free_gb():.1f} GB")
    print(f"  GPU processes after unload: {nvidia_smi_processes()}")


def main():
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("TWO MODEL SWITCH TEST (Clean)")
    print("=" * 70)
    print(f"GPU free initially: {nvidia_smi_free_gb():.1f} GB")

    sampling = SamplingParams(max_tokens=20, temperature=0.7)

    # Load Model A
    print(f"\n=== Loading {MODEL_A} ===")
    t0 = time.time()
    llm_a = LLM(model=MODEL_A, **VLLM_CONFIG)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Inference
    out = llm_a.generate(["Hello, I am"], sampling)
    print(f"Output A: {out[0].outputs[0].text[:40]}...")

    # Unload Model A
    print(f"\n=== Unloading Model A ===")
    t0 = time.time()
    unload(llm_a)
    print(f"Unload took {time.time() - t0:.1f}s")

    # Check if we have enough memory
    free_gb = nvidia_smi_free_gb()
    print(f"\nGPU free before loading Model B: {free_gb:.1f} GB")
    if free_gb < 90:
        print(f"WARNING: Only {free_gb:.1f} GB free, need ~90 GB")
        print("Waiting additional 10 seconds...")
        time.sleep(10)
        free_gb = nvidia_smi_free_gb()
        print(f"GPU free after wait: {free_gb:.1f} GB")

    # Load Model B
    print(f"\n=== Loading {MODEL_B} ===")
    t0 = time.time()
    llm_b = LLM(model=MODEL_B, **VLLM_CONFIG)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Inference
    out = llm_b.generate(["Hello, I am"], sampling)
    print(f"Output B: {out[0].outputs[0].text[:40]}...")

    # Cleanup
    print(f"\n=== Cleanup ===")
    unload(llm_b)
    print(f"Final GPU free: {nvidia_smi_free_gb():.1f} GB")

    print("\n" + "=" * 70)
    print("SUCCESS: Both models loaded and ran inference")
    print("=" * 70)


if __name__ == '__main__':
    main()
