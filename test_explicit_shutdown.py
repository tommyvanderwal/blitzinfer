#!/usr/bin/env python3
"""Test explicit engine_core shutdown before deletion."""
import os
import gc
import time
import subprocess as sp
import multiprocessing

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup

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
    result = sp.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def nvidia_smi_processes():
    result = sp.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip() or "(none)"


def unload_with_explicit_shutdown(llm):
    """Unload with explicit engine_core shutdown."""
    print(f"  GPU procs before: {nvidia_smi_processes()}")

    # Explicitly shutdown engine_core
    print("  Calling engine_core.shutdown()...")
    try:
        llm.llm_engine.engine_core.shutdown()
        print("  Shutdown called successfully")
    except Exception as e:
        print(f"  Shutdown error: {e}")

    time.sleep(2)
    print(f"  GPU procs after shutdown: {nvidia_smi_processes()}")

    # Now delete
    print("  Deleting llm object...")
    del llm

    # Wait for cleanup
    for i in range(30):
        time.sleep(0.5)
        procs = nvidia_smi_processes()
        if 'EngineCore' not in procs:
            if i > 0:
                print(f"  EngineCore process terminated after {(i+1)*0.5:.1f}s")
            break

    gc.collect()
    time.sleep(1)

    free_gb = nvidia_smi_free_gb()
    print(f"  Final GPU free: {free_gb:.1f} GB")
    print(f"  Final GPU procs: {nvidia_smi_processes()}")
    return free_gb


def main():
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("EXPLICIT SHUTDOWN TEST")
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

    # Unload with explicit shutdown
    print(f"\n=== Unloading Model A with explicit shutdown ===")
    free_after = unload_with_explicit_shutdown(llm_a)

    if free_after < 90:
        print(f"\nWARNING: Only {free_after:.1f} GB free!")
        print("Memory not properly released. Cannot load second model.")
        return

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
    unload_with_explicit_shutdown(llm_b)

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)


if __name__ == '__main__':
    main()
