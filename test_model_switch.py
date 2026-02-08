#!/usr/bin/env python3
"""Test model switching between Qwen3-VL-32B and Llama-3.1-70B-AWQ"""

import time
import os
import gc
import torch

# Environment for 780M (ROCm)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from vllm import LLM, SamplingParams

KV_CACHE_GB = 15
KV_CACHE_BYTES = KV_CACHE_GB * 1024 * 1024 * 1024

def load_qwen():
    """Load Qwen3-VL-32B for vision tasks"""
    print("\n" + "="*60)
    print("Loading Qwen3-VL-32B-Instruct...")
    print("="*60)
    start = time.time()
    llm = LLM(
        model="Qwen/Qwen3-VL-32B-Instruct",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        disable_log_stats=True,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
        kv_cache_memory_bytes=KV_CACHE_BYTES,
    )
    load_time = time.time() - start
    print(f"Qwen loaded in {load_time:.2f}s")
    return llm, load_time

def load_llama():
    """Load Llama-3.1-70B-AWQ for text tasks"""
    print("\n" + "="*60)
    print("Loading Llama-3.1-70B-Instruct-AWQ-INT4...")
    print("="*60)
    start = time.time()
    llm = LLM(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        disable_log_stats=True,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
        kv_cache_memory_bytes=KV_CACHE_BYTES,
    )
    load_time = time.time() - start
    print(f"Llama loaded in {load_time:.2f}s")
    return llm, load_time

def unload_model(llm):
    """Properly unload model and free GPU memory"""
    print("\nUnloading model...")
    start = time.time()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    unload_time = time.time() - start
    print(f"Unloaded in {unload_time:.2f}s")
    return unload_time

def test_text(llm, model_name):
    """Quick text generation test"""
    print(f"\n{model_name} text test...")
    start = time.time()
    outputs = llm.generate(
        ["Write a one-sentence summary of machine learning:"],
        SamplingParams(max_tokens=50)
    )
    gen_time = time.time() - start
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Output: {outputs[0].outputs[0].text[:100]}...")
    return gen_time

def main():
    print("="*60)
    print("BlitzInfer Model Switching Test")
    print("="*60)

    results = {
        "qwen_loads": [],
        "llama_loads": [],
        "unloads": [],
        "switches": [],
    }

    num_switches = 3

    for i in range(num_switches):
        print(f"\n{'#'*60}")
        print(f"# Switch cycle {i+1}/{num_switches}")
        print(f"{'#'*60}")

        # Load Qwen
        switch_start = time.time()
        qwen, load_time = load_qwen()
        results["qwen_loads"].append(load_time)
        test_text(qwen, "Qwen")

        # Unload Qwen
        unload_time = unload_model(qwen)
        results["unloads"].append(unload_time)

        # Load Llama
        llama, load_time = load_llama()
        results["llama_loads"].append(load_time)
        test_text(llama, "Llama")

        # Unload Llama
        unload_time = unload_model(llama)
        results["unloads"].append(unload_time)

        switch_time = time.time() - switch_start
        results["switches"].append(switch_time)
        print(f"\nCycle {i+1} total time: {switch_time:.2f}s")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Qwen load times: {[f'{t:.1f}s' for t in results['qwen_loads']]}")
    print(f"  Average: {sum(results['qwen_loads'])/len(results['qwen_loads']):.1f}s")
    print(f"Llama load times: {[f'{t:.1f}s' for t in results['llama_loads']]}")
    print(f"  Average: {sum(results['llama_loads'])/len(results['llama_loads']):.1f}s")
    print(f"Unload times: avg {sum(results['unloads'])/len(results['unloads']):.2f}s")
    print(f"Full cycle times: {[f'{t:.1f}s' for t in results['switches']]}")
    print(f"  Average: {sum(results['switches'])/len(results['switches']):.1f}s")

if __name__ == "__main__":
    main()
