#!/usr/bin/env python3
"""
Large model swap test for RTX PRO 6000 (95GB VRAM).
Tests swapping between GPT-OSS-120B (~60GB) and GLM-4.6V-AWQ-4bit (~65GB).
"""

import os
import sys
import time
import gc
import json
from datetime import datetime

# Force vLLM V1 single-process mode for clean GPU management
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"

import torch
from vllm import LLM, SamplingParams

# Models to test
GPT_MODEL = "openai/gpt-oss-120b"
GLM_MODEL = "cyankiwi/GLM-4.6V-AWQ-4bit"

def get_gpu_memory():
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2)
        }
    return {}

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model(model_name: str):
    """Load model with profiled timing."""
    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"Pre-load GPU: {get_gpu_memory()}")

    start = time.perf_counter()

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.90,  # Use most of 95GB VRAM
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        enforce_eager=False,  # Use CUDA graphs on NVIDIA
        trust_remote_code=True,
    )

    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.2f}s")
    print(f"Post-load GPU: {get_gpu_memory()}")

    return llm, load_time

def unload_model(llm):
    """Unload model with memory cleanup."""
    start = time.perf_counter()

    # Access internal engine for cleanup
    if hasattr(llm, 'llm_engine'):
        engine = llm.llm_engine
        if hasattr(engine, 'model_executor'):
            executor = engine.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'gpu_model_runner'):
                    model_runner = worker.gpu_model_runner

                    # Clear model parameters
                    if hasattr(model_runner, 'model') and model_runner.model is not None:
                        model = model_runner.model
                        for param in model.parameters():
                            param.data = torch.empty(0, device='cpu')

                    # Clear KV caches
                    if hasattr(model_runner, 'kv_caches'):
                        for i, cache in enumerate(model_runner.kv_caches):
                            if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                        model_runner.kv_caches.clear()

    del llm
    clear_gpu_memory()

    unload_time = time.perf_counter() - start
    print(f"Unload time: {unload_time:.2f}s")
    print(f"Post-unload GPU: {get_gpu_memory()}")

    return unload_time

def run_inference(llm, model_name: str):
    """Run a quick inference test."""
    prompt = "What is the capital of France? Answer in one word:"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    inference_time = time.perf_counter() - start

    result = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    tok_s = tokens / inference_time if inference_time > 0 else 0

    print(f"Inference: '{result}' ({tokens} tokens in {inference_time:.2f}s = {tok_s:.1f} tok/s)")
    return inference_time, tokens

def main():
    num_swaps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"BlitzInfer Large Model Swap Test")
    print(f"Target: {num_swaps} swaps between {GPT_MODEL} and {GLM_MODEL}")
    print(f"System: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Time: {datetime.now().isoformat()}")

    results = {
        "system": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "num_swaps": num_swaps,
        "swaps": []
    }

    models = [GPT_MODEL, GLM_MODEL]

    try:
        for swap_idx in range(num_swaps):
            model_name = models[swap_idx % 2]
            print(f"\n{'#'*60}")
            print(f"SWAP {swap_idx + 1}/{num_swaps}")

            swap_start = time.perf_counter()

            # Load
            llm, load_time = load_model(model_name)

            # Inference
            inf_time, tokens = run_inference(llm, model_name)

            # Unload
            unload_time = unload_model(llm)

            swap_total = time.perf_counter() - swap_start

            swap_data = {
                "swap_number": swap_idx + 1,
                "model": model_name,
                "load_time_s": round(load_time, 2),
                "inference_time_s": round(inf_time, 2),
                "unload_time_s": round(unload_time, 2),
                "total_time_s": round(swap_total, 2),
                "tokens": tokens,
                "gpu_after": get_gpu_memory()
            }
            results["swaps"].append(swap_data)

            print(f"Swap {swap_idx + 1} complete: {swap_total:.2f}s total")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if results["swaps"]:
        load_times = [s["load_time_s"] for s in results["swaps"]]
        unload_times = [s["unload_time_s"] for s in results["swaps"]]
        total_times = [s["total_time_s"] for s in results["swaps"]]

        print(f"Completed swaps: {len(results['swaps'])}/{num_swaps}")
        print(f"Avg load time: {sum(load_times)/len(load_times):.2f}s")
        print(f"Avg unload time: {sum(unload_times)/len(unload_times):.2f}s")
        print(f"Avg total swap time: {sum(total_times)/len(total_times):.2f}s")
        print(f"Total test time: {sum(total_times):.2f}s")

        results["summary"] = {
            "completed_swaps": len(results["swaps"]),
            "avg_load_time_s": round(sum(load_times)/len(load_times), 2),
            "avg_unload_time_s": round(sum(unload_times)/len(unload_times), 2),
            "avg_swap_time_s": round(sum(total_times)/len(total_times), 2),
            "total_time_s": round(sum(total_times), 2)
        }

    # Save results
    results_file = f"swap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
