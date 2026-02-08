#!/usr/bin/env python3
"""
BlitzInfer Queue Test - Tests model switching with queued requests.

Sends 10 requests in pattern: 2A, 2B, 2A, 2B, 2A
Should trigger 1-2 model switches depending on batching.
"""

import asyncio
import gc
import os
import sys
import time
import types

# Environment setup for ROCm
os.environ['HIP_VISIBLE_DEVICES'] = '0'
# Use single-process mode for fast model switching (~6s vs ~30s with multiprocessing)
# Memory leak is fixed by properly clearing model parameters and KV cache in unload_model()
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
# Note: Removed VLLM_SKIP_WARMUP - it can cause GPU hangs on first inference
# The warmup takes a bit longer but is more stable
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
from blitzinfer.config import ModelConfig, BlitzInferConfig
from blitzinfer.orchestrator import BlitzInferOrchestrator


def get_mem():
    """Get GPU memory usage."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


# Test prompts - short prompts that will generate ~100 tokens
PROMPTS = [
    "Write a short paragraph about the color blue:",
    "Explain what makes the sky appear blue in three sentences:",
    "List five things that are commonly blue:",
    "Describe a blue ocean sunset briefly:",
    "What emotions do people associate with blue? Explain:",
    "Tell me about a famous blue painting:",
    "Why do companies use blue in their logos?",
    "Describe a bluebird in a few sentences:",
    "What foods are naturally blue? List some:",
    "Write about why blue is a calming color:",
]


async def run_queue_test(model_a: str, model_b: str):
    """Run the queue test with two models."""
    print("=" * 70)
    print("BLITZINFER QUEUE TEST")
    print("=" * 70)
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Request pattern: 2A, 2B, 2A, 2B, 2A (10 requests total)")
    print()

    # Initial memory
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"Initial GPU: {used:.1f}GB used, {free:.1f}GB free")
    print()

    # Configure BlitzInfer
    config = BlitzInferConfig(
        models=[
            ModelConfig(
                name=model_a,
                alias="model_a",
                dtype="float16",
                max_model_len=512,
                max_num_batched_tokens=512,
                gpu_memory_utilization=0.25,
                kv_cache_memory_bytes=2 * 1024**3,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            ),
            ModelConfig(
                name=model_b,
                alias="model_b",
                dtype="float16",
                max_model_len=512,
                max_num_batched_tokens=512,
                gpu_memory_utilization=0.25,
                kv_cache_memory_bytes=2 * 1024**3,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            ),
        ],
    )

    # Create orchestrator
    print("-" * 70)
    print("Initializing BlitzInfer orchestrator...")
    print("-" * 70)
    orchestrator = BlitzInferOrchestrator(config)

    # Build request queue: 2A, 2B, 2A, 2B, 2A (10 requests total)
    request_queue = []
    models = [model_a, model_b, model_a, model_b, model_a]
    prompt_idx = 0
    for i, model in enumerate(models):
        for j in range(2):
            request_queue.append({
                "model": model,
                "prompt": PROMPTS[prompt_idx % len(PROMPTS)],
                "max_tokens": 100,
                "request_id": f"{i * 2 + j + 1}",
            })
            prompt_idx += 1

    print(f"\nRequest queue ({len(request_queue)} requests):")
    for i, req in enumerate(request_queue):
        model_label = "A" if req["model"] == model_a else "B"
        print(f"  {i + 1}. Model {model_label}: {req['prompt'][:40]}...")

    # Execute requests
    print()
    print("-" * 70)
    print("Executing requests...")
    print("-" * 70)

    results = []
    total_start = time.perf_counter()

    for req in request_queue:
        req_start = time.perf_counter()
        model_label = "A" if req["model"] == model_a else "B"
        print(f"\nRequest {req['request_id']} (Model {model_label})...")

        try:
            result = await orchestrator.generate(
                model=req["model"],
                prompt=req["prompt"],
                max_tokens=req["max_tokens"],
                temperature=0.7,
            )
            req_time = (time.perf_counter() - req_start) * 1000
            results.append({
                "request_id": req["request_id"],
                "model": model_label,
                "success": True,
                "time_ms": req_time,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "text_preview": result.text[:50] + "..." if len(result.text) > 50 else result.text,
            })
            print(f"  Done in {req_time:.0f}ms - {result.completion_tokens} tokens")
        except Exception as e:
            req_time = (time.perf_counter() - req_start) * 1000
            results.append({
                "request_id": req["request_id"],
                "model": model_label,
                "success": False,
                "time_ms": req_time,
                "error": str(e),
            })
            print(f"  FAILED in {req_time:.0f}ms: {e}")

    total_time = (time.perf_counter() - total_start) * 1000

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Request results
    print(f"\n{'Req':<5} {'Model':<8} {'Status':<8} {'Time':<10} {'Tokens':<10}")
    print("-" * 50)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        tokens = f"{r.get('completion_tokens', '-')}" if r["success"] else "-"
        print(f"{r['request_id']:<5} {r['model']:<8} {status:<8} {r['time_ms']:<10.0f} {tokens:<10}")

    # Statistics
    successful = [r for r in results if r["success"]]
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Total time: {total_time:.0f}ms ({total_time/1000:.1f}s)")

    if successful:
        total_tokens = sum(r["completion_tokens"] for r in successful)
        avg_time = sum(r["time_ms"] for r in successful) / len(successful)
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average request time: {avg_time:.0f}ms")

    # Switch metrics
    switches = orchestrator.get_switch_metrics()
    print(f"\nModel switches: {len(switches)}")
    for i, sw in enumerate(switches):
        print(f"  {i + 1}. {sw.from_model or 'None'} → {sw.to_model}: {sw.total_time:.2f}s")

    if switches:
        avg_switch = sum(sw.total_time for sw in switches) / len(switches)
        print(f"Average switch time: {avg_switch:.2f}s")

    # Cleanup
    orchestrator.shutdown()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"\nFinal GPU: {used:.1f}GB used, {free:.1f}GB free")
    print("=" * 70)

    return len(successful) == len(results)


def main():
    # Default models - smaller models for iGPU (stable with multiple switches)
    # 7B models cause GPU hangs after 3-4 switches on shared-memory iGPU
    model_a = "Qwen/Qwen2.5-3B-Instruct"
    model_b = "Qwen/Qwen2.5-1.5B-Instruct"

    # Allow override from command line
    # For discrete GPU with dedicated VRAM, use larger models:
    #   python test_blitzinfer_queue.py "Qwen/Qwen2.5-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"
    if len(sys.argv) >= 3:
        model_a = sys.argv[1]
        model_b = sys.argv[2]

    success = asyncio.run(run_queue_test(model_a, model_b))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
