#!/usr/bin/env python3
"""Test model switching with BlitzInfer."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    from blitzinfer.config import ModelConfig, BlitzInferConfig
    from blitzinfer.orchestrator import BlitzInferOrchestrator

    # Configure two models for testing
    # Settings optimized for ROCm gfx1100 (Radeon 780M iGPU)
    config = BlitzInferConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-7B-Instruct",
                alias="qwen-7b",
                dtype="float16",
                gpu_memory_utilization=0.25,
                max_model_len=512,  # Small for fast switching
                max_num_seqs=4,
                max_num_batched_tokens=512,
                enforce_eager=True,
                kv_cache_memory_bytes=2 * 1024**3,  # 2GB fixed - skip memory profiling
            ),
            ModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.3",
                alias="mistral-7b",
                dtype="float16",
                gpu_memory_utilization=0.25,
                max_model_len=512,  # Small for fast switching
                max_num_seqs=4,
                max_num_batched_tokens=512,
                enforce_eager=True,
                kv_cache_memory_bytes=2 * 1024**3,  # 2GB fixed - skip memory profiling
            ),
        ]
    )

    print("=" * 60)
    print("BlitzInfer Model Switching Test")
    print("=" * 60)

    orchestrator = BlitzInferOrchestrator(config)

    try:
        # Test 1: First model load
        print("\n[Test 1] Loading first model (Qwen-7B)...")
        start = time.time()
        result1 = await orchestrator.generate(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt="What is the capital of France? Answer briefly.",
            max_tokens=50,
            temperature=0.0,
        )
        elapsed1 = time.time() - start
        print(f"Response: {result1.text}")
        print(f"Time: {elapsed1:.2f}s (includes first load)")

        # Test 2: Same model, should be fast
        print("\n[Test 2] Second request to same model...")
        start = time.time()
        result2 = await orchestrator.generate(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt="What is 2 + 2?",
            max_tokens=20,
            temperature=0.0,
        )
        elapsed2 = time.time() - start
        print(f"Response: {result2.text}")
        print(f"Time: {elapsed2:.2f}s (no switch)")

        # Test 3: Switch to different model
        print("\n[Test 3] Switching to second model (Mistral-7B)...")
        start = time.time()
        result3 = await orchestrator.generate(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt="What is the largest planet in our solar system? Answer briefly.",
            max_tokens=50,
            temperature=0.0,
        )
        elapsed3 = time.time() - start
        print(f"Response: {result3.text}")
        print(f"Time: {elapsed3:.2f}s (includes switch)")

        # Test 4: Second request to new model
        print("\n[Test 4] Second request to Mistral...")
        start = time.time()
        result4 = await orchestrator.generate(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt="What is 3 + 3?",
            max_tokens=20,
            temperature=0.0,
        )
        elapsed4 = time.time() - start
        print(f"Response: {result4.text}")
        print(f"Time: {elapsed4:.2f}s (no switch)")

        # Test 5: Switch back to first model
        print("\n[Test 5] Switching back to Qwen-7B...")
        start = time.time()
        result5 = await orchestrator.generate(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt="Name a color.",
            max_tokens=10,
            temperature=0.0,
        )
        elapsed5 = time.time() - start
        print(f"Response: {result5.text}")
        print(f"Time: {elapsed5:.2f}s (includes switch)")

        # Summary
        print("\n" + "=" * 60)
        print("SWITCH METRICS SUMMARY")
        print("=" * 60)

        status = orchestrator.get_status()
        print(f"Active model: {status['active_model']}")
        print(f"Total switches: {status['switch_count']}")

        for metrics in orchestrator.get_switch_metrics():
            print(f"\nSwitch: {metrics.from_model} -> {metrics.to_model}")
            print(f"  Unload time: {metrics.unload_time:.2f}s")
            print(f"  Load time:   {metrics.load_time:.2f}s")
            print(f"  Total time:  {metrics.total_time:.2f}s")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
