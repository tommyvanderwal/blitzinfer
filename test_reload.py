#!/usr/bin/env python3
"""Test model reload (same model) with BlitzInfer."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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

    # Single model for reload testing
    config = BlitzInferConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-7B-Instruct",
                alias="qwen-7b",
                gpu_memory_utilization=0.20,
                max_model_len=4096,
                max_num_seqs=20,
                max_num_batched_tokens=512,
                enforce_eager=True,
            ),
        ]
    )

    print("=" * 60)
    print("BlitzInfer Model Reload Test")
    print("=" * 60)

    orchestrator = BlitzInferOrchestrator(config)

    try:
        # Test 1: First load
        print("\n[Test 1] First load...")
        start = time.time()
        result1 = await orchestrator.generate(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt="Say hello.",
            max_tokens=10,
            temperature=0.0,
        )
        print(f"Response: {result1.text}")
        print(f"Time: {time.time() - start:.2f}s")

        # Test 2: Manual unload
        print("\n[Test 2] Unloading model...")
        start = time.time()
        orchestrator.engine.unload_model()
        print(f"Unload time: {time.time() - start:.2f}s")

        # Test 3: Reload same model
        print("\n[Test 3] Reloading same model...")
        start = time.time()
        result2 = await orchestrator.generate(
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt="Say goodbye.",
            max_tokens=10,
            temperature=0.0,
        )
        print(f"Response: {result2.text}")
        print(f"Time: {time.time() - start:.2f}s (includes reload)")

        print("\n" + "=" * 60)
        print("RELOAD TEST PASSED")
        print("=" * 60)

    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
