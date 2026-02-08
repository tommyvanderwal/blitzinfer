#!/usr/bin/env python3
"""Test persistent engine for faster model switching."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# IMPORTANT: Don't import torch or anything CUDA-related here
# to avoid CUDA initialization before spawning the worker

import time
import logging
import sys

# Add blitzinfer to path
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    from blitzinfer.engine.persistent_engine import PersistentEngine

    print("=" * 60)
    print("Persistent Engine Test")
    print("=" * 60)

    engine = PersistentEngine(
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
    )

    try:
        # Test 1: Start engine (includes vLLM import)
        print("\n[Phase 1] Starting persistent engine...")
        start = time.time()
        engine.start()
        print(f"Engine started in {time.time() - start:.2f}s")

        # Wait a bit for the process to be fully ready
        time.sleep(1)

        # Test 2: First model load
        print("\n[Phase 2] First model load (Qwen-7B)...")
        start = time.time()
        load_time = engine.load_model('Qwen/Qwen2.5-7B-Instruct')
        print(f"Model loaded in {load_time:.2f}s (total: {time.time() - start:.2f}s)")

        # Test 3: Generate
        print("\n[Phase 3] Generate text...")
        start = time.time()
        results = engine.generate(["What is 2+2?"], max_tokens=10, temperature=0)
        print(f"Generated in {time.time() - start:.2f}s: {results[0]}")

        # Test 4: Unload
        print("\n[Phase 4] Unload model...")
        start = time.time()
        engine.unload()
        print(f"Unloaded in {time.time() - start:.2f}s")

        # Test 5: Second model load (same model - should be faster)
        print("\n[Phase 5] Second model load (Qwen-7B again)...")
        start = time.time()
        load_time = engine.load_model('Qwen/Qwen2.5-7B-Instruct')
        print(f"Model loaded in {load_time:.2f}s (total: {time.time() - start:.2f}s)")

        # Test 6: Generate
        print("\n[Phase 6] Generate text...")
        start = time.time()
        results = engine.generate(["What is 3+3?"], max_tokens=10, temperature=0)
        print(f"Generated in {time.time() - start:.2f}s: {results[0]}")

        # Test 7: Switch to different model
        print("\n[Phase 7] Switch to Mistral-7B...")
        start = time.time()
        load_time = engine.load_model('mistralai/Mistral-7B-Instruct-v0.3')
        print(f"Model loaded in {load_time:.2f}s (total: {time.time() - start:.2f}s)")

        # Test 8: Generate
        print("\n[Phase 8] Generate text...")
        start = time.time()
        results = engine.generate(["What is 4+4?"], max_tokens=10, temperature=0)
        print(f"Generated in {time.time() - start:.2f}s: {results[0]}")

        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)

    finally:
        print("\nShutting down engine...")
        engine.shutdown()


if __name__ == '__main__':
    main()
