#!/usr/bin/env python3
"""
Integration test: BlitzInferOrchestrator with PageCacheWarmer.

This test validates the full orchestrator stack with page cache warming:
1. Orchestrator correctly initializes with PageCacheWarmer
2. Model switching works with proper cleanup
3. Page cache warming speeds up subsequent loads
4. Inference works correctly after each switch
5. Memory is properly managed across switches

Target: RTX PRO 6000 with GPT-OSS-120B and Qwen3-VL-32B-FP8
"""

import os
import gc
import time
import asyncio
import subprocess
import random
import string
import logging

# Configure for single-process mode
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models to test
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Thinking-FP8"

# Test configuration
NUM_SWITCHES = 10
REQUESTS_PER_MODEL = 2
MAX_TOKENS = 50


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def generate_random_prompt() -> str:
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
             for _ in range(30)]
    return f"Explain the following terms briefly: {' '.join(words)}"


async def run_test():
    print("=" * 80)
    print("BLITZINFER ORCHESTRATOR INTEGRATION TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Switches: {NUM_SWITCHES}")
    print(f"Requests per model: {REQUESTS_PER_MODEL}")

    # Import here after env setup
    from blitzinfer.config import BlitzInferConfig, ModelConfig, PrefetchConfig
    from blitzinfer.orchestrator import BlitzInferOrchestrator

    # Configure for RTX PRO 6000
    config = BlitzInferConfig(
        models=[
            ModelConfig(
                name=MODEL_A,
                dtype="bfloat16",
                max_model_len=32768,
                gpu_memory_utilization=0.90,
                max_num_seqs=8,
                max_num_batched_tokens=4096,
                enforce_eager=True,
                trust_remote_code=True,
                kv_cache_memory_bytes=None,  # Let vLLM profile
            ),
            ModelConfig(
                name=MODEL_B,
                dtype="bfloat16",
                max_model_len=32768,
                gpu_memory_utilization=0.90,
                max_num_seqs=8,
                max_num_batched_tokens=4096,
                enforce_eager=True,
                trust_remote_code=True,
                kv_cache_memory_bytes=None,
            ),
        ],
        prefetch=PrefetchConfig(
            enabled=True,
            use_page_cache=True,  # Use page cache warming
            trigger_on_queue_entry=True,
            chunk_size_mb=64,
        ),
    )

    print("\nInitializing orchestrator...")
    orch = BlitzInferOrchestrator(config)

    # Register model paths (resolve from HF cache)
    from huggingface_hub import snapshot_download
    path_a = snapshot_download(MODEL_A, local_files_only=True)
    path_b = snapshot_download(MODEL_B, local_files_only=True)
    orch.register_model_path(MODEL_A, path_a)
    orch.register_model_path(MODEL_B, path_b)

    print(f"  Model A path: {path_a}")
    print(f"  Model B path: {path_b}")

    # Results tracking
    results = []
    models = [MODEL_A, MODEL_B]
    initial_free = nvidia_smi_free_gb()

    for i in range(NUM_SWITCHES):
        switch_num = i + 1
        next_model = models[i % 2]
        other_model = models[(i + 1) % 2]

        print(f"\n{'#' * 80}")
        print(f"# SWITCH {switch_num}/{NUM_SWITCHES}: {next_model.split('/')[-1]}")
        print(f"{'#' * 80}")

        try:
            # Check warm status before switch
            warm_status_before = orch.get_warm_status(next_model)
            is_warm = orch.is_model_warm(next_model)
            print(f"\n  Warm status: {warm_status_before}")

            # Generate requests (this triggers model switch)
            print(f"\n  Running {REQUESTS_PER_MODEL} inference requests...")
            switch_time = 0
            infer_times = []

            for j in range(REQUESTS_PER_MODEL):
                prompt = generate_random_prompt()
                t0 = time.time()
                result = await orch.generate(
                    model=next_model,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
                req_time = time.time() - t0

                # First request includes switch time
                if j == 0:
                    switch_time = req_time
                else:
                    infer_times.append(req_time)

                response_preview = result.text[:50] + "..." if len(result.text) > 50 else result.text
                print(f"    Request {j+1}: {response_preview} ({req_time:.2f}s)")

            # Start warming the other model for next switch
            if not orch.is_model_warm(other_model):
                print(f"\n  Starting background warming for {other_model.split('/')[-1]}...")
                orch.start_warming(other_model)

            # Get status
            status = orch.get_status()
            gpu_free = nvidia_smi_free_gb()

            print(f"\n  GPU free: {gpu_free:.1f} GB")
            print(f"  Switch metrics: {switch_time:.2f}s total")

            if 'prefetch' in status and 'models' in status['prefetch']:
                for name, info in status['prefetch']['models'].items():
                    print(f"  Warming status: {name.split('/')[-1]}: {info['status']} ({info['progress']*100:.0f}%)")

            results.append({
                'switch': switch_num,
                'model': next_model.split('/')[-1],
                'was_warm': is_warm,
                'switch_time': switch_time,
                'avg_infer_time': sum(infer_times) / len(infer_times) if infer_times else 0,
                'gpu_free_after': gpu_free,
                'success': True,
            })

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'switch': switch_num,
                'model': next_model.split('/')[-1],
                'success': False,
                'error': str(e),
            })

    # Shutdown
    print("\n" + "=" * 80)
    print("SHUTTING DOWN")
    print("=" * 80)
    orch.shutdown()

    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    warm_loads = [r for r in successful if r['was_warm']]
    cold_loads = [r for r in successful if not r['was_warm']]

    print(f"\nTotal switches: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if cold_loads:
        avg_cold = sum(r['switch_time'] for r in cold_loads) / len(cold_loads)
        print(f"\nCold loads: {len(cold_loads)}")
        print(f"  Average switch time: {avg_cold:.2f}s")

    if warm_loads:
        avg_warm = sum(r['switch_time'] for r in warm_loads) / len(warm_loads)
        print(f"\nWarm loads: {len(warm_loads)}")
        print(f"  Average switch time: {avg_warm:.2f}s")

        if cold_loads:
            speedup = avg_cold / avg_warm
            saved = avg_cold - avg_warm
            print(f"  Speedup vs cold: {speedup:.2f}x")
            print(f"  Time saved per switch: {saved:.1f}s")

    # Memory check
    final_free = nvidia_smi_free_gb()
    memory_diff = initial_free - final_free

    print(f"\nMemory check:")
    print(f"  Initial free: {initial_free:.1f} GB")
    print(f"  Final free: {final_free:.1f} GB")
    print(f"  Difference: {memory_diff:+.1f} GB")

    if abs(memory_diff) < 3.0:
        print("  -> Memory stable")
    else:
        print(f"  -> WARNING: Memory drift of {memory_diff:.1f} GB")

    # Per-switch details
    print("\nPer-switch details:")
    for r in successful:
        warm_str = "[WARM]" if r['was_warm'] else "[COLD]"
        print(f"  Switch {r['switch']:2d} ({r['model'][:12]:12s}): "
              f"{r['switch_time']:.1f}s {warm_str}")

    if failed:
        print("\nFailed switches:")
        for r in failed:
            print(f"  Switch {r['switch']}: {r.get('error', 'Unknown')}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return len(failed) == 0


if __name__ == "__main__":
    success = asyncio.run(run_test())
    exit(0 if success else 1)
