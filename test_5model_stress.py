#!/usr/bin/env python3
"""5-Model Stress Test: Maximum context, architecture diversity.

Models:
1. Qwen/Qwen3-VL-32B-Thinking-FP8 (VL, FP8)
2. openai/gpt-oss-120b (LLM, MXFP4)
3. moonshotai/Kimi-VL-A3B-Thinking-2506 (VL, MoE)
4. zai-org/GLM-4.6V-Flash (VL)
5. mistralai/Mistral-Small-3.2-24B-Instruct-2506 (LLM)

Test phases:
1. Load each model at least 2x (10 loads total), verify lucidity
2. Extended inference: 5+ requests per model
"""

import os
import gc
import sys
import time
import random
from datetime import datetime

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Model configurations - use native max context, high VRAM utilization
MODELS = {
    "qwen-32b": {
        "name": "Qwen/Qwen3-VL-32B-Thinking-FP8",
        "max_model_len": 131072,  # 128K native
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "gpt-oss-120b": {
        "name": "openai/gpt-oss-120b",
        "max_model_len": 131072,  # 128K
        "gpu_memory_utilization": 0.90,  # Needs more headroom due to size
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "kimi-vl": {
        "name": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "max_model_len": 131072,  # 128K (will adjust if different)
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "glm-4v": {
        "name": "zai-org/GLM-4.6V-Flash",
        "max_model_len": 131072,  # 128K (will adjust if different)
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "mistral-24b": {
        "name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "max_model_len": 131072,  # 128K native
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
}

# Simple questions to verify lucidity
LUCIDITY_QUESTIONS = [
    "What is 2 + 2? Answer with just the number.",
    "What color is the sky on a clear day? One word answer.",
    "What is the capital of France? One word answer.",
    "How many legs does a cat have? Answer with just the number.",
    "What planet do we live on? One word answer.",
]

# Extended inference prompts
INFERENCE_PROMPTS = [
    "Explain the concept of recursion in programming in 2-3 sentences.",
    "What are the three states of matter? List them briefly.",
    "Write a haiku about technology.",
    "What is the Pythagorean theorem? State it simply.",
    "Name three programming languages and their main use cases.",
]


def log(msg):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_memory():
    """Get GPU memory info."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return {
        'used_gb': (total - free) / 1024**3,
        'free_gb': free / 1024**3,
        'allocated_gb': allocated / 1024**3,
    }


def load_model(model_key):
    """Load a model by key."""
    from vllm import LLM

    config = MODELS[model_key]
    log(f"Loading {model_key}: {config['name']}")
    log(f"  max_model_len: {config['max_model_len']}, gpu_util: {config['gpu_memory_utilization']}")

    start = time.time()

    try:
        llm = LLM(
            model=config['name'],
            dtype=config['dtype'],
            max_model_len=config['max_model_len'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            enforce_eager=True,
            trust_remote_code=config['trust_remote_code'],
        )
        load_time = time.time() - start
        mem = get_memory()
        log(f"  Loaded in {load_time:.1f}s, VRAM: {mem['used_gb']:.1f}GB")
        return llm, load_time
    except Exception as e:
        log(f"  FAILED to load: {e}")
        return None, 0


def test_lucidity(llm, model_key):
    """Test if model is lucid with a simple question."""
    from vllm import SamplingParams

    question = random.choice(LUCIDITY_QUESTIONS)
    log(f"  Lucidity test: '{question}'")

    try:
        start = time.time()
        out = llm.generate([question], SamplingParams(max_tokens=20, temperature=0.1))
        gen_time = time.time() - start
        answer = out[0].outputs[0].text.strip()
        log(f"  Answer: '{answer}' ({gen_time:.2f}s)")

        # Basic sanity check
        if len(answer) > 0 and len(answer) < 100:
            return True, answer, gen_time
        else:
            log(f"  WARNING: Suspicious answer length")
            return False, answer, gen_time
    except Exception as e:
        log(f"  FAILED lucidity test: {e}")
        return False, str(e), 0


def cleanup_model(llm):
    """Clean up model and free memory."""
    from blitzinfer.engine.cleanup import full_cleanup

    start = time.time()
    freed = full_cleanup(llm, nuclear=True)
    cleanup_time = time.time() - start

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem = get_memory()
    log(f"  Cleanup: freed {freed:.1f}GB in {cleanup_time:.2f}s, VRAM now: {mem['used_gb']:.2f}GB")
    return cleanup_time


def run_inference_batch(llm, model_key, num_prompts=5):
    """Run multiple inference requests."""
    from vllm import SamplingParams

    log(f"  Running {num_prompts} inference requests...")
    results = []

    prompts = random.sample(INFERENCE_PROMPTS, min(num_prompts, len(INFERENCE_PROMPTS)))
    if num_prompts > len(INFERENCE_PROMPTS):
        prompts = prompts * (num_prompts // len(INFERENCE_PROMPTS) + 1)
        prompts = prompts[:num_prompts]

    for i, prompt in enumerate(prompts):
        try:
            start = time.time()
            out = llm.generate([prompt], SamplingParams(max_tokens=100, temperature=0.7))
            gen_time = time.time() - start
            answer = out[0].outputs[0].text.strip()[:100]  # Truncate for display
            results.append({'success': True, 'time': gen_time})
            log(f"    [{i+1}/{num_prompts}] {gen_time:.2f}s: {answer[:50]}...")
        except Exception as e:
            results.append({'success': False, 'time': 0, 'error': str(e)})
            log(f"    [{i+1}/{num_prompts}] FAILED: {e}")

    return results


def select_next_model(current_model, load_counts):
    """Select next model (not current, prefer least loaded)."""
    available = [k for k in MODELS.keys() if k != current_model]

    # Find models with minimum load count
    min_count = min(load_counts.get(k, 0) for k in available)
    candidates = [k for k in available if load_counts.get(k, 0) == min_count]

    return random.choice(candidates)


def phase1_load_test(num_loads=10):
    """Phase 1: Load each model at least 2x, verify lucidity."""
    log("=" * 70)
    log("PHASE 1: Model Loading Test (10 loads, each model at least 2x)")
    log("=" * 70)

    load_counts = {k: 0 for k in MODELS}
    results = []
    current_model = None
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    baseline = get_memory()
    log(f"Baseline VRAM: {baseline['used_gb']:.2f}GB")

    for i in range(num_loads):
        log(f"\n{'='*70}")
        log(f"LOAD {i+1}/{num_loads}")
        log("=" * 70)

        # Select next model
        next_model = select_next_model(current_model, load_counts)
        log(f"Switching: {current_model} -> {next_model}")

        # Cleanup current model if exists
        if llm is not None:
            cleanup_time = cleanup_model(llm)
            llm = None
        else:
            cleanup_time = 0

        # Load new model
        llm, load_time = load_model(next_model)

        if llm is None:
            log(f"SKIPPING {next_model} due to load failure")
            results.append({
                'load': i + 1,
                'model': next_model,
                'success': False,
                'error': 'Load failed',
            })
            continue

        load_counts[next_model] += 1
        current_model = next_model

        # Test lucidity
        lucid, answer, gen_time = test_lucidity(llm, next_model)

        mem = get_memory()
        drift = mem['used_gb'] - baseline['used_gb']

        results.append({
            'load': i + 1,
            'model': next_model,
            'success': True,
            'lucid': lucid,
            'load_time': load_time,
            'cleanup_time': cleanup_time,
            'gen_time': gen_time,
            'vram_used': mem['used_gb'],
            'drift': drift,
        })

        log(f"Result: {'PASS' if lucid else 'FAIL'}, Load: {load_time:.1f}s, VRAM: {mem['used_gb']:.1f}GB, Drift: {drift:+.2f}GB")

    # Final cleanup
    if llm is not None:
        cleanup_model(llm)
        llm = None

    # Summary
    log("\n" + "=" * 70)
    log("PHASE 1 SUMMARY")
    log("=" * 70)

    log(f"\nLoad counts: {load_counts}")

    successful = [r for r in results if r.get('success', False)]
    lucid_count = sum(1 for r in successful if r.get('lucid', False))

    log(f"Successful loads: {len(successful)}/{len(results)}")
    log(f"Lucid responses: {lucid_count}/{len(successful)}")

    if successful:
        avg_load = sum(r['load_time'] for r in successful) / len(successful)
        avg_cleanup = sum(r.get('cleanup_time', 0) for r in successful if r.get('cleanup_time', 0) > 0)
        cleanup_count = sum(1 for r in successful if r.get('cleanup_time', 0) > 0)
        if cleanup_count > 0:
            avg_cleanup /= cleanup_count
        log(f"Avg load time: {avg_load:.1f}s")
        log(f"Avg cleanup time: {avg_cleanup:.2f}s")

    final_mem = get_memory()
    log(f"Final VRAM: {final_mem['used_gb']:.2f}GB (drift: {final_mem['used_gb'] - baseline['used_gb']:+.2f}GB)")

    return results, load_counts


def phase2_inference_test(load_counts):
    """Phase 2: Extended inference on each model."""
    log("\n" + "=" * 70)
    log("PHASE 2: Extended Inference Test (5+ requests per model)")
    log("=" * 70)

    results = {}
    current_model = None
    llm = None

    # Order by least loaded first
    model_order = sorted(MODELS.keys(), key=lambda k: load_counts.get(k, 0))

    for model_key in model_order:
        log(f"\n{'='*70}")
        log(f"INFERENCE TEST: {model_key}")
        log("=" * 70)

        # Cleanup current if needed
        if llm is not None:
            cleanup_model(llm)
            llm = None

        # Load model
        llm, load_time = load_model(model_key)

        if llm is None:
            log(f"SKIPPING {model_key} due to load failure")
            results[model_key] = {'success': False, 'error': 'Load failed'}
            continue

        current_model = model_key

        # Run inference batch
        inference_results = run_inference_batch(llm, model_key, num_prompts=5)

        success_count = sum(1 for r in inference_results if r['success'])
        avg_time = sum(r['time'] for r in inference_results if r['success']) / max(success_count, 1)

        results[model_key] = {
            'success': True,
            'load_time': load_time,
            'inference_count': len(inference_results),
            'inference_success': success_count,
            'avg_inference_time': avg_time,
        }

        log(f"Result: {success_count}/{len(inference_results)} successful, avg time: {avg_time:.2f}s")

    # Final cleanup
    if llm is not None:
        cleanup_model(llm)

    # Summary
    log("\n" + "=" * 70)
    log("PHASE 2 SUMMARY")
    log("=" * 70)

    for model_key, result in results.items():
        if result.get('success'):
            log(f"  {model_key}: {result['inference_success']}/{result['inference_count']} requests, "
                f"avg {result['avg_inference_time']:.2f}s")
        else:
            log(f"  {model_key}: FAILED - {result.get('error', 'Unknown')}")

    return results


def main():
    log("=" * 70)
    log("5-MODEL STRESS TEST")
    log("=" * 70)
    log(f"Models: {list(MODELS.keys())}")
    log(f"Target: Max context, 90GB+ VRAM utilization")
    log("")

    # Phase 1: Load test
    phase1_results, load_counts = phase1_load_test(num_loads=10)

    # Check if all models loaded at least once
    failed_models = [k for k, v in load_counts.items() if v == 0]
    if failed_models:
        log(f"\nWARNING: These models never loaded successfully: {failed_models}")

    # Phase 2: Inference test
    phase2_results = phase2_inference_test(load_counts)

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    final_mem = get_memory()
    log(f"Final VRAM: {final_mem['used_gb']:.2f}GB")

    phase1_success = sum(1 for r in phase1_results if r.get('success') and r.get('lucid'))
    phase2_success = sum(1 for r in phase2_results.values() if r.get('success'))

    log(f"Phase 1: {phase1_success}/{len(phase1_results)} loads successful and lucid")
    log(f"Phase 2: {phase2_success}/{len(MODELS)} models completed inference")

    if phase1_success == len(phase1_results) and phase2_success == len(MODELS):
        log("\n*** ALL TESTS PASSED ***")
        return 0
    else:
        log("\n*** SOME TESTS FAILED ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
