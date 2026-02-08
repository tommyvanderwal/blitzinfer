#!/usr/bin/env python3
"""Benchmark vLLM on Radeon 780M (gfx1100)."""

import os
import time
import torch

# REQUIRED for gfx1100 (valid values: 0 or 1)
os.environ["AMD_SERIALIZE_KERNEL"] = "1"

from vllm import LLM, SamplingParams

def benchmark_model(model_name: str, prompts: list[str], max_tokens: int = 100):
    """Benchmark a model with timing measurements."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # System info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model with timing
    print(f"\n[1] Loading model...")
    torch.cuda.synchronize()
    load_start = time.perf_counter()

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="float16",
        max_model_len=4096  # Limit context for faster warmup
    )

    torch.cuda.synchronize()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Memory after load
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory allocated: {mem_allocated:.2f} GB")
    print(f"GPU memory reserved: {mem_reserved:.2f} GB")

    # Warmup run
    print(f"\n[2] Warmup run...")
    warmup_params = SamplingParams(temperature=0, max_tokens=10)
    warmup_start = time.perf_counter()
    _ = llm.generate(["Hello"], warmup_params)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup done in {warmup_time:.2f}s")

    # Benchmark inference
    print(f"\n[3] Benchmark inference ({len(prompts)} prompts, {max_tokens} tokens each)...")
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    torch.cuda.synchronize()
    infer_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    infer_time = time.perf_counter() - infer_start

    # Calculate stats
    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    print(f"\nResults:")
    print(f"  Total time: {infer_time:.2f}s")
    print(f"  Input tokens: {total_input_tokens}")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Throughput: {total_output_tokens / infer_time:.2f} tok/s")
    print(f"  Time per token: {infer_time / total_output_tokens * 1000:.2f} ms")

    # Print sample output
    print(f"\nSample output:")
    print(f"  Prompt: {outputs[0].prompt[:50]!r}...")
    print(f"  Generated: {outputs[0].outputs[0].text[:100]!r}...")

    return {
        "model": model_name,
        "load_time": load_time,
        "warmup_time": warmup_time,
        "inference_time": infer_time,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "throughput": total_output_tokens / infer_time,
        "mem_allocated_gb": mem_allocated,
        "mem_reserved_gb": mem_reserved,
    }


def main():
    print("vLLM Benchmark on Radeon 780M (gfx1100)")
    print("AMD_SERIALIZE_KERNEL=3 enabled")

    # Test prompts
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about the ocean.",
        "What are the main differences between Python and JavaScript?",
        "Describe how a computer processor works.",
    ]

    results = []

    # Benchmark smaller model first
    try:
        result = benchmark_model("Qwen/Qwen2.5-0.5B-Instruct", prompts, max_tokens=100)
        results.append(result)
    except Exception as e:
        print(f"Error with Qwen2.5-0.5B: {e}")

    # Benchmark larger model (>5GB)
    try:
        result = benchmark_model("Qwen/Qwen2.5-3B-Instruct", prompts, max_tokens=100)
        results.append(result)
    except Exception as e:
        print(f"Error with Qwen2.5-3B: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Load time: {r['load_time']:.2f}s")
        print(f"  Memory: {r['mem_allocated_gb']:.2f} GB")
        print(f"  Throughput: {r['throughput']:.2f} tok/s")


if __name__ == "__main__":
    main()
