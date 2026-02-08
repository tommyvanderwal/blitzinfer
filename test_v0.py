#!/usr/bin/env python3
"""Test vLLM V0 engine with a 7B model on gfx1100."""

import os
# Force V0 engine BEFORE any vLLM imports
os.environ["VLLM_USE_V1"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "1"

import time
import torch


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen2.5-7B-Instruct"  # ~14GB in fp16, >6GB

    print(f"\nLoading {model_name}...")
    start = time.time()

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="float16",
        max_model_len=2048,  # Keep short for faster testing
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Quick inference test
    print("\nRunning inference...")
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0, max_tokens=30)
    )

    print(f"Generated: {outputs[0].outputs[0].text!r}")
    print("\nSUCCESS!")


if __name__ == "__main__":
    main()
