#!/usr/bin/env python3
"""Quick smoke test for vLLM on ROCm."""

import time
import torch
from vllm import LLM, SamplingParams

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/HIP available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    start = time.time()

    # Start with enforce_eager=True to avoid CUDA graph issues
    # Use V0 engine for better stability
    import os
    os.environ["VLLM_USE_V1"] = "0"

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="float16"
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Use greedy sampling (no randomness, no sorting needed)
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    prompts = ["Hello, my name is"]

    print("Running inference...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - start

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    print(f"Inference time: {infer_time:.2f}s")
    print(f"Tokens: {len(outputs[0].outputs[0].token_ids)}")

if __name__ == "__main__":
    main()
