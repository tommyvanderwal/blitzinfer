#!/usr/bin/env python3
"""Basic vLLM test on 780M with ROCm 7.2."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"\nLoading {model_name}...")
start = time.time()

llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.15,  # Conservative for iGPU
    enforce_eager=True,
    dtype="float16",
    max_model_len=2048,
    max_num_seqs=4,
)

load_time = time.time() - start
print(f"Model loaded in {load_time:.2f}s")

# Test inference
print("\nRunning inference...")
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0, max_tokens=50)
)

print(f"Generated: {outputs[0].outputs[0].text!r}")
print("\nSUCCESS!")
