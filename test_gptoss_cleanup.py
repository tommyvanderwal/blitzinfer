#!/usr/bin/env python3
"""Quick test: load gpt-oss-120b, free it, check GPU memory."""

import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import torch

def mem():
    ga = torch.cuda.memory_allocated() / 1024**3
    gr = torch.cuda.memory_reserved() / 1024**3
    return f"alloc={ga:.1f}GB rsrv={gr:.1f}GB"

def main():
    from sglang.srt.entrypoints.engine import Engine

    print(f"Start: {mem()}")

    # Load gpt-oss with triton_kernel + bf16
    engine = Engine(
        model_path="openai/gpt-oss-120b",
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
        moe_runner_backend="triton_kernel",
        dtype="bfloat16",
    )
    print(f"After load: {mem()}")

    # Generate to verify it works
    result = engine.generate(
        prompt="What is 2+3? Answer with just the number.",
        sampling_params={"temperature": 0, "max_new_tokens": 50},
    )
    text = result.get("text", "")
    print(f"Generate: {text[:60]!r}")

    # Now try to reload to qwen2.5-7b (which requires freeing gpt-oss)
    print(f"\nBefore reload: {mem()}")
    success, msg = engine.reload_model(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        server_args_overrides={},
    )
    print(f"After reload: success={success}, {mem()}")
    if not success:
        print(f"Error: {msg[:200]}")

    if success:
        result = engine.generate(
            prompt="What is 2+3? Answer with just the number.",
            sampling_params={"temperature": 0, "max_new_tokens": 50},
        )
        text = result.get("text", "")
        print(f"Qwen generate: {text[:60]!r}")

    engine.shutdown()
    print("Done")

if __name__ == "__main__":
    main()
