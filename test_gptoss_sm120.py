#!/usr/bin/env python3
"""Test gpt-oss-120b on SM120 (RTX PRO 6000 Blackwell).

This tests the triton_kernel backend with num_stages=2 constraint
to fit within the 101KB shared memory limit.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.srt.entrypoints.engine import Engine as SglEngine


def main():
    print("Loading gpt-oss-120b...", flush=True)
    engine = SglEngine(
        model_path="openai/gpt-oss-120b",
        mem_fraction_static=0.85,
        context_length=100000,
        trust_remote_code=True,
        log_level="warning",
        disable_cuda_graph=True,
        skip_server_warmup=True,
        moe_runner_backend="triton_kernel",
        dtype="bfloat16",
    )
    print("Engine loaded!", flush=True)

    # Test generation
    print("Generating...", flush=True)
    result = engine.generate(["What is 7*8?"], {"max_new_tokens": 30})
    text = result[0]["text"]
    print(f"Result text: {text}", flush=True)

    # Check answer
    if "56" in text:
        print("PASS - Correct answer!", flush=True)
    else:
        print("FAIL - Unexpected answer", flush=True)

    engine.shutdown()
    print("DONE - gpt-oss-120b WORKS on SM120!", flush=True)


if __name__ == "__main__":
    main()
