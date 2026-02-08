#!/usr/bin/env python3
"""Quick diagnostic: reload to gpt-oss-120b, test plain text and Harmony encoding."""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import time
import torch

def main():
    from sglang.srt.entrypoints.engine import Engine

    # Start with a small model (fast)
    engine = Engine(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
    )
    print(f"Qwen loaded. GPU alloc={torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Reload to gpt-oss
    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path="openai/gpt-oss-120b",
        server_args_overrides={
            "context_length": 131072,
            "moe_runner_backend": "triton_kernel",
            "dtype": "bfloat16",
        },
    )
    print(f"Reload: success={success}, time={time.perf_counter()-t0:.1f}s")
    if not success:
        print(f"Error: {msg[:200]}")
        engine.shutdown()
        return

    # Test 1: Plain text prompt
    print("\n=== TEST 1: Plain text prompt ===")
    result = engine.generate(
        prompt="What is 2+3? Answer with just the number.",
        sampling_params={"temperature": 0, "max_new_tokens": 100},
    )
    text = result.get("text", "")
    meta = result.get("meta_info", {})
    output_ids = meta.get("output_token_ids", [])
    print(f"  text: {text[:120]!r}")
    print(f"  output_ids count: {len(output_ids) if output_ids else 0}")
    print(f"  finish_reason: {meta.get('finish_reason', 'N/A')}")
    has_5 = "5" in text
    print(f"  Contains '5': {has_5}")

    # Test 2: Harmony encoding
    print("\n=== TEST 2: Harmony encoding ===")
    try:
        from vllm.entrypoints.openai.parser.harmony_utils import (
            parse_chat_inputs_to_harmony_messages,
            render_for_completion,
            get_system_message,
            parse_chat_output,
            get_stop_tokens_for_assistant_actions,
        )

        chat_msgs = [{"role": "user", "content": "What is 2+3? Answer with just the number."}]
        sys_msg = get_system_message(with_custom_tools=False)
        harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)
        token_ids = render_for_completion(harmony_msgs)
        stop_tokens = get_stop_tokens_for_assistant_actions()

        print(f"  Input token count: {len(token_ids)}")
        print(f"  Stop tokens: {stop_tokens}")

        result = engine.generate(
            input_ids=token_ids,
            sampling_params={"temperature": 0, "max_new_tokens": 500, "stop_token_ids": stop_tokens},
        )
        text = result.get("text", "")
        meta = result.get("meta_info", {})
        output_ids = meta.get("output_token_ids", [])

        print(f"  Raw text: {text[:200]!r}")
        print(f"  output_ids count: {len(output_ids) if output_ids else 0}")
        print(f"  finish_reason: {meta.get('finish_reason', 'N/A')}")

        if output_ids:
            parsed = parse_chat_output(list(output_ids))
            print(f"  Harmony final_content: {parsed.final_content!r}")
            print(f"  Harmony reasoning: {parsed.reasoning[:100] if parsed.reasoning else 'None'!r}")
        else:
            print("  No output_ids available for Harmony parsing")

    except ImportError as e:
        print(f"  Harmony not available: {e}")

    engine.shutdown()
    print("\nDone")

if __name__ == "__main__":
    main()
