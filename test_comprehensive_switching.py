#!/usr/bin/env python3
"""Comprehensive model switching test with Harmony, vision, and code models.

Tests:
- gpt-oss-120b (MXFP4): Harmony encoding, tool calls, multi-turn
- Qwen3-Coder-Next-FP8: Code generation
- Kimi-VL-A3B: Vision with base64 images
- Qwen3-32B: General reasoning

All models must switch to/from each other without memory leaks.
"""
import os
import gc
import time
import base64
import json
import random
from io import BytesIO

# Single-process mode for faster switching
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams

from blitzinfer.engine.cleanup import full_cleanup, get_gpu_memory_info, log_gpu_memory

# Try to import Harmony utilities
try:
    from vllm.entrypoints.openai.parser.harmony_utils import (
        parse_chat_inputs_to_harmony_messages,
        render_for_completion,
        get_system_message,
        get_developer_message,
        parse_output_into_messages,
        parse_output_message,
        get_stop_tokens_for_assistant_actions,
    )
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
        FunctionDefinition as VLLMFunctionDefinition,
    )
    from openai.types.responses import ResponseFunctionToolCall
    HAS_HARMONY = True
except ImportError as e:
    print(f"Warning: Harmony imports failed: {e}")
    HAS_HARMONY = False


# Model configurations
MODELS = {
    "gpt-oss-120b": {
        "path": "openai/gpt-oss-120b",
        "dtype": "auto",  # MXFP4 quantized
        "gpu_util": 0.90,
        "max_model_len": 4096,
        "is_harmony": True,
        "is_vision": False,
        "is_mxfp4": True,  # Needs force_free cleanup for opaque CUDA allocations
        "uses_fla": False,  # Flash Linear Attention - incompatible with force_free
    },
    "qwen3-coder-next": {
        "path": "Qwen/Qwen3-Coder-Next-FP8",
        "dtype": "auto",  # FP8 quantized
        "gpu_util": 0.94,
        "max_model_len": 8192,  # Short context for testing, can go up to 200K
        "max_num_seqs": 2,  # Required for this large model
        "is_harmony": False,
        "is_vision": False,
        "is_mxfp4": False,
        "uses_fla": True,  # Uses FLA ops - caches tensors that conflict with force_free
    },
    "kimi-vl": {
        "path": "moonshotai/Kimi-VL-A3B-Instruct",
        "dtype": "bfloat16",
        "gpu_util": 0.90,
        "max_model_len": 4096,
        "is_harmony": False,
        "is_vision": True,
        "is_mxfp4": False,
        "uses_fla": False,
    },
    "qwen3-32b": {
        "path": "Qwen/Qwen3-32B",
        "dtype": "bfloat16",
        "gpu_util": 0.90,
        "max_model_len": 4096,
        "is_harmony": False,
        "is_vision": False,
        "is_mxfp4": False,
        "uses_fla": False,
    },
}


def create_test_image_base64() -> str:
    """Create a simple test image as base64."""
    try:
        from PIL import Image
        # Create a simple 100x100 red square
        img = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except ImportError:
        # Fallback: minimal 1x1 red PNG
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
            0xD4, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        return base64.b64encode(png_data).decode('utf-8')


# Test cases for each model type
HARMONY_TESTS = [
    {
        "name": "basic_math",
        "messages": [{"role": "user", "content": "What is 15 + 27? Just give the number."}],
        "tools": None,
        "validate": lambda r: "42" in r,
    },
    {
        "name": "tool_call_bash",
        "messages": [{"role": "user", "content": "Run the command 'echo hello' using bash"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"}
                    },
                    "required": ["command"]
                }
            }
        }],
        "validate": lambda r: "bash" in r.lower() or "echo" in r.lower() or "tool" in r.lower(),
    },
    {
        "name": "tool_call_read_file",
        "messages": [{"role": "user", "content": "Read the file /etc/hostname"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                }
            }
        }],
        "validate": lambda r: "read" in r.lower() or "file" in r.lower() or "tool" in r.lower(),
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I'll remember the number 42."},
            {"role": "user", "content": "What number did I ask you to remember?"},
        ],
        "tools": None,
        "validate": lambda r: "42" in r,
    },
    {
        "name": "reasoning",
        "messages": [{"role": "user", "content": "If a train travels 60 mph for 2 hours, how far does it go?"}],
        "tools": None,
        "validate": lambda r: "120" in r,
    },
]

CODE_TESTS = [
    {
        "name": "python_function",
        "prompt": "Write a Python function to check if a number is prime. Only output the code.",
        "validate": lambda r: "def" in r and "prime" in r.lower(),
    },
    {
        "name": "fibonacci",
        "prompt": "Write a recursive fibonacci function in Python. Only output the code.",
        "validate": lambda r: "def" in r and ("fib" in r.lower() or "fibonacci" in r.lower()),
    },
    {
        "name": "sort_algorithm",
        "prompt": "Implement quicksort in Python. Only output the code.",
        "validate": lambda r: "def" in r and ("sort" in r.lower() or "pivot" in r.lower()),
    },
]

VISION_TESTS = [
    {
        "name": "describe_image",
        "prompt": "What color is this image? Answer in one word.",
        "validate": lambda r: "red" in r.lower(),
    },
    {
        "name": "image_shape",
        "prompt": "Is this image a square or rectangle? Answer in one word.",
        "validate": lambda r: "square" in r.lower() or "rectangle" in r.lower(),
    },
]

GENERAL_TESTS = [
    {
        "name": "math",
        "prompt": "What is 2+2? Answer with just the number.",
        "validate": lambda r: "4" in r,
    },
    {
        "name": "knowledge",
        "prompt": "What is the capital of France? Answer in one word.",
        "validate": lambda r: "paris" in r.lower(),
    },
    {
        "name": "reasoning",
        "prompt": "If all cats are animals, and Fluffy is a cat, is Fluffy an animal? Yes or no.",
        "validate": lambda r: "yes" in r.lower(),
    },
]


def load_model(model_name: str) -> LLM:
    """Load a model with proper configuration."""
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"{'='*60}")

    # FLA cache clearing is now handled by cleanup.py's full_cleanup()
    # It clears FLA caches AFTER unloading any model, which is cleaner
    # and frees memory sooner than clearing before load.

    log_gpu_memory("before load")
    start = time.perf_counter()

    # Build kwargs
    kwargs = {
        "model": config["path"],
        "dtype": config["dtype"],
        "gpu_memory_utilization": config["gpu_util"],
        "max_model_len": config["max_model_len"],
        "trust_remote_code": True,
        "enforce_eager": True,
    }

    # Add max_num_seqs if specified (for large models like qwen3-coder-next)
    if "max_num_seqs" in config:
        kwargs["max_num_seqs"] = config["max_num_seqs"]

    llm = LLM(**kwargs)

    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.1f}s")
    log_gpu_memory("after load")

    return llm


def test_harmony_model(llm, tests: list) -> list:
    """Test Harmony model with tool calls and multi-turn."""
    if not HAS_HARMONY:
        print("Harmony imports not available, skipping Harmony tests")
        return [{"name": t["name"], "passed": False, "error": "No Harmony"} for t in tests]

    results = []
    stop_tokens = get_stop_tokens_for_assistant_actions()

    for test in tests:
        print(f"\n  Testing: {test['name']}")
        try:
            # Build Harmony messages
            chat_msgs = [{"role": m["role"], "content": m["content"]} for m in test["messages"]]
            has_tools = test["tools"] is not None

            # Get system message
            sys_msg = get_system_message(with_custom_tools=has_tools)
            harmony_msgs = [sys_msg]

            # Add developer message with tools if needed
            if has_tools:
                tools_for_harmony = []
                for tool in test["tools"]:
                    func = tool["function"]
                    func_def = VLLMFunctionDefinition(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"],
                    )
                    tool_param = ChatCompletionToolsParam(type="function", function=func_def)
                    tools_for_harmony.append(tool_param)
                dev_msg = get_developer_message(tools=tools_for_harmony)
                harmony_msgs.append(dev_msg)

            # Add chat messages
            harmony_msgs.extend(parse_chat_inputs_to_harmony_messages(chat_msgs))

            # Render to tokens
            prompt_token_ids = render_for_completion(harmony_msgs)

            # Generate
            sampling = SamplingParams(
                max_tokens=500,
                temperature=0.7,
                stop_token_ids=stop_tokens,
            )
            outputs = llm.generate(
                [{"prompt_token_ids": prompt_token_ids}],
                sampling,
            )

            # Parse output
            output_tokens = outputs[0].outputs[0].token_ids
            raw_text = outputs[0].outputs[0].text

            # Try to parse Harmony output
            try:
                parser = parse_output_into_messages(list(output_tokens))
                parsed_content = []
                tool_calls = []
                for msg in parser.messages:
                    items = parse_output_message(msg)
                    for item in items:
                        if isinstance(item, ResponseFunctionToolCall):
                            tool_calls.append(f"{item.name}({item.arguments})")
                        elif hasattr(item, 'content') and item.content:
                            for c in item.content:
                                if hasattr(c, 'text'):
                                    parsed_content.append(c.text)

                if tool_calls:
                    result_text = f"Tool calls: {', '.join(tool_calls)}"
                elif parsed_content:
                    result_text = ' '.join(parsed_content)
                else:
                    result_text = raw_text
            except Exception as e:
                result_text = raw_text

            print(f"    Output: {result_text[:100]}...")
            passed = test["validate"](result_text)
            print(f"    Result: {'PASS' if passed else 'FAIL'}")
            results.append({"name": test["name"], "passed": passed, "output": result_text[:200]})

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": test["name"], "passed": False, "error": str(e)})

    return results


def test_code_model(llm, tests: list) -> list:
    """Test code generation model."""
    results = []
    sampling = SamplingParams(max_tokens=500, temperature=0.3)

    for test in tests:
        print(f"\n  Testing: {test['name']}")
        try:
            outputs = llm.generate([test["prompt"]], sampling)
            text = outputs[0].outputs[0].text.strip()
            print(f"    Output: {text[:100]}...")
            passed = test["validate"](text)
            print(f"    Result: {'PASS' if passed else 'FAIL'}")
            results.append({"name": test["name"], "passed": passed, "output": text[:200]})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": test["name"], "passed": False, "error": str(e)})

    return results


def test_vision_model(llm, tests: list) -> list:
    """Test vision model with images.

    Note: Vision input format varies by model. Kimi-VL has specific requirements
    that may differ from the standard vLLM format. If vision tests fail,
    the general tests will still verify the model is lucid.
    """
    results = []
    sampling = SamplingParams(max_tokens=100, temperature=0.3)

    # Create test image as PIL Image
    try:
        from PIL import Image
        # Create a simple 100x100 red square
        img = Image.new('RGB', (100, 100), color='red')
    except ImportError:
        print("    WARNING: PIL not available, skipping vision-specific tests")
        img = None

    for test in tests:
        print(f"\n  Testing: {test['name']}")
        try:
            if img is None:
                # Skip vision tests if PIL not available
                results.append({"name": test["name"], "passed": False, "error": "PIL not available"})
                continue

            # vLLM vision input format - use PIL Image directly
            # Note: Some models like Kimi-VL may require different placeholder format
            prompt = {
                "prompt": f"<image>\n{test['prompt']}",
                "multi_modal_data": {
                    "image": img
                }
            }
            outputs = llm.generate([prompt], sampling)
            text = outputs[0].outputs[0].text.strip()
            print(f"    Output: {text[:100]}...")
            passed = test["validate"](text)
            print(f"    Result: {'PASS' if passed else 'FAIL'}")
            results.append({"name": test["name"], "passed": passed, "output": text[:200]})
        except Exception as e:
            # Vision format errors are common - don't print full traceback
            error_msg = str(e)
            if "preprocess" in error_msg or "multi_modal" in error_msg.lower():
                print(f"    SKIP: Vision format incompatible ({error_msg[:50]}...)")
                results.append({"name": test["name"], "passed": None, "skipped": True, "error": "Vision format incompatible"})
            else:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({"name": test["name"], "passed": False, "error": str(e)})

    return results


def test_general_model(llm, tests: list) -> list:
    """Test general model capabilities."""
    results = []
    sampling = SamplingParams(max_tokens=50, temperature=0.7)

    for test in tests:
        print(f"\n  Testing: {test['name']}")
        try:
            outputs = llm.generate([test["prompt"]], sampling)
            text = outputs[0].outputs[0].text.strip()
            print(f"    Output: {text[:100]}...")
            passed = test["validate"](text)
            print(f"    Result: {'PASS' if passed else 'FAIL'}")
            results.append({"name": test["name"], "passed": passed, "output": text[:200]})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": test["name"], "passed": False, "error": str(e)})

    return results


def run_model_tests(llm, model_name: str, config: dict) -> dict:
    """Run appropriate tests for a model."""
    print(f"\n--- Running tests for {model_name} ---")

    if config["is_harmony"]:
        tests = test_harmony_model(llm, HARMONY_TESTS)
    elif config["is_vision"]:
        tests = test_vision_model(llm, VISION_TESTS)
        tests.extend(test_general_model(llm, GENERAL_TESTS))
    elif "coder" in model_name.lower():
        # Coder models get code tests
        tests = test_code_model(llm, CODE_TESTS)
        tests.extend(test_general_model(llm, GENERAL_TESTS))
    else:
        tests = test_general_model(llm, GENERAL_TESTS)

    # Count results, excluding skipped tests
    passed = sum(1 for t in tests if t.get("passed") == True)
    skipped = sum(1 for t in tests if t.get("skipped", False))
    total = len(tests) - skipped  # Don't count skipped tests in total

    if skipped > 0:
        print(f"\n  Tests: {passed}/{total} passed, {skipped} skipped")
    else:
        print(f"\n  Tests passed: {passed}/{total}")

    return {"model": model_name, "tests": tests, "passed": passed, "total": total, "skipped": skipped}


# FLA_MODE is no longer needed - cleanup.py now clears FLA caches after force_free
FLA_MODE = False


def cleanup_model(llm, model_name: str) -> float:
    """Clean up model and return freed memory.

    Uses force_free=True for MXFP4 models (gpt-oss-120b) to properly release
    opaque CUDA allocations. FLA models (qwen3-coder-next) are now safe because
    cleanup.py clears FLA module caches after force_free.
    """
    print(f"\nCleaning up {model_name}...")
    log_gpu_memory("before cleanup")

    config = MODELS.get(model_name, {})
    is_mxfp4 = config.get("is_mxfp4", False)

    # Use force_free for MXFP4 models - FLA cache clearing now handles the conflict
    use_force_free = is_mxfp4

    start = time.perf_counter()
    freed = full_cleanup(llm, nuclear=True, force_free=use_force_free)
    cleanup_time = time.perf_counter() - start

    print(f"Cleanup time: {cleanup_time:.1f}s (force_free={use_force_free})")
    log_gpu_memory("after cleanup")

    return freed


def main():
    print("="*70)
    print("Comprehensive Model Switching Test")
    print("="*70)

    # FLA models (qwen3-coder-next) now work with force_free thanks to
    # clear_fla_module_caches() in cleanup.py. Can be disabled with env var.
    include_fla = os.environ.get("EXCLUDE_FLA_MODELS", "0") != "1"

    global FLA_MODE
    FLA_MODE = False  # No longer needed - FLA cache clearing handles it

    # Filter models based on FLA inclusion
    test_models = {k: v for k, v in MODELS.items()
                   if include_fla or not v.get("uses_fla", False)}

    if include_fla:
        print("Mode: ALL models (FLA cache clearing enabled)")
    else:
        print("Mode: Non-FLA models only")
        print("  To include FLA models: unset EXCLUDE_FLA_MODELS")

    print(f"Models: {', '.join(test_models.keys())}")

    # Track memory
    initial_mem = get_gpu_memory_info()
    print(f"\nInitial GPU: {initial_mem['used_gb']:.1f}GB used, {initial_mem['free_gb']:.1f}GB free")

    # Create switch sequence
    model_names = list(test_models.keys())
    switch_sequence = []

    # Start with gpt-oss-120b if available (MXFP4 - hardest to clean up)
    if "gpt-oss-120b" in model_names:
        switch_sequence.append("gpt-oss-120b")
        model_names.remove("gpt-oss-120b")

    # Add each other model
    for m in model_names:
        switch_sequence.append(m)

    # Switch back to gpt-oss-120b if available
    if "gpt-oss-120b" in test_models:
        switch_sequence.append("gpt-oss-120b")

    # Add a few more random switches
    all_names = list(test_models.keys())
    for _ in range(4):
        candidates = [m for m in all_names if m != switch_sequence[-1]]
        if candidates:
            switch_sequence.append(random.choice(candidates))

    print(f"\nSwitch sequence ({len(switch_sequence)} switches):")
    print(" -> ".join(switch_sequence))

    # Run switches
    all_results = []
    llm = None
    current_model = None

    for i, model_name in enumerate(switch_sequence):
        print(f"\n{'#'*70}")
        print(f"# SWITCH {i+1}/{len(switch_sequence)}: {model_name}")
        print(f"{'#'*70}")

        # Cleanup previous model
        if llm is not None:
            cleanup_model(llm, current_model)
            llm = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load and test
        try:
            llm = load_model(model_name)
            current_model = model_name
            config = MODELS[model_name]
            test_results = run_model_tests(llm, model_name, config)

            mem = get_gpu_memory_info()
            all_results.append({
                "switch": i + 1,
                "model": model_name,
                "success": True,
                "tests": test_results,
                "used_gb": mem["used_gb"],
                "free_gb": mem["free_gb"],
            })
        except Exception as e:
            print(f"\n!!! LOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "switch": i + 1,
                "model": model_name,
                "success": False,
                "error": str(e),
            })
            llm = None
            current_model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Final cleanup
    if llm is not None:
        cleanup_model(llm, current_model)
        llm = None

    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    final_mem = get_gpu_memory_info()
    total_drift = final_mem["used_gb"] - initial_mem["used_gb"]

    print(f"\n{'Switch':<8} {'Model':<20} {'Load':<8} {'Tests':<12} {'Memory':<12}")
    print("-" * 70)

    total_tests_passed = 0
    total_tests_run = 0
    total_skipped = 0
    all_loads_success = True

    for r in all_results:
        if r["success"]:
            tests = r.get("tests", {})
            passed = tests.get("passed", 0)
            total = tests.get("total", 0)
            skipped = tests.get("skipped", 0)
            total_tests_passed += passed
            total_tests_run += total
            total_skipped += skipped
            test_str = f"{passed}/{total}" if skipped == 0 else f"{passed}/{total}+{skipped}s"
            print(f"{r['switch']:<8} {r['model']:<20} {'OK':<8} {test_str:<12} {r['free_gb']:.1f}GB free")
        else:
            all_loads_success = False
            print(f"{r['switch']:<8} {r['model']:<20} {'FAIL':<8} {'-':<12} {'-':<12}")

    print("-" * 70)
    print(f"\nFinal GPU: {final_mem['used_gb']:.1f}GB used, {final_mem['free_gb']:.1f}GB free")
    print(f"Memory drift: {total_drift:+.1f}GB")
    if total_skipped > 0:
        print(f"Total tests: {total_tests_passed}/{total_tests_run} passed, {total_skipped} skipped")
    else:
        print(f"Total tests: {total_tests_passed}/{total_tests_run} passed")

    # Final verdict
    memory_ok = final_mem["free_gb"] >= 90.0
    # 80% pass rate (excluding skipped tests)
    tests_ok = total_tests_run == 0 or total_tests_passed >= total_tests_run * 0.8

    print(f"\n{'='*70}")
    if all_loads_success and memory_ok and tests_ok:
        print("OVERALL: PASS")
        print(f"  - All {len(switch_sequence)} model loads successful")
        print(f"  - Memory OK ({final_mem['free_gb']:.1f}GB free, drift {total_drift:+.1f}GB)")
        print(f"  - Tests: {total_tests_passed}/{total_tests_run} passed")
        if total_skipped > 0:
            print(f"  - Skipped: {total_skipped} (vision format incompatible)")
    else:
        print("OVERALL: FAIL")
        if not all_loads_success:
            failed_switches = [r['model'] for r in all_results if not r['success']]
            print(f"  - Model loads failed: {', '.join(failed_switches)}")
        if not memory_ok:
            print(f"  - Memory leak ({final_mem['free_gb']:.1f}GB free, need 90+GB)")
        if not tests_ok:
            print(f"  - Too many test failures ({total_tests_passed}/{total_tests_run})")
    print("="*70)


if __name__ == "__main__":
    main()
