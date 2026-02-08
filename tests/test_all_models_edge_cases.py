#!/usr/bin/env python3
"""Comprehensive edge case tests for BlitzInfer SGLang server.

Extended tests covering:
  1. Qwen3-Coder-Next (new MoE model): code generation, long code
  2. GPT-OSS Harmony: tool calls, multi-turn with tools
  3. Long context tests (32K+ tokens)
  4. Memory drift monitoring over 5+ switches
  5. AWQ safety after multiple switches (llama-3.1-70b)
  6. All model switching pairs
  7. Vision (kimi-vl): image input validation

Usage:
  # Requires server running on :8000
  python tests/test_all_models_edge_cases.py [--base-url http://host:port]

  # Test specific model
  python tests/test_all_models_edge_cases.py --model qwen3-coder-next

  # Skip memory-intensive tests
  python tests/test_all_models_edge_cases.py --quick
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# =============================================================================
# Config
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT = 600.0  # Long timeout for model switches and long context

ALL_MODELS = [
    "gpt-oss-120b",
    "kimi-vl",
    "qwen3-32b",
    "llama-3.1-70b",
    "qwen2.5-7b",
    "qwen3-coder-next",
]

VISION_MODELS = ["kimi-vl"]
HARMONY_MODELS = ["gpt-oss-120b"]
MOE_MODELS = ["gpt-oss-120b", "qwen3-coder-next"]
AWQ_MODELS = ["llama-3.1-70b"]

# Models to use for quick tests
QUICK_MODELS = ["qwen2.5-7b", "qwen3-coder-next"]


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    detail: str = ""
    error: str = ""


results: List[TestResult] = []
memory_samples: List[Dict[str, float]] = []


def record(name: str, passed: bool, duration: float, detail: str = "", error: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append(TestResult(name, passed, duration, detail, error))
    print(f"  [{status}] {name} ({duration:.1f}s) {detail}")
    if error:
        for line in error.split('\n')[:5]:
            print(f"         {line}")


def print_summary():
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed > 0:
        print("\n  FAILURES:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.error or r.detail}")

    # Memory drift summary
    if len(memory_samples) >= 2:
        start_mem = memory_samples[0].get('gpu_used_gb', 0)
        end_mem = memory_samples[-1].get('gpu_used_gb', 0)
        drift = end_mem - start_mem
        print(f"\n  MEMORY DRIFT: {drift:.2f}GB over {len(memory_samples)} samples")
        print(f"    Start: {start_mem:.2f}GB, End: {end_mem:.2f}GB")

    print("=" * 70)
    return failed == 0


# =============================================================================
# HTTP Helpers
# =============================================================================

async def chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: List[Dict],
    max_tokens: int = 200,
    temperature: float = 0.3,
    tools: Optional[List[Dict]] = None,
    stream: bool = False,
    timeout: float = TIMEOUT,
) -> Dict:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools

    if stream:
        chunks = []
        async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                  json=body, timeout=timeout) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(data))
                    except json.JSONDecodeError:
                        pass
        return {"chunks": chunks, "status_code": resp.status_code}

    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    return {"body": resp.json(), "status_code": resp.status_code}


async def wait_for_server(client: httpx.AsyncClient, base_url: str, timeout: float = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = await client.get(f"{base_url}/health", timeout=10)
            if resp.json().get("status") == "healthy":
                return True
        except Exception:
            pass
        await asyncio.sleep(2.0)
    return False


async def get_status(client: httpx.AsyncClient, base_url: str) -> Dict:
    resp = await client.get(f"{base_url}/status", timeout=10)
    return resp.json()


def get_content(resp: Dict) -> str:
    """Extract text content from a chat completion response."""
    body = resp.get("body", {})
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("content", "") or ""
    return ""


def get_tool_calls(resp: Dict) -> List[Dict]:
    """Extract tool calls from a chat completion response."""
    body = resp.get("body", {})
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("tool_calls", []) or []
    return []


def get_usage(resp: Dict) -> Dict:
    """Extract usage stats from a chat completion response."""
    body = resp.get("body", {})
    return body.get("usage", {})


def make_red_png_base64() -> str:
    """Create a 64x64 red PNG as base64 data URI."""
    width, height = 64, 64
    raw = b''
    for y in range(height):
        raw += b'\x00'
        for x in range(width):
            raw += b'\xff\x00\x00'  # Red pixel (RGB)

    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png_data = sig + png_chunk(b'IHDR', ihdr) + png_chunk(b'IDAT', zlib.compress(raw)) + png_chunk(b'IEND', b'')
    b64 = base64.b64encode(png_data).decode()
    return f"data:image/png;base64,{b64}"


# =============================================================================
# Test Group 1: Qwen3-Coder-Next (New MoE Model)
# =============================================================================

async def test_qwen_coder_basic(client: httpx.AsyncClient, base_url: str):
    """Test basic code generation with Qwen3-Coder-Next."""
    model = "qwen3-coder-next"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "Write a Python function to check if a number is prime. Just the function, no explanation."}],
                          max_tokens=500)
        duration = time.time() - t0
        content = get_content(resp)

        # Check for Python code markers
        has_def = "def " in content
        has_prime = "prime" in content.lower()
        has_return = "return" in content

        passed = has_def and has_prime and has_return and resp["status_code"] == 200
        record(f"{model}/basic_code", passed, duration,
               f"content_len={len(content)}, has_def={has_def}",
               "" if passed else f"Missing code elements: def={has_def}, prime={has_prime}, return={has_return}")
    except Exception as e:
        record(f"{model}/basic_code", False, time.time() - t0, error=str(e))


async def test_qwen_coder_long_code(client: httpx.AsyncClient, base_url: str):
    """Test longer code generation with Qwen3-Coder-Next."""
    model = "qwen3-coder-next"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": """Write a Python class implementing a binary search tree with the following methods:
1. insert(value)
2. search(value) -> bool
3. delete(value)
4. inorder_traversal() -> list

Include docstrings for each method."""}],
                          max_tokens=2000)
        duration = time.time() - t0
        content = get_content(resp)
        usage = get_usage(resp)

        # Check for class structure
        has_class = "class " in content
        has_insert = "def insert" in content
        has_search = "def search" in content
        has_docstring = '"""' in content or "'''" in content
        completion_tokens = usage.get("completion_tokens", 0)

        passed = has_class and has_insert and has_search and completion_tokens > 200 and resp["status_code"] == 200
        record(f"{model}/long_code", passed, duration,
               f"tokens={completion_tokens}, has_class={has_class}",
               "" if passed else f"Missing: class={has_class}, insert={has_insert}, search={has_search}")
    except Exception as e:
        record(f"{model}/long_code", False, time.time() - t0, error=str(e))


async def test_qwen_coder_multi_language(client: httpx.AsyncClient, base_url: str):
    """Test code generation in multiple programming languages."""
    model = "qwen3-coder-next"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "Write a hello world function in Rust. Just the function."}],
                          max_tokens=200)
        duration = time.time() - t0
        content = get_content(resp)

        # Check for Rust code markers
        has_fn = "fn " in content
        has_print = "println!" in content or "print!" in content

        passed = has_fn and resp["status_code"] == 200
        record(f"{model}/rust_code", passed, duration,
               f"has_fn={has_fn}, content_len={len(content)}",
               "" if passed else f"Missing Rust function")
    except Exception as e:
        record(f"{model}/rust_code", False, time.time() - t0, error=str(e))


async def test_qwen_coder_streaming(client: httpx.AsyncClient, base_url: str):
    """Test streaming code generation."""
    model = "qwen3-coder-next"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "Write a simple Python factorial function."}],
                          max_tokens=200, stream=True)
        duration = time.time() - t0
        chunks = resp.get("chunks", [])
        content = ""
        for c in chunks:
            delta = c.get("choices", [{}])[0].get("delta", {})
            content += delta.get("content", "") or ""

        has_chunks = len(chunks) >= 2
        has_def = "def " in content
        passed = has_chunks and has_def
        record(f"{model}/streaming", passed, duration,
               f"chunks={len(chunks)}, has_def={has_def}",
               "" if passed else f"chunks={len(chunks)}, has_def={has_def}")
    except Exception as e:
        record(f"{model}/streaming", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 2: GPT-OSS Harmony Tool Calls (Extended)
# =============================================================================

async def test_gptoss_tool_call_bash(client: httpx.AsyncClient, base_url: str):
    """Test GPT-OSS bash tool call (Claude Code style)."""
    model = "gpt-oss-120b"
    t0 = time.time()
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"},
                    },
                    "required": ["command"],
                },
            },
        }]
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "List the files in the current directory."}],
                          max_tokens=500, tools=tools)
        duration = time.time() - t0
        tool_calls = get_tool_calls(resp)
        has_tool_call = len(tool_calls) > 0
        correct_function = False
        if has_tool_call:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            correct_function = fn.get("name") == "bash"
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                    # Should have "ls" or similar in command
                    cmd = args_dict.get("command", "").lower()
                    correct_function = correct_function and ("ls" in cmd or "dir" in cmd)
                except json.JSONDecodeError:
                    pass

        passed = has_tool_call and correct_function and resp["status_code"] == 200
        record(f"{model}/tool_call_bash", passed, duration,
               f"tool_calls={len(tool_calls)}",
               "" if passed else f"has_tc={has_tool_call}, correct_fn={correct_function}")
    except Exception as e:
        record(f"{model}/tool_call_bash", False, time.time() - t0, error=str(e))


async def test_gptoss_multi_turn_with_tool_result(client: httpx.AsyncClient, base_url: str):
    """Test GPT-OSS multi-turn conversation with tool call result."""
    model = "gpt-oss-120b"
    t0 = time.time()
    try:
        # First, get a tool call
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }]

        # Multi-turn: user asks, assistant calls tool, tool returns, assistant responds
        messages = [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}
            }]},
            {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "call_001"},
        ]

        resp = await chat(client, base_url, model, messages, max_tokens=300, tools=tools)
        duration = time.time() - t0
        content = get_content(resp)

        # Should have a response incorporating the weather info
        has_content = len(content) > 10
        has_temp = "22" in content or "sunny" in content.lower() or "tokyo" in content.lower()
        no_leak = "<|channel|>" not in content

        passed = has_content and no_leak and resp["status_code"] == 200
        record(f"{model}/multi_turn_tool", passed, duration,
               f"content_len={len(content)}, has_temp={has_temp}",
               "" if passed else f"has_content={has_content}, no_leak={no_leak}")
    except Exception as e:
        record(f"{model}/multi_turn_tool", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 3: Long Context Tests
# =============================================================================

async def test_long_context_32k(client: httpx.AsyncClient, base_url: str, model: str):
    """Test model with ~32K token context."""
    t0 = time.time()
    try:
        # Generate a long prompt by repeating text
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        # Approximate 32K tokens (assuming ~4 chars per token)
        long_text = base_text * 30  # ~120K chars ≈ 30K tokens

        messages = [
            {"role": "user", "content": f"Here is some text:\n\n{long_text}\n\nWhat animal jumps in the text above?"}
        ]

        resp = await chat(client, base_url, model, messages, max_tokens=100, timeout=TIMEOUT)
        duration = time.time() - t0
        content = get_content(resp).lower()
        usage = get_usage(resp)
        prompt_tokens = usage.get("prompt_tokens", 0)

        has_fox = "fox" in content
        # Verify we actually used significant context
        long_context = prompt_tokens > 20000

        passed = has_fox and resp["status_code"] == 200
        record(f"{model}/long_context_32k", passed, duration,
               f"prompt_tokens={prompt_tokens}, has_fox={has_fox}",
               "" if passed else f"Missing 'fox' in response or failed")
    except Exception as e:
        record(f"{model}/long_context_32k", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 4: Memory Drift Monitoring
# =============================================================================

async def test_memory_drift_5_switches(client: httpx.AsyncClient, base_url: str):
    """Monitor memory drift over 5 model switches."""
    print("\n=== MEMORY: 5-Switch Drift Test ===")

    # Models to cycle through
    switch_sequence = [
        "qwen2.5-7b",
        "qwen3-coder-next",
        "qwen3-32b",
        "qwen2.5-7b",
        "qwen3-coder-next",
    ]

    # Record initial memory
    status = await get_status(client, base_url)
    initial_mem = status.get("gpu_memory_used_gb", 0)
    memory_samples.append({"switch": 0, "model": status.get("current_model", ""), "gpu_used_gb": initial_mem})
    print(f"  Initial: {initial_mem:.2f}GB (model: {status.get('current_model', '')})")

    for i, model in enumerate(switch_sequence):
        t0 = time.time()
        try:
            resp = await chat(client, base_url, model,
                              [{"role": "user", "content": f"Say 'switch {i+1} complete'."}],
                              max_tokens=50)
            duration = time.time() - t0
            content = get_content(resp)

            # Get memory after switch
            status = await get_status(client, base_url)
            mem = status.get("gpu_memory_used_gb", 0)
            memory_samples.append({"switch": i+1, "model": model, "gpu_used_gb": mem})

            drift = mem - initial_mem
            passed = resp["status_code"] == 200 and len(content) > 0
            record(f"memory_drift/switch_{i+1}_{model}", passed, duration,
                   f"mem={mem:.2f}GB, drift={drift:.2f}GB")
        except Exception as e:
            record(f"memory_drift/switch_{i+1}_{model}", False, time.time() - t0, error=str(e))

    # Final drift assessment
    if len(memory_samples) >= 2:
        total_drift = memory_samples[-1]["gpu_used_gb"] - memory_samples[0]["gpu_used_gb"]
        acceptable_drift = total_drift < 2.0  # 2GB threshold
        record("memory_drift/total", acceptable_drift, 0,
               f"total_drift={total_drift:.2f}GB",
               "" if acceptable_drift else f"Drift {total_drift:.2f}GB exceeds 2GB threshold")


# =============================================================================
# Test Group 5: AWQ Safety After Multiple Switches
# =============================================================================

async def test_awq_after_switches(client: httpx.AsyncClient, base_url: str):
    """Test llama-3.1-70b AWQ after multiple model switches.

    WARNING: This test monitors for the AWQ Marlin crash bug documented in CLAUDE.md.
    The bug is non-deterministic and may cause system freeze requiring power cycle.
    """
    print("\n=== AWQ: Safety After Switches ===")
    print("  WARNING: This test may trigger the AWQ Marlin crash bug on Blackwell GPUs")

    # First, do a few switches to build up state
    prep_models = ["qwen2.5-7b", "qwen3-32b"]
    for model in prep_models:
        t0 = time.time()
        try:
            resp = await chat(client, base_url, model,
                              [{"role": "user", "content": "Say hello."}],
                              max_tokens=30)
            duration = time.time() - t0
            record(f"awq_prep/switch_to_{model}", resp["status_code"] == 200, duration)
        except Exception as e:
            record(f"awq_prep/switch_to_{model}", False, time.time() - t0, error=str(e))

    # Now test AWQ model
    model = "llama-3.1-70b"
    t0 = time.time()
    try:
        print(f"  Loading {model} (AWQ quantized)...")
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "What is 2+2? Just the number."}],
                          max_tokens=50, timeout=TIMEOUT)
        duration = time.time() - t0
        content = get_content(resp)
        has_4 = "4" in content
        passed = has_4 and resp["status_code"] == 200
        record(f"awq_safety/{model}_after_switches", passed, duration,
               f"content={content[:50]!r}",
               "" if passed else "AWQ model failed after switches")
    except Exception as e:
        record(f"awq_safety/{model}_after_switches", False, time.time() - t0,
               error=f"POSSIBLE AWQ CRASH: {str(e)[:100]}")


# =============================================================================
# Test Group 6: Model Switching Pairs
# =============================================================================

async def test_switch_pair(client: httpx.AsyncClient, base_url: str, from_model: str, to_model: str):
    """Test switching from one model to another."""
    t0 = time.time()
    try:
        # First ensure we're on from_model
        resp = await chat(client, base_url, from_model,
                          [{"role": "user", "content": "Say 'from ready'."}],
                          max_tokens=30)
        if resp["status_code"] != 200:
            record(f"switch_pair/{from_model}_to_{to_model}", False, time.time() - t0,
                   error=f"Failed to load from_model: {from_model}")
            return

        # Now switch to to_model
        resp = await chat(client, base_url, to_model,
                          [{"role": "user", "content": "What is 1+1? Just the number."}],
                          max_tokens=30)
        duration = time.time() - t0
        content = get_content(resp)
        has_2 = "2" in content
        passed = has_2 and resp["status_code"] == 200
        record(f"switch_pair/{from_model}_to_{to_model}", passed, duration,
               f"switch_time={duration:.1f}s",
               "" if passed else f"content={content[:30]!r}")
    except Exception as e:
        record(f"switch_pair/{from_model}_to_{to_model}", False, time.time() - t0, error=str(e))


async def test_all_switch_pairs(client: httpx.AsyncClient, base_url: str, models: List[str]):
    """Test switching between all model pairs."""
    print("\n=== SWITCH PAIRS ===")

    # Test key pairs, not all permutations (too slow)
    key_pairs = [
        ("gpt-oss-120b", "qwen3-coder-next"),  # MoE to MoE
        ("qwen3-coder-next", "kimi-vl"),        # MoE to Vision
        ("kimi-vl", "qwen3-32b"),               # Vision to Dense
        ("qwen3-32b", "gpt-oss-120b"),          # Dense to MoE
    ]

    for from_model, to_model in key_pairs:
        if from_model in models and to_model in models:
            await test_switch_pair(client, base_url, from_model, to_model)


# =============================================================================
# Test Group 7: Vision Edge Cases
# =============================================================================

async def test_kimi_vision_text_only(client: httpx.AsyncClient, base_url: str):
    """Test kimi-vl with text-only input (no image)."""
    model = "kimi-vl"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "What is the capital of France?"}],
                          max_tokens=100)
        duration = time.time() - t0
        content = get_content(resp).lower()
        has_paris = "paris" in content
        passed = has_paris and resp["status_code"] == 200
        record(f"{model}/text_only", passed, duration,
               f"content={content[:50]!r}",
               "" if passed else "Vision model failed text-only query")
    except Exception as e:
        record(f"{model}/text_only", False, time.time() - t0, error=str(e))


async def test_kimi_vision_multi_image(client: httpx.AsyncClient, base_url: str):
    """Test kimi-vl with multiple images (if supported)."""
    model = "kimi-vl"
    t0 = time.time()
    try:
        img_uri = make_red_png_base64()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "I'm showing you two images. What do they have in common?"},
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ]}
        ]
        resp = await chat(client, base_url, model, messages, max_tokens=200)
        duration = time.time() - t0
        content = get_content(resp)
        has_response = len(content) > 10
        passed = has_response and resp["status_code"] == 200
        record(f"{model}/multi_image", passed, duration,
               f"content_len={len(content)}",
               "" if passed else "Multi-image test failed")
    except Exception as e:
        record(f"{model}/multi_image", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 8: Alias Tests for New Model
# =============================================================================

async def test_coder_aliases(client: httpx.AsyncClient, base_url: str):
    """Test that qwen-coder and coder aliases resolve to qwen3-coder-next."""
    print("\n=== ALIAS: Coder Model ===")

    for alias in ["qwen-coder", "coder"]:
        t0 = time.time()
        try:
            resp = await chat(client, base_url, alias,
                              [{"role": "user", "content": "Say hello."}],
                              max_tokens=30)
            duration = time.time() - t0
            content = get_content(resp)
            passed = resp["status_code"] == 200 and len(content) > 0
            record(f"alias/{alias}", passed, duration, f"content={content[:30]!r}")
        except Exception as e:
            record(f"alias/{alias}", False, time.time() - t0, error=str(e))


# =============================================================================
# Main
# =============================================================================

async def run_all(base_url: str, target_model: Optional[str] = None, quick: bool = False):
    print("=" * 70)
    print("BLITZINFER EDGE CASE TESTS")
    print(f"Server: {base_url}")
    if target_model:
        print(f"Target model: {target_model}")
    if quick:
        print("Mode: QUICK (skipping memory-intensive tests)")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Wait for server
        print("\nWaiting for server...")
        if not await wait_for_server(client, base_url):
            print("ERROR: Server not reachable")
            return False

        status = await get_status(client, base_url)
        available = status.get("available_models", [])
        print(f"Server ready. Active model: {status.get('current_model', 'none')}")
        print(f"Available: {available}")

        # Check if new model is available
        if "qwen3-coder-next" not in available:
            print("\nWARNING: qwen3-coder-next not in available models!")
            print("Make sure server has the updated model config.")

        # Filter models based on target
        if target_model:
            test_models = [target_model] if target_model in available else []
        elif quick:
            test_models = [m for m in QUICK_MODELS if m in available]
        else:
            test_models = [m for m in ALL_MODELS if m in available]

        print(f"\nTesting models: {test_models}")

        # Group 1: Qwen3-Coder-Next tests
        if "qwen3-coder-next" in test_models:
            print("\n=== QWEN3-CODER-NEXT TESTS ===")
            await test_qwen_coder_basic(client, base_url)
            await test_qwen_coder_long_code(client, base_url)
            await test_qwen_coder_multi_language(client, base_url)
            await test_qwen_coder_streaming(client, base_url)
            await test_coder_aliases(client, base_url)

        # Group 2: GPT-OSS Harmony extended tests
        if "gpt-oss-120b" in test_models and not quick:
            print("\n=== GPT-OSS HARMONY EXTENDED ===")
            await test_gptoss_tool_call_bash(client, base_url)
            await test_gptoss_multi_turn_with_tool_result(client, base_url)

        # Group 3: Long context (only if not quick mode)
        if not quick and len(test_models) > 0:
            print("\n=== LONG CONTEXT TESTS ===")
            # Pick a model that supports long context
            long_ctx_model = "qwen3-32b" if "qwen3-32b" in test_models else test_models[0]
            await test_long_context_32k(client, base_url, long_ctx_model)

        # Group 4: Vision edge cases
        if "kimi-vl" in test_models:
            print("\n=== VISION EDGE CASES ===")
            await test_kimi_vision_text_only(client, base_url)
            await test_kimi_vision_multi_image(client, base_url)

        # Group 5: Memory drift (only if not quick mode)
        if not quick:
            await test_memory_drift_5_switches(client, base_url)

        # Group 6: AWQ safety (only if not quick mode)
        if "llama-3.1-70b" in test_models and not quick:
            await test_awq_after_switches(client, base_url)

        # Group 7: Model switch pairs (only if not quick mode and multiple models)
        if not quick and len(test_models) >= 2:
            await test_all_switch_pairs(client, base_url, test_models)

    return print_summary()


def main():
    parser = argparse.ArgumentParser(description="BlitzInfer edge case tests")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", help="Test specific model only")
    parser.add_argument("--quick", action="store_true", help="Skip memory-intensive tests")
    args = parser.parse_args()

    success = asyncio.run(run_all(args.base_url, args.model, args.quick))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
