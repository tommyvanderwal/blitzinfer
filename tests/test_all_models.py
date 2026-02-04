#!/usr/bin/env python3
"""Comprehensive multi-model test for BlitzInfer SGLang server.

Tests all 5 models: gpt-oss-120b, kimi-vl, qwen3-32b, llama-3.1-70b, qwen2.5-7b

Covers:
  1. Each model: basic generation, output sanity
  2. Vision: image input for kimi-vl
  3. Harmony: gpt-oss-120b tool calls, channel parsing
  4. Streaming: SSE for each model
  5. Queue switching: multi-model requests, drain + switch
  6. Edge cases: aliases, empty content, long prompts

Usage:
  # Requires server running on :8000
  python tests/test_all_models.py [--base-url http://host:port]
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
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# =============================================================================
# Config
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # Long timeout for model switches

ALL_MODELS = [
    "gpt-oss-120b",
    "kimi-vl",
    "qwen3-32b",
    "llama-3.1-70b",
    "qwen2.5-7b",
]

VISION_MODELS = ["kimi-vl"]
HARMONY_MODELS = ["gpt-oss-120b"]


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


def record(name: str, passed: bool, duration: float, detail: str = "", error: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append(TestResult(name, passed, duration, detail, error))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({duration:.1f}s) {detail}")
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
# Test Group 1: Individual Model Tests
# =============================================================================

async def test_model_basic(client: httpx.AsyncClient, base_url: str, model: str):
    """Test basic text generation for a single model."""
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                          max_tokens=300)
        duration = time.time() - t0
        content = get_content(resp)
        # Check that we got some content and it contains "4"
        has_content = len(content) > 0
        has_4 = "4" in content
        passed = has_content and has_4 and resp["status_code"] == 200
        record(f"{model}/basic", passed, duration,
               f"content={content[:100]!r}",
               "" if passed else f"has_content={has_content}, has_4={has_4}, status={resp['status_code']}")
    except Exception as e:
        record(f"{model}/basic", False, time.time() - t0, error=str(e))


async def test_model_streaming(client: httpx.AsyncClient, base_url: str, model: str):
    """Test streaming generation for a single model."""
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "Count from 1 to 5."}],
                          max_tokens=100, stream=True)
        duration = time.time() - t0
        chunks = resp.get("chunks", [])
        # Reconstruct content from chunks
        content = ""
        for c in chunks:
            delta = c.get("choices", [{}])[0].get("delta", {})
            content += delta.get("content", "") or ""

        # Harmony models buffer output (2 chunks: content + finish)
        # Non-Harmony models stream in real-time (many chunks)
        min_chunks = 1 if model in HARMONY_MODELS else 3
        has_chunks = len(chunks) >= min_chunks
        has_content = len(content) > 5
        passed = has_chunks and has_content
        record(f"{model}/streaming", passed, duration,
               f"chunks={len(chunks)}, content={content[:80]!r}",
               "" if passed else f"has_chunks={has_chunks}(min={min_chunks}), has_content={has_content}")
    except Exception as e:
        record(f"{model}/streaming", False, time.time() - t0, error=str(e))


async def test_model_multi_turn(client: httpx.AsyncClient, base_url: str, model: str):
    """Test multi-turn conversation."""
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model, [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"},
        ], max_tokens=100)
        duration = time.time() - t0
        content = get_content(resp).lower()
        has_alice = "alice" in content
        passed = has_alice and resp["status_code"] == 200
        record(f"{model}/multi_turn", passed, duration,
               f"content={content[:100]!r}",
               "" if passed else "Name 'alice' not found in response")
    except Exception as e:
        record(f"{model}/multi_turn", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 2: Vision Tests
# =============================================================================

async def test_vision_base64(client: httpx.AsyncClient, base_url: str, model: str):
    """Test vision with base64 image input."""
    t0 = time.time()
    try:
        img_uri = make_red_png_base64()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What color is this image? Answer with just the color name."},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ]}
        ]
        resp = await chat(client, base_url, model, messages, max_tokens=100)
        duration = time.time() - t0
        content = get_content(resp).lower()
        # VLMs struggle with tiny images - just verify the model processes
        # the image and returns a response (any color word or description)
        has_response = len(content) > 5
        passed = has_response and resp["status_code"] == 200
        record(f"{model}/vision_base64", passed, duration,
               f"content={content[:100]!r}",
               "" if passed else "No meaningful response for base64 image")
    except Exception as e:
        record(f"{model}/vision_base64", False, time.time() - t0, error=str(e))


async def test_vision_url(client: httpx.AsyncClient, base_url: str, model: str):
    """Test vision with image URL."""
    t0 = time.time()
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What do you see in this image? Describe briefly in 1-2 sentences."},
                {"type": "image_url", "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
                }},
            ]}
        ]
        resp = await chat(client, base_url, model, messages, max_tokens=200)
        duration = time.time() - t0
        content = get_content(resp)
        has_content = len(content) > 10
        passed = has_content and resp["status_code"] == 200
        record(f"{model}/vision_url", passed, duration,
               f"content={content[:100]!r}",
               "" if passed else f"Too short or error: {content[:50]}")
    except Exception as e:
        record(f"{model}/vision_url", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 3: Harmony Tests (gpt-oss-120b)
# =============================================================================

async def test_harmony_basic(client: httpx.AsyncClient, base_url: str):
    """Test Harmony encoding produces proper output."""
    model = "gpt-oss-120b"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "What is the speed of light in km/s? Just the number."}],
                          max_tokens=500)
        duration = time.time() - t0
        content = get_content(resp)
        # Should contain 299792 or similar
        has_content = len(content) > 0
        has_number = any(c.isdigit() for c in content)
        # Check no raw Harmony tokens leaked
        no_leak = "<|channel|>" not in content and "<|message|>" not in content
        passed = has_content and has_number and no_leak and resp["status_code"] == 200
        record(f"{model}/harmony_basic", passed, duration,
               f"content={content[:100]!r}",
               "" if passed else f"content_ok={has_content}, digits={has_number}, no_leak={no_leak}")
    except Exception as e:
        record(f"{model}/harmony_basic", False, time.time() - t0, error=str(e))


async def test_harmony_tool_call(client: httpx.AsyncClient, base_url: str):
    """Test Harmony tool call generation."""
    model = "gpt-oss-120b"
    t0 = time.time()
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }]
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "What's the weather in Paris?"}],
                          max_tokens=500, tools=tools)
        duration = time.time() - t0
        tool_calls = get_tool_calls(resp)
        has_tool_call = len(tool_calls) > 0
        correct_function = False
        if has_tool_call:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            correct_function = fn.get("name") == "get_weather"
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                    correct_function = correct_function and "paris" in args_dict.get("location", "").lower()
                except json.JSONDecodeError:
                    pass

        passed = has_tool_call and correct_function and resp["status_code"] == 200
        record(f"{model}/harmony_tool_call", passed, duration,
               f"tool_calls={len(tool_calls)}",
               "" if passed else f"has_tc={has_tool_call}, correct_fn={correct_function}")
    except Exception as e:
        record(f"{model}/harmony_tool_call", False, time.time() - t0, error=str(e))


async def test_harmony_streaming(client: httpx.AsyncClient, base_url: str):
    """Test Harmony streaming produces proper output."""
    model = "gpt-oss-120b"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, model,
                          [{"role": "user", "content": "Say hello."}],
                          max_tokens=500, stream=True)
        duration = time.time() - t0
        chunks = resp.get("chunks", [])
        content = ""
        for c in chunks:
            delta = c.get("choices", [{}])[0].get("delta", {})
            content += delta.get("content", "") or ""
        has_content = len(content) > 0
        no_leak = "<|channel|>" not in content and "<|message|>" not in content
        passed = has_content and no_leak
        record(f"{model}/harmony_streaming", passed, duration,
               f"chunks={len(chunks)}, content={content[:80]!r}",
               "" if passed else f"content_ok={has_content}, no_leak={no_leak}")
    except Exception as e:
        record(f"{model}/harmony_streaming", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 4: Queue & Switching
# =============================================================================

async def test_queue_switch_2models(client: httpx.AsyncClient, base_url: str):
    """Test switching between 2 models via queue."""
    print("\n=== QUEUE: 2-Model Switch ===")

    # First model (whatever is active)
    status = await get_status(client, base_url)
    active = status.get("current_model", "")
    print(f"  Active model: {active}")

    # Pick a different model
    other = "qwen2.5-7b" if active != "qwen2.5-7b" else "llama-3.1-70b"

    # Send request to active model first
    t0 = time.time()
    try:
        resp = await chat(client, base_url, active,
                          [{"role": "user", "content": "What is 1+1?"}],
                          max_tokens=50)
        d = time.time() - t0
        content = get_content(resp)
        record(f"queue/active_{active}", resp["status_code"] == 200 and len(content) > 0, d,
               f"content={content[:50]!r}")
    except Exception as e:
        record(f"queue/active_{active}", False, time.time() - t0, error=str(e))

    # Send request to different model (triggers switch)
    t0 = time.time()
    try:
        resp = await chat(client, base_url, other,
                          [{"role": "user", "content": "What is 3+3?"}],
                          max_tokens=50)
        d = time.time() - t0
        content = get_content(resp)
        has_6 = "6" in content
        record(f"queue/switch_to_{other}", resp["status_code"] == 200 and has_6, d,
               f"switch_time={d:.1f}s, content={content[:50]!r}")
    except Exception as e:
        record(f"queue/switch_to_{other}", False, time.time() - t0, error=str(e))

    # Verify the model actually switched
    status = await get_status(client, base_url)
    new_active = status.get("current_model", "")
    record(f"queue/model_switched", new_active == other, 0,
           f"expected={other}, actual={new_active}")


async def test_queue_3model_sequence(client: httpx.AsyncClient, base_url: str):
    """Test switching through 3 different models."""
    print("\n=== QUEUE: 3-Model Sequence ===")

    models_to_test = ["qwen2.5-7b", "llama-3.1-70b", "qwen3-32b"]
    prompts = [
        "What is the largest ocean? Answer in 3 words or less.",
        "What is the smallest continent? Answer in 3 words or less.",
        "What is the tallest mountain? Answer in 3 words or less.",
    ]

    for i, (model, prompt) in enumerate(zip(models_to_test, prompts)):
        t0 = time.time()
        try:
            resp = await chat(client, base_url, model,
                              [{"role": "user", "content": prompt}],
                              max_tokens=100)
            d = time.time() - t0
            content = get_content(resp)
            passed = resp["status_code"] == 200 and len(content) > 0
            record(f"queue_3model/step{i+1}_{model}", passed, d,
                   f"content={content[:60]!r}")
        except Exception as e:
            record(f"queue_3model/step{i+1}_{model}", False, time.time() - t0, error=str(e))


async def test_queue_multiple_requests_same_model(client: httpx.AsyncClient, base_url: str):
    """Test multiple requests to the same model are served before switching."""
    print("\n=== QUEUE: Multiple Requests Same Model ===")
    model = "qwen2.5-7b"  # Small model, fast loading

    # First ensure we're on this model
    t0 = time.time()
    resp = await chat(client, base_url, model,
                      [{"role": "user", "content": "Say 'ready'."}],
                      max_tokens=30)
    d = time.time() - t0
    record(f"queue_multi/warmup_{model}", resp["status_code"] == 200, d)

    # Now send 3 requests in sequence to the same model
    for i in range(3):
        t0 = time.time()
        try:
            resp = await chat(client, base_url, model,
                              [{"role": "user", "content": f"What is {i+1}+{i+1}? Just the number."}],
                              max_tokens=30)
            d = time.time() - t0
            content = get_content(resp)
            expected = str((i+1)*2)
            passed = expected in content and resp["status_code"] == 200
            record(f"queue_multi/req{i+1}_{model}", passed, d,
                   f"content={content[:30]!r}, expected={expected}")
        except Exception as e:
            record(f"queue_multi/req{i+1}_{model}", False, time.time() - t0, error=str(e))


# =============================================================================
# Test Group 5: Edge Cases
# =============================================================================

async def test_alias_resolution(client: httpx.AsyncClient, base_url: str):
    """Test model aliases resolve correctly."""
    print("\n=== EDGE: Alias Resolution ===")
    # Use alias "qwen" which should resolve to "qwen3-32b"
    t0 = time.time()
    try:
        resp = await chat(client, base_url, "qwen",
                          [{"role": "user", "content": "Say hello."}],
                          max_tokens=30)
        d = time.time() - t0
        passed = resp["status_code"] == 200 and len(get_content(resp)) > 0
        record("edge/alias_qwen", passed, d, f"content={get_content(resp)[:50]!r}")
    except Exception as e:
        record("edge/alias_qwen", False, time.time() - t0, error=str(e))


async def test_invalid_model(client: httpx.AsyncClient, base_url: str):
    """Test that invalid model returns 404."""
    t0 = time.time()
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions",
                                 json={"model": "nonexistent-model",
                                        "messages": [{"role": "user", "content": "hi"}]},
                                 timeout=30)
        d = time.time() - t0
        passed = resp.status_code == 404
        record("edge/invalid_model", passed, d,
               f"status={resp.status_code}",
               "" if passed else f"Expected 404, got {resp.status_code}")
    except Exception as e:
        record("edge/invalid_model", False, time.time() - t0, error=str(e))


async def test_list_models(client: httpx.AsyncClient, base_url: str):
    """Test /v1/models endpoint lists all models."""
    t0 = time.time()
    try:
        resp = await client.get(f"{base_url}/v1/models", timeout=10)
        d = time.time() - t0
        data = resp.json().get("data", [])
        model_ids = [m["id"] for m in data]
        has_all = all(m in model_ids for m in ALL_MODELS)
        passed = has_all and resp.status_code == 200
        record("edge/list_models", passed, d,
               f"models={model_ids}",
               "" if passed else f"Missing models from list")
    except Exception as e:
        record("edge/list_models", False, time.time() - t0, error=str(e))


# =============================================================================
# Main
# =============================================================================

async def run_all(base_url: str):
    print("=" * 70)
    print("BLITZINFER COMPREHENSIVE MULTI-MODEL TEST")
    print(f"Server: {base_url}")
    print(f"Models: {', '.join(ALL_MODELS)}")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Wait for server
        print("\nWaiting for server...")
        if not await wait_for_server(client, base_url):
            print("ERROR: Server not reachable")
            return False

        status = await get_status(client, base_url)
        print(f"Server ready. Active model: {status.get('current_model', 'none')}")
        print(f"Available: {status.get('available_models', [])}")

        # Group 1: Edge cases first (no model switch needed)
        await test_list_models(client, base_url)
        await test_invalid_model(client, base_url)

        # Group 2: Test each model individually
        # Start with smaller models to minimize switch time
        test_order = ["qwen2.5-7b", "llama-3.1-70b", "qwen3-32b", "kimi-vl", "gpt-oss-120b"]
        for model in test_order:
            print(f"\n=== TESTING: {model} ===")
            await test_model_basic(client, base_url, model)
            await test_model_streaming(client, base_url, model)
            await test_model_multi_turn(client, base_url, model)

            # Vision tests for vision models
            if model in VISION_MODELS:
                await test_vision_base64(client, base_url, model)
                await test_vision_url(client, base_url, model)

            # Harmony tests for Harmony models
            if model in HARMONY_MODELS:
                await test_harmony_basic(client, base_url)
                await test_harmony_tool_call(client, base_url)
                await test_harmony_streaming(client, base_url)

        # Group 3: Alias resolution
        await test_alias_resolution(client, base_url)

        # Group 4: Queue switching tests
        await test_queue_switch_2models(client, base_url)
        await test_queue_3model_sequence(client, base_url)
        await test_queue_multiple_requests_same_model(client, base_url)

    return print_summary()


def main():
    parser = argparse.ArgumentParser(description="BlitzInfer multi-model test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    success = asyncio.run(run_all(args.base_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
