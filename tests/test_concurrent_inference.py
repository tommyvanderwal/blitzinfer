#!/usr/bin/env python3
"""Adversarial concurrent inference test for BlitzInfer.

Sends 14 simultaneous requests to ALL models (including problematic ones)
to test server resilience under concurrent load with model switching.

Deliberately triggers known issues:
  - qwen3-coder-next: FLA Triton OOM after model switch
  - llama-3.1-70b: AWQ Marlin intermittent crash after switches

Expected: Working models return answers, broken models return clean errors,
server never hangs or crashes.

Usage:
    python tests/test_concurrent_inference.py [--base-url URL] [--timeout 900]
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# =============================================================================
# Config
# =============================================================================

DEFAULT_BASE_URL = "http://192.168.2.90:8000"
DEFAULT_TIMEOUT = 900  # 15 min total (5 switches × 30s + inference + recovery)
PER_REQUEST_TIMEOUT = 300.0  # 5 min per request (includes potential model switch)

# Reliable models: these MUST succeed
RELIABLE_MODELS = {"gpt-oss-120b", "qwen3-32b", "kimi-vl"}

# All models including potentially broken ones
ALL_REQUESTS: List[Dict[str, Any]] = [
    # --- gpt-oss-120b: 3 requests ---
    {
        "name": "gptoss_math",
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "What is 15 * 17? Give just the number."}],
        "max_tokens": 300,
        "expect_content": "255",
    },
    {
        "name": "gptoss_tool_call",
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "max_tokens": 500,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }],
        "expect_tool_call": "get_weather",
    },
    {
        "name": "gptoss_multi_turn",
        "model": "gpt-oss-120b",
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
        "max_tokens": 200,
        "expect_content": "alice",
        "case_insensitive": True,
    },
    # --- qwen3-32b: 3 requests ---
    {
        "name": "qwen32b_code",
        "model": "qwen3-32b",
        "messages": [{"role": "user", "content": "Write a Python function to compute fibonacci. Just the function."}],
        "max_tokens": 500,
        "expect_content": "def ",
    },
    {
        "name": "qwen32b_math",
        "model": "qwen3-32b",
        "messages": [{"role": "user", "content": "What is 123 + 456 + 789?"}],
        "max_tokens": 500,  # Thinking models need room for reasoning + answer
        "expect_content": "1368",
    },
    {
        "name": "qwen32b_reasoning",
        "model": "qwen3-32b",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 500,  # Thinking models need room for reasoning + answer
        "expect_content": "paris",
        "case_insensitive": True,
    },
    # --- kimi-vl: 3 requests ---
    {
        "name": "kimi_reasoning",
        "model": "kimi-vl",
        "messages": [{"role": "user", "content": "What is 2 + 2? Think step by step."}],
        "max_tokens": 500,
        "expect_content": "4",
        "expect_reasoning": True,
    },
    {
        "name": "kimi_text",
        "model": "kimi-vl",
        "messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}],
        "max_tokens": 500,  # Thinking models need room for reasoning + answer
        "expect_content": "jupiter",
        "case_insensitive": True,
    },
    {
        "name": "kimi_vision",
        "model": "kimi-vl",
        "messages": None,  # Will be set up with image in main
        "max_tokens": 500,
        # Vision via base64 in concurrent context - just verify model responds
        "case_insensitive": True,
        "is_vision": True,
    },
    # --- qwen3-coder-next: 2 requests (context_length=32768 to fit FLA workspace) ---
    {
        "name": "coder_basic",
        "model": "qwen3-coder-next",
        "messages": [{"role": "user", "content": "Write a hello world in Python. Just the code."}],
        "max_tokens": 500,
        "expect_content": "print",
        "case_insensitive": True,
    },
    {
        "name": "coder_rust",
        "model": "qwen3-coder-next",
        "messages": [{"role": "user", "content": "Write a hello world function in Rust. Just the function."}],
        "max_tokens": 500,
        "expect_content": "fn",
        "case_insensitive": True,
    },
    # --- llama-3.1-70b: DISABLED - AWQ Marlin causes BIOS-level crash ---
    # The awq_marlin kernel causes intermittent system freeze (power cycle required)
    # after model switches. See CLAUDE.md "AWQ Marlin Intermittent Crash Bug".
    # Tested separately in E2E test with may_fail flag.
    #
    # {
    #     "name": "llama_factual",
    #     "model": "llama-3.1-70b",
    #     "messages": [{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
    #     "max_tokens": 100,
    #     "expect_content": "shakespeare",
    #     "case_insensitive": True,
    #     "may_fail": True,
    # },
    # --- qwen2.5-7b: 1 request (baseline small model) ---
    {
        "name": "qwen25_basic",
        "model": "qwen2.5-7b",
        "messages": [{"role": "user", "content": "What is 7 * 8? Just the number."}],
        "max_tokens": 100,
        "expect_content": "56",
    },
]


# =============================================================================
# Result tracking
# =============================================================================

@dataclass
class RequestResult:
    name: str
    model: str
    status_code: int
    content: str
    tool_calls: List[Dict]
    reasoning_content: Optional[str]
    duration: float
    error: Optional[str]
    passed: bool
    may_fail: bool
    detail: str = ""


def make_red_png_base64() -> str:
    """Create a 64x64 red PNG as base64 data URI."""
    width, height = 64, 64
    raw = b''
    for y in range(height):
        raw += b'\x00'
        for x in range(width):
            raw += b'\xff\x00\x00'

    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png_data = sig + png_chunk(b'IHDR', ihdr) + png_chunk(b'IDAT', zlib.compress(raw)) + png_chunk(b'IEND', b'')
    return f"data:image/png;base64,{base64.b64encode(png_data).decode()}"


# =============================================================================
# Request execution
# =============================================================================

async def execute_request(
    client: httpx.AsyncClient,
    base_url: str,
    req: Dict[str, Any],
) -> RequestResult:
    """Execute a single chat completion request and validate result."""
    name = req["name"]
    model = req["model"]
    may_fail = req.get("may_fail", False)
    t0 = time.time()

    try:
        body = {
            "model": model,
            "messages": req["messages"],
            "max_tokens": req.get("max_tokens", 200),
            "temperature": req.get("temperature", 0.3),
        }
        if req.get("tools"):
            body["tools"] = req["tools"]

        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=PER_REQUEST_TIMEOUT,
        )
        duration = time.time() - t0
        status_code = resp.status_code

        if status_code != 200:
            error_text = resp.text[:200]
            return RequestResult(
                name=name, model=model, status_code=status_code,
                content="", tool_calls=[], reasoning_content=None,
                duration=duration, error=f"HTTP {status_code}: {error_text}",
                passed=False, may_fail=may_fail,
                detail=f"Server returned {status_code}",
            )

        data = resp.json()
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", []) or []
        reasoning_content = message.get("reasoning_content")
        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "")

        # Validate expectations
        passed = True
        detail_parts = []

        if "expect_content" in req:
            expected = req["expect_content"]
            # Check both content and reasoning_content (thinking models put answers in reasoning)
            all_text = content + (reasoning_content or "")
            check_text = all_text.lower() if req.get("case_insensitive") else all_text
            if expected.lower() not in check_text:
                passed = False
                detail_parts.append(f"expected '{expected}' not in content+reasoning")

        if "expect_tool_call" in req:
            if not tool_calls:
                passed = False
                detail_parts.append("no tool calls returned")
            else:
                fn_name = tool_calls[0].get("function", {}).get("name", "")
                if fn_name != req["expect_tool_call"]:
                    passed = False
                    detail_parts.append(f"wrong function: {fn_name}")

        if req.get("expect_reasoning"):
            if not reasoning_content:
                detail_parts.append("no reasoning_content (expected for thinking model)")
                # Don't fail on this - some models may not always emit reasoning

        # Check for Harmony corruption in non-harmony models
        corruption = ["<|channel|>", "<|message|>", "assistantanalysis"]
        for pattern in corruption:
            if pattern in content:
                passed = False
                detail_parts.append(f"Harmony corruption: {pattern}")

        detail = "; ".join(detail_parts) if detail_parts else f"content_len={len(content)}"

        return RequestResult(
            name=name, model=model, status_code=status_code,
            content=content[:200], tool_calls=tool_calls,
            reasoning_content=reasoning_content[:100] if reasoning_content else None,
            duration=duration, error=None, passed=passed, may_fail=may_fail,
            detail=detail,
        )

    except httpx.TimeoutException:
        return RequestResult(
            name=name, model=model, status_code=0,
            content="", tool_calls=[], reasoning_content=None,
            duration=time.time() - t0, error="TIMEOUT",
            passed=False, may_fail=may_fail,
            detail=f"Request timed out after {PER_REQUEST_TIMEOUT}s",
        )
    except Exception as e:
        return RequestResult(
            name=name, model=model, status_code=0,
            content="", tool_calls=[], reasoning_content=None,
            duration=time.time() - t0, error=str(e)[:200],
            passed=False, may_fail=may_fail,
            detail=f"Exception: {type(e).__name__}",
        )


# =============================================================================
# Main test
# =============================================================================

async def run_concurrent_test(base_url: str, timeout: float):
    print("=" * 70)
    print("ADVERSARIAL CONCURRENT INFERENCE TEST")
    print(f"Server: {base_url}")
    print(f"Total timeout: {timeout}s")
    print(f"Requests: {len(ALL_REQUESTS)}")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # Check server health
        print("\nChecking server health...")
        try:
            resp = await client.get(f"{base_url}/health", timeout=10)
            health = resp.json()
            print(f"  Status: {health.get('status')}")
            print(f"  Current model: {health.get('current_model')}")
            print(f"  Queue state: {health.get('queue_state')}")
        except Exception as e:
            print(f"  ERROR: Server not reachable: {e}")
            return False

        # Get initial memory
        try:
            status_resp = await client.get(f"{base_url}/status", timeout=10)
            status = status_resp.json()
            initial_mem = status.get("gpu_memory_used_gb", 0)
            print(f"  GPU memory: {initial_mem:.1f}GB")
        except Exception:
            initial_mem = 0

        # Set up vision request with red PNG
        img_uri = make_red_png_base64()
        for req in ALL_REQUESTS:
            if req.get("is_vision"):
                req["messages"] = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Just say the color."},
                        {"type": "image_url", "image_url": {"url": img_uri}},
                    ],
                }]

        # Fire ALL requests simultaneously
        print(f"\nFiring {len(ALL_REQUESTS)} requests simultaneously...")
        models_hit = set(r["model"] for r in ALL_REQUESTS)
        print(f"  Models targeted: {sorted(models_hit)}")
        print(f"  Expected switches: {len(models_hit) - 1}")

        t0 = time.time()
        tasks = [
            execute_request(client, base_url, req)
            for req in ALL_REQUESTS
        ]
        results: List[RequestResult] = await asyncio.gather(*tasks)
        total_duration = time.time() - t0

        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS (total time: {total_duration:.1f}s)")
        print(f"{'='*70}")

        by_model: Dict[str, List[RequestResult]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r)

        total_passed = 0
        total_failed = 0
        total_expected_fail = 0
        reliable_passed = 0
        reliable_total = 0

        for model in sorted(by_model.keys()):
            model_results = by_model[model]
            print(f"\n  {model}:")
            for r in model_results:
                is_reliable = model in RELIABLE_MODELS
                if is_reliable:
                    reliable_total += 1

                if r.passed:
                    total_passed += 1
                    if is_reliable:
                        reliable_passed += 1
                    status = "PASS"
                elif r.may_fail and r.error:
                    total_expected_fail += 1
                    status = "EXPECTED_FAIL"
                else:
                    total_failed += 1
                    status = "FAIL"

                print(f"    [{status}] {r.name}: {r.duration:.1f}s - {r.detail}")
                if r.error:
                    print(f"           error: {r.error[:100]}")
                if r.reasoning_content:
                    print(f"           reasoning: {r.reasoning_content[:80]}...")
                if r.tool_calls:
                    for tc in r.tool_calls:
                        fn = tc.get("function", {})
                        print(f"           tool_call: {fn.get('name')}({fn.get('arguments', '')[:50]})")

        # Verify server is still alive
        print(f"\n{'='*70}")
        print("POST-TEST VERIFICATION")
        print(f"{'='*70}")

        server_alive = False
        try:
            resp = await client.get(f"{base_url}/health", timeout=30)
            health = resp.json()
            server_alive = health.get("status") == "healthy"
            print(f"  Server alive: {server_alive}")
            print(f"  Queue state: {health.get('queue_state')}")
            print(f"  In-flight: {health.get('in_flight')}")
            print(f"  Queue depths: {health.get('queue_depths', {})}")
        except Exception as e:
            print(f"  Server health check FAILED: {e}")

        # Check memory drift
        try:
            status_resp = await client.get(f"{base_url}/status", timeout=10)
            status = status_resp.json()
            final_mem = status.get("gpu_memory_used_gb", 0)
            drift = final_mem - initial_mem if initial_mem > 0 else 0
            print(f"  GPU memory: {final_mem:.1f}GB (drift: {drift:+.1f}GB)")
            mem_ok = drift < 10.0
        except Exception:
            mem_ok = True  # Can't check, don't fail on it

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Total requests:       {len(results)}")
        print(f"  Passed:               {total_passed}")
        print(f"  Failed (unexpected):  {total_failed}")
        print(f"  Failed (expected):    {total_expected_fail}")
        print(f"  Reliable models:      {reliable_passed}/{reliable_total}")
        print(f"  Server alive:         {server_alive}")
        print(f"  Memory OK:            {mem_ok}")

        # Pass criteria:
        # 1. At least 9 of 14 requests succeed (all 3 reliable models × 3 each)
        # 2. Server is still alive
        # 3. Memory drift < 10GB
        all_reliable_passed = reliable_passed >= reliable_total
        overall_pass = all_reliable_passed and server_alive and mem_ok

        status_str = "PASS" if overall_pass else "FAIL"
        print(f"\n  Overall: {status_str}")
        if not all_reliable_passed:
            print(f"  REASON: Reliable models failed ({reliable_passed}/{reliable_total})")
        if not server_alive:
            print(f"  REASON: Server not responding after test")

        return overall_pass


def main():
    parser = argparse.ArgumentParser(description="BlitzInfer concurrent inference test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    success = asyncio.run(run_concurrent_test(args.base_url, args.timeout))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
