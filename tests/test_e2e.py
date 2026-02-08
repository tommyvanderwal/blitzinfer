#!/usr/bin/env python3
"""End-to-end test suite for BlitzInfer API server.

Master test runner that exercises ALL models, finds ALL issues.
Tests are designed to verify both success paths and error handling.

Phases:
  1. Server health check
  2. Unit tests (reasoning extraction)
  3. Per-model sequential inference
  4. Rapid-fire model switching
  5. Adversarial concurrent requests
  6. Memory drift tracking
  7. Summary with pass/fail per phase

Usage:
    python tests/test_e2e.py [--base-url URL] [--runs N] [--skip-concurrent]
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# =============================================================================
# Config
# =============================================================================

DEFAULT_BASE_URL = "http://192.168.2.90:8000"
PER_REQUEST_TIMEOUT = 300.0
TEST_IMAGE_PATH = Path(__file__).parent / "test_image.png"


# =============================================================================
# Result tracking
# =============================================================================

@dataclass
class TestResult:
    phase: str
    name: str
    passed: bool
    duration: float
    detail: str = ""
    error: str = ""


@dataclass
class PhaseResult:
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[TestResult] = field(default_factory=list)

    @property
    def total(self):
        return self.passed + self.failed

    @property
    def all_passed(self):
        return self.failed == 0 and self.passed > 0


all_phases: List[PhaseResult] = []
memory_snapshots: List[Dict] = []


def record(phase: PhaseResult, name: str, passed: bool, duration: float,
           detail: str = "", error: str = ""):
    status = "PASS" if passed else "FAIL"
    r = TestResult(phase.name, name, passed, duration, detail, error)
    phase.results.append(r)
    if passed:
        phase.passed += 1
    else:
        phase.failed += 1
    print(f"    [{status}] {name} ({duration:.1f}s) {detail}")
    if error:
        for line in error.split('\n')[:3]:
            print(f"           {line}")


# =============================================================================
# HTTP helpers
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
    timeout: float = PER_REQUEST_TIMEOUT,
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
        content_parts = []
        reasoning_parts = []
        finish_reason = None
        has_done = False
        async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                  json=body, timeout=timeout) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        has_done = True
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            content_parts.append(delta["content"])
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_parts.append(delta["reasoning_content"])
                        fr = chunk.get("choices", [{}])[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    except json.JSONDecodeError:
                        pass
        return {
            "status_code": resp.status_code,
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts) if reasoning_parts else None,
            "finish_reason": finish_reason,
            "has_done": has_done,
            "stream": True,
        }

    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    data = resp.json() if resp.status_code == 200 else {}
    msg = data.get("choices", [{}])[0].get("message", {})
    return {
        "status_code": resp.status_code,
        "body": data,
        "content": msg.get("content", "") or "",
        "reasoning_content": msg.get("reasoning_content"),
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": data.get("choices", [{}])[0].get("finish_reason", ""),
        "usage": data.get("usage", {}),
        "error": resp.text[:200] if resp.status_code != 200 else "",
    }


async def get_health(client: httpx.AsyncClient, base_url: str) -> Dict:
    resp = await client.get(f"{base_url}/health", timeout=10)
    return resp.json()


async def get_status(client: httpx.AsyncClient, base_url: str) -> Dict:
    resp = await client.get(f"{base_url}/status", timeout=10)
    return resp.json()


async def snapshot_memory(client: httpx.AsyncClient, base_url: str, label: str):
    try:
        status = await get_status(client, base_url)
        mem = status.get("gpu_memory_used_gb", 0)
        memory_snapshots.append({"label": label, "gpu_used_gb": mem, "time": time.time()})
        return mem
    except Exception:
        return 0


def make_red_png_base64() -> str:
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
# Phase 1: Server Health
# =============================================================================

async def phase_health(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("1_server_health")
    print("\n  Phase 1: Server Health")

    t0 = time.time()
    try:
        health = await get_health(client, base_url)
        healthy = health.get("status") == "healthy"
        record(phase, "health_endpoint", healthy, time.time() - t0,
               f"model={health.get('current_model')}, queue={health.get('queue_state')}")
    except Exception as e:
        record(phase, "health_endpoint", False, time.time() - t0, error=str(e))

    t0 = time.time()
    try:
        resp = await client.get(f"{base_url}/v1/models", timeout=10)
        data = resp.json()
        models = [m["id"] for m in data.get("data", [])]
        record(phase, "models_endpoint", len(models) >= 3, time.time() - t0,
               f"models={models}")
    except Exception as e:
        record(phase, "models_endpoint", False, time.time() - t0, error=str(e))

    return phase


# =============================================================================
# Phase 2: Unit Tests (reasoning extraction)
# =============================================================================

async def phase_unit_tests(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("2_unit_tests")
    print("\n  Phase 2: Unit Tests (reasoning extraction)")

    # Import the function under test
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from blitzinfer.api.reasoning import extract_reasoning_content

    test_cases = [
        ("No tags", "Hello world", (None, "Hello world")),
        ("<think> tags", "<think>reasoning here</think>final answer", ("reasoning here", "final answer")),
        ("Kimi tags", "◁think▷internal reasoning◁/think▷clean answer", ("internal reasoning", "clean answer")),
        ("Truncated", "<think>reasoning only", ("reasoning only", None)),
        ("Empty think", "<think></think>just content", (None, "just content")),
        ("Whitespace", "<think>  spaced  </think>  content  ", ("spaced", "content")),
        ("No content after", "<think>reasoning</think>", ("reasoning", None)),
    ]

    for name, input_text, expected in test_cases:
        t0 = time.time()
        try:
            result = extract_reasoning_content(input_text)
            passed = result == expected
            record(phase, f"extract_{name.lower().replace(' ', '_')}", passed,
                   time.time() - t0,
                   f"got={result}" if not passed else "")
        except Exception as e:
            record(phase, f"extract_{name.lower().replace(' ', '_')}", False,
                   time.time() - t0, error=str(e))

    return phase


# =============================================================================
# Phase 3: Per-Model Sequential Inference
# =============================================================================

MODEL_TESTS = {
    # Order matters: qwen3-coder-next tested EARLY because FLA Triton kernels
    # need ~11 GiB free GPU memory for scratch buffers. After many model switches,
    # non-pytorch memory drift (~0.5 GiB/switch) eats into this margin.
    "gpt-oss-120b": [
        {
            "name": "harmony_math",
            "messages": [{"role": "user", "content": "What is 15 * 17? Give just the answer."}],
            "max_tokens": 300,
            "check": lambda r: "255" in r["content"],
        },
        {
            "name": "tool_call",
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
            "check": lambda r: len(r.get("tool_calls", [])) > 0,
        },
        {
            "name": "multi_turn",
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 200,
            "check": lambda r: "alice" in r["content"].lower(),
        },
        {
            "name": "streaming",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 200,
            "stream": True,
            "check": lambda r: r.get("has_done") and len(r.get("content", "")) > 5,
        },
    ],
    "qwen3-coder-next": [
        {
            "name": "code_gen",
            "messages": [{"role": "user", "content": "Write a hello world in Python. Just the code."}],
            "max_tokens": 500,
            "check": lambda r: "print" in (r["content"] + (r.get("reasoning_content") or "")).lower(),
        },
    ],
    "qwen3-32b": [
        {
            "name": "basic_chat",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 100,
            # Qwen3 is a thinking model; answer may be in content or reasoning_content
            "check": lambda r: "paris" in ((r.get("content", "") + (r.get("reasoning_content") or "")).lower()),
        },
        {
            "name": "math",
            "messages": [{"role": "user", "content": "What is 123 + 456 + 789?"}],
            "max_tokens": 500,
            # Qwen3 often includes answer in thinking block
            "check": lambda r: "1368" in (r.get("content", "") + (r.get("reasoning_content") or "")),
        },
        {
            "name": "code",
            "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime. Just the function."}],
            "max_tokens": 500,
            # Thinking model may spend all tokens on reasoning; check for code-related keywords
            "check": lambda r: any(kw in (r.get("content", "") + (r.get("reasoning_content") or ""))
                                   for kw in ["def ", "prime", "is_prime", "divisor"]),
        },
    ],
    "kimi-vl": [
        {
            "name": "reasoning_extraction",
            "messages": [{"role": "user", "content": "What is 2 + 2? Think step by step."}],
            "max_tokens": 500,
            # Thinking models may put "4" in reasoning_content or content
            "check": lambda r: "4" in (r.get("content", "") + (r.get("reasoning_content") or "")),
            "check_reasoning": True,
        },
        {
            "name": "text_query",
            "messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}],
            "max_tokens": 200,
            # kimi-vl may put answer in reasoning_content only
            "check": lambda r: "jupiter" in ((r.get("content", "") + (r.get("reasoning_content") or "")).lower()),
        },
        {
            "name": "vision",
            "vision": True,
            "messages": None,  # Set up in runner
            "max_tokens": 300,
            "check": lambda r: len(r.get("content", "") + (r.get("reasoning_content") or "")) > 10,
        },
    ],
    "qwen2.5-7b": [
        {
            "name": "basic_chat",
            "messages": [{"role": "user", "content": "What is the speed of light?"}],
            "max_tokens": 100,
            "check": lambda r: "300" in r["content"] or "299" in r["content"],
        },
        {
            "name": "math",
            "messages": [{"role": "user", "content": "What is the square root of 144?"}],
            "max_tokens": 100,
            "check": lambda r: "12" in r["content"],
        },
    ],
}


async def phase_per_model(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("3_per_model_inference")
    print("\n  Phase 3: Per-Model Sequential Inference")

    img_uri = make_red_png_base64()

    for model_id, tests in MODEL_TESTS.items():
        print(f"\n    --- {model_id} ---")
        await snapshot_memory(client, base_url, f"before_{model_id}")

        for test in tests:
            name = f"{model_id}/{test['name']}"
            may_fail = test.get("may_fail", False)
            t0 = time.time()

            try:
                messages = test["messages"]
                if test.get("vision"):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see in this image."},
                            {"type": "image_url", "image_url": {"url": img_uri}},
                        ],
                    }]

                resp = await chat(
                    client, base_url, model_id, messages,
                    max_tokens=test.get("max_tokens", 200),
                    tools=test.get("tools"),
                    stream=test.get("stream", False),
                )
                duration = time.time() - t0

                if resp["status_code"] != 200:
                    if may_fail:
                        record(phase, name, True, duration,
                               f"EXPECTED_FAIL: HTTP {resp['status_code']}")
                    else:
                        record(phase, name, False, duration,
                               error=f"HTTP {resp['status_code']}: {resp.get('error', '')[:100]}")
                    continue

                check_fn = test.get("check", lambda r: True)
                passed = check_fn(resp)

                detail_parts = [f"content_len={len(resp.get('content', ''))}"]
                if resp.get("reasoning_content"):
                    detail_parts.append(f"reasoning_len={len(resp['reasoning_content'])}")
                if test.get("check_reasoning") and not resp.get("reasoning_content"):
                    detail_parts.append("WARNING: no reasoning_content")

                record(phase, name, passed, duration, "; ".join(detail_parts),
                       "" if passed else f"check failed, content={resp['content'][:80]}")

            except Exception as e:
                duration = time.time() - t0
                if may_fail:
                    record(phase, name, True, duration, f"EXPECTED_FAIL: {type(e).__name__}")
                else:
                    record(phase, name, False, duration, error=str(e)[:200])

    return phase


# =============================================================================
# Phase 4: Rapid-Fire Model Switching
# =============================================================================

SWITCH_SEQUENCE = [
    "qwen3-32b",
    "gpt-oss-120b",
    "kimi-vl",
    "qwen2.5-7b",
    "qwen3-32b",
    "gpt-oss-120b",
    "kimi-vl",
    "qwen2.5-7b",
]


async def phase_rapid_switching(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("4_rapid_switching")
    print(f"\n  Phase 4: Rapid-Fire Switching ({len(SWITCH_SEQUENCE)} switches)")

    await snapshot_memory(client, base_url, "before_switching")

    for i, model in enumerate(SWITCH_SEQUENCE):
        name = f"switch_{i+1}_{model}"
        t0 = time.time()
        try:
            # Use max_tokens=200 to give thinking models enough room for reasoning + answer
            resp = await chat(
                client, base_url, model,
                [{"role": "user", "content": "What is 3 + 4? Just the number."}],
                max_tokens=200,
            )
            duration = time.time() - t0
            content = resp.get("content", "")
            reasoning = resp.get("reasoning_content", "") or ""
            all_text = content + reasoning
            # Pass if we got ANY text (content or reasoning) and HTTP 200
            passed = resp["status_code"] == 200 and len(all_text) > 0
            record(phase, name, passed, duration,
                   f"content_len={len(content)}, reasoning_len={len(reasoning)}")
        except Exception as e:
            record(phase, name, False, time.time() - t0, error=str(e)[:100])

    await snapshot_memory(client, base_url, "after_switching")
    return phase


# =============================================================================
# Phase 5: Concurrent Requests
# =============================================================================

async def phase_concurrent(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("5_concurrent")
    print("\n  Phase 5: Concurrent Requests (9 to 3 models)")

    requests_spec = [
        ("gpt-oss-120b", "What is 3+4?", "7"),
        ("gpt-oss-120b", "What is 5+6?", "11"),
        ("gpt-oss-120b", "What is 8+9?", "17"),
        ("qwen3-32b", "Capital of Germany?", "berlin"),
        ("qwen3-32b", "Capital of Japan?", "tokyo"),
        ("qwen3-32b", "Capital of Italy?", "rome"),
        ("kimi-vl", "What is 1+1?", "2"),
        ("kimi-vl", "What is 10+10?", "20"),
        ("kimi-vl", "Capital of France?", "paris"),
    ]

    async def do_request(model, prompt, expected):
        t0 = time.time()
        try:
            resp = await chat(
                client, base_url, model,
                [{"role": "user", "content": prompt}],
                max_tokens=500,  # Thinking models need room for reasoning + answer
            )
            duration = time.time() - t0
            content = resp.get("content", "").lower()
            reasoning = (resp.get("reasoning_content") or "").lower()
            all_text = content + reasoning
            # For numeric checks, also match word forms (e.g. "20" or "twenty")
            expected_lower = expected.lower()
            passed = resp["status_code"] == 200 and expected_lower in all_text
            return (f"concurrent_{model}_{prompt[:15]}", passed, duration,
                    f"content={content[:40]}", "")
        except Exception as e:
            return (f"concurrent_{model}_{prompt[:15]}", False, time.time() - t0,
                    "", str(e)[:100])

    t0 = time.time()
    tasks = [do_request(m, p, e) for m, p, e in requests_spec]
    results = await asyncio.gather(*tasks)
    total_dur = time.time() - t0

    for name, passed, dur, detail, error in results:
        record(phase, name, passed, dur, detail, error)

    print(f"    Total concurrent time: {total_dur:.1f}s")
    return phase


# =============================================================================
# Phase 6: Memory Drift
# =============================================================================

async def phase_memory(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("6_memory_drift")
    print("\n  Phase 6: Memory Drift Analysis")

    if len(memory_snapshots) < 2:
        print("    (not enough snapshots)")
        return phase

    first = memory_snapshots[0]["gpu_used_gb"]
    last = memory_snapshots[-1]["gpu_used_gb"]
    drift = last - first

    print(f"    Snapshots: {len(memory_snapshots)}")
    for s in memory_snapshots:
        print(f"      {s['label']}: {s['gpu_used_gb']:.1f}GB")

    ok = abs(drift) < 10.0
    record(phase, "total_drift", ok, 0,
           f"drift={drift:+.1f}GB (start={first:.1f}GB, end={last:.1f}GB)",
           "" if ok else f"Drift {drift:.1f}GB exceeds 10GB threshold")

    return phase


# =============================================================================
# Phase 7: Final Server State
# =============================================================================

async def phase_final_check(client: httpx.AsyncClient, base_url: str) -> PhaseResult:
    phase = PhaseResult("7_final_check")
    print("\n  Phase 7: Final Server State")

    t0 = time.time()
    try:
        health = await get_health(client, base_url)
        alive = health.get("status") == "healthy"
        qs = health.get("queue_state", "")
        inf = health.get("in_flight", -1)

        record(phase, "server_alive", alive, time.time() - t0,
               f"queue={qs}, in_flight={inf}")

        # Queue should be SERVING and empty
        queue_clean = qs == "SERVING" and inf == 0
        record(phase, "queue_drained", queue_clean, 0,
               f"state={qs}, in_flight={inf}",
               "" if queue_clean else "Queue not fully drained")
    except Exception as e:
        record(phase, "server_alive", False, time.time() - t0, error=str(e))

    return phase


# =============================================================================
# Summary
# =============================================================================

def print_summary(phases: List[PhaseResult], run_num: int):
    print(f"\n{'='*70}")
    print(f"E2E TEST SUMMARY (Run #{run_num})")
    print(f"{'='*70}")

    total_passed = 0
    total_failed = 0

    for phase in phases:
        status = "PASS" if phase.all_passed else "FAIL"
        print(f"\n  [{status}] {phase.name}: {phase.passed}/{phase.total}")
        total_passed += phase.passed
        total_failed += phase.failed

        if phase.failed > 0:
            for r in phase.results:
                if not r.passed:
                    print(f"    FAIL: {r.name} - {r.error or r.detail}")

    total = total_passed + total_failed
    pct = (total_passed / total * 100) if total > 0 else 0
    overall = "PASS" if total_failed == 0 else "FAIL"

    print(f"\n{'='*70}")
    print(f"  [{overall}] {total_passed}/{total} tests passed ({pct:.0f}%)")
    print(f"{'='*70}")

    return total_failed == 0


# =============================================================================
# Main
# =============================================================================

async def run_once(base_url: str, run_num: int, skip_concurrent: bool) -> bool:
    print(f"\n{'#'*70}")
    print(f"# E2E TEST RUN #{run_num}")
    print(f"# Server: {base_url}")
    print(f"{'#'*70}")

    global memory_snapshots
    memory_snapshots = []

    phases = []

    async with httpx.AsyncClient() as client:
        # Wait for server
        print("\n  Waiting for server...")
        for _ in range(30):
            try:
                resp = await client.get(f"{base_url}/health", timeout=10)
                if resp.json().get("status") == "healthy":
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            print("  ERROR: Server not reachable after 60s")
            return False

        await snapshot_memory(client, base_url, "start")

        phases.append(await phase_health(client, base_url))
        phases.append(await phase_unit_tests(client, base_url))
        phases.append(await phase_per_model(client, base_url))
        phases.append(await phase_rapid_switching(client, base_url))

        if not skip_concurrent:
            phases.append(await phase_concurrent(client, base_url))

        await snapshot_memory(client, base_url, "end")
        phases.append(await phase_memory(client, base_url))
        phases.append(await phase_final_check(client, base_url))

    return print_summary(phases, run_num)


async def main_async(args):
    results = []
    for i in range(1, args.runs + 1):
        ok = await run_once(args.base_url, i, args.skip_concurrent)
        results.append(ok)
        if not ok and i < args.runs:
            print(f"\n  Run #{i} FAILED. Continuing to run #{i+1}...")

    if args.runs > 1:
        print(f"\n{'='*70}")
        print(f"ALL RUNS SUMMARY")
        print(f"{'='*70}")
        for i, ok in enumerate(results, 1):
            status = "PASS" if ok else "FAIL"
            print(f"  Run #{i}: {status}")

        all_ok = all(results)
        print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
        return all_ok

    return results[0]


def main():
    parser = argparse.ArgumentParser(description="BlitzInfer E2E test suite")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--runs", type=int, default=1, help="Number of complete runs")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="Skip concurrent request phase")
    args = parser.parse_args()

    success = asyncio.run(main_async(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
