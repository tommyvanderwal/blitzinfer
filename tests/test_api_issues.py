#!/usr/bin/env python3
"""
Test suite for BlitzInfer API - covers all known issues.

Run with: pytest tests/test_api_issues.py -v
Or directly: python tests/test_api_issues.py

Tests cover:
1. Basic chat completion (non-streaming)
2. Streaming chat completion (SSE format)
3. Tool/function calling with GPT-OSS Harmony
4. Tool call streaming
5. Model listing
6. Model switching
7. Conversation history with tool results
8. Disabled models (GLM) should 404
"""

import json
import requests
import sys
import time
from typing import Optional

# Configuration
BASE_URL = "http://192.168.2.90:8000"
TIMEOUT = 120  # seconds

# Test results tracking
results = []


def log_result(test_name: str, passed: bool, message: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((test_name, passed, message))
    print(f"{status}: {test_name}")
    if message:
        print(f"       {message}")


def test_health_endpoint():
    """Test /health endpoint is responsive."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log_result("health_endpoint", True, f"current_model={data.get('current_model')}")
            return True
        else:
            log_result("health_endpoint", False, f"status={resp.status_code}")
            return False
    except Exception as e:
        log_result("health_endpoint", False, str(e))
        return False


def test_models_list():
    """Test /v1/models endpoint returns available models."""
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            # GLM should NOT be in the list (disabled)
            if "glm-4.6v-flash" in models:
                log_result("models_list", False, "GLM should be disabled but is listed")
                return False
            log_result("models_list", True, f"models={models}")
            return True
        else:
            log_result("models_list", False, f"status={resp.status_code}")
            return False
    except Exception as e:
        log_result("models_list", False, str(e))
        return False


def test_disabled_model_404():
    """Test that disabled models (GLM) return 404."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "glm-4.6v-flash",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        if resp.status_code == 404:
            log_result("disabled_model_404", True, "GLM correctly returns 404")
            return True
        else:
            log_result("disabled_model_404", False, f"Expected 404, got {resp.status_code}")
            return False
    except Exception as e:
        log_result("disabled_model_404", False, str(e))
        return False


def test_basic_chat_gptoss(model: str = "gpt-oss-120b"):
    """Test basic non-streaming chat completion with GPT-OSS."""
    try:
        # GPT-OSS uses Harmony format which requires sufficient tokens for both
        # reasoning (analysis channel) and final response (final channel).
        # 50 tokens is not enough - the model gets truncated mid-reasoning.
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'hello test' and nothing else."}],
                "max_tokens": 200,  # Enough for reasoning + final response
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            # Check for TRUE Harmony corruption (malformed token sequences, not model saying "analysis" as a word)
            # Real corruption: "assistantanalysis" (no space), "to=functions.", "<|channel|>" as literal text
            corruption_patterns = [
                "assistantanalysis",  # Fused channel marker
                "assistantfinal",     # Fused channel marker
                "assistantcommentary", # Fused channel marker
                "<|channel|>",        # Literal token text
                "<|message|>",        # Literal token text
                "<|end|>",            # Literal token text
                "to=functions.",      # Tool call marker as text
            ]
            for pattern in corruption_patterns:
                if pattern in content:
                    log_result(f"basic_chat_{model}", False, f"Harmony corruption '{pattern}' in: {content[:100]}")
                    return False
            log_result(f"basic_chat_{model}", True, f"response={content[:50]}...")
            return True
        else:
            log_result(f"basic_chat_{model}", False, f"status={resp.status_code}, body={resp.text[:200]}")
            return False
    except Exception as e:
        log_result(f"basic_chat_{model}", False, str(e))
        return False


def test_streaming_chat(model: str = "gpt-oss-120b"):
    """Test streaming chat completion returns proper SSE format."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "max_tokens": 100,
                "stream": True,
            },
            timeout=TIMEOUT,
            stream=True,
        )
        if resp.status_code != 200:
            log_result(f"streaming_chat_{model}", False, f"status={resp.status_code}")
            return False

        chunks = []
        content = ""
        has_done = False
        has_finish_reason = False

        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        has_done = True
                        continue
                    try:
                        chunk = json.loads(data)
                        chunks.append(chunk)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content += delta["content"]
                        if chunk["choices"][0].get("finish_reason"):
                            has_finish_reason = True
                    except json.JSONDecodeError:
                        pass

        if not has_done:
            log_result(f"streaming_chat_{model}", False, "Missing [DONE] marker")
            return False
        if not has_finish_reason:
            log_result(f"streaming_chat_{model}", False, "Missing finish_reason")
            return False

        # Check for TRUE Harmony corruption
        corruption_patterns = ["assistantanalysis", "assistantfinal", "<|channel|>", "to=functions."]
        for pattern in corruption_patterns:
            if pattern in content:
                log_result(f"streaming_chat_{model}", False, f"Harmony corruption '{pattern}': {content[:100]}")
                return False

        log_result(f"streaming_chat_{model}", True, f"chunks={len(chunks)}, content={content[:50]}...")
        return True

    except Exception as e:
        log_result(f"streaming_chat_{model}", False, str(e))
        return False


def test_tool_call_non_streaming():
    """Test tool calling returns proper format (non-streaming)."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": "Get the weather in Paris"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "City name"},
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "max_tokens": 200,
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            log_result("tool_call_non_streaming", False, f"status={resp.status_code}")
            return False

        data = resp.json()
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        finish_reason = data["choices"][0].get("finish_reason")

        if not tool_calls:
            # Model might respond with text instead of tool call - check for corruption
            content = message.get("content", "")
            if "to=functions" in content or "assistantanalysis" in content:
                log_result("tool_call_non_streaming", False, f"Tool call not parsed: {content[:100]}")
                return False
            log_result("tool_call_non_streaming", False, "No tool_calls in response (model declined)")
            return False

        # Verify tool call format
        tc = tool_calls[0]
        if not tc.get("id") or not tc.get("function", {}).get("name"):
            log_result("tool_call_non_streaming", False, f"Invalid tool call format: {tc}")
            return False

        if finish_reason != "tool_calls":
            log_result("tool_call_non_streaming", False, f"finish_reason should be 'tool_calls', got '{finish_reason}'")
            return False

        log_result("tool_call_non_streaming", True, f"tool={tc['function']['name']}, args={tc['function']['arguments'][:50]}")
        return True

    except Exception as e:
        log_result("tool_call_non_streaming", False, str(e))
        return False


def test_tool_call_streaming():
    """Test tool calling with streaming returns proper SSE format."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": "Search for Python documentation"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search the web",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                },
                                "required": ["query"],
                            },
                        },
                    }
                ],
                "max_tokens": 200,
                "stream": True,
            },
            timeout=TIMEOUT,
            stream=True,
        )
        if resp.status_code != 200:
            log_result("tool_call_streaming", False, f"status={resp.status_code}")
            return False

        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""
        finish_reason = None

        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            tc = delta["tool_calls"][0]
                            if "id" in tc:
                                tool_call_id = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_call_name = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_call_args += tc["function"]["arguments"]
                        fr = chunk["choices"][0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    except json.JSONDecodeError:
                        pass

        if not tool_call_id or not tool_call_name:
            log_result("tool_call_streaming", False, "No tool call found in stream")
            return False

        if finish_reason != "tool_calls":
            log_result("tool_call_streaming", False, f"finish_reason should be 'tool_calls', got '{finish_reason}'")
            return False

        log_result("tool_call_streaming", True, f"tool={tool_call_name}, args={tool_call_args[:50]}")
        return True

    except Exception as e:
        log_result("tool_call_streaming", False, str(e))
        return False


def test_tool_result_conversation():
    """Test multi-turn conversation with tool results."""
    try:
        # First turn: user asks, assistant calls tool
        messages = [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 22°C"},
        ]

        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-120b",
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        },
                    }
                ],
                "max_tokens": 200,
                "stream": False,
            },
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            log_result("tool_result_conversation", False, f"status={resp.status_code}, body={resp.text[:200]}")
            return False

        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")

        # Should mention Tokyo and/or the weather
        if not content:
            log_result("tool_result_conversation", False, "Empty response")
            return False

        # Check for corruption
        if "assistantanalysis" in content.lower() or "to=functions" in content:
            log_result("tool_result_conversation", False, f"Harmony corruption: {content[:100]}")
            return False

        log_result("tool_result_conversation", True, f"response={content[:80]}...")
        return True

    except Exception as e:
        log_result("tool_result_conversation", False, str(e))
        return False


def test_model_switch():
    """Test switching between models works."""
    try:
        # First request to gpt-oss
        resp1 = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": "Say 'model1'"}],
                "max_tokens": 20,
            },
            timeout=TIMEOUT,
        )
        if resp1.status_code != 200:
            log_result("model_switch", False, f"First request failed: {resp1.status_code}")
            return False

        # Switch to qwen
        resp2 = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "Say 'model2'"}],
                "max_tokens": 20,
            },
            timeout=TIMEOUT,
        )
        if resp2.status_code != 200:
            log_result("model_switch", False, f"Second request (switch) failed: {resp2.status_code}")
            return False

        # Verify health shows new model
        health = requests.get(f"{BASE_URL}/health", timeout=10).json()
        if health.get("current_model") != "qwen3-32b":
            log_result("model_switch", False, f"Model not switched: {health.get('current_model')}")
            return False

        log_result("model_switch", True, "Switched gpt-oss -> qwen3-32b")
        return True

    except Exception as e:
        log_result("model_switch", False, str(e))
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, msg in results:
        status = "✅" if p else "❌"
        print(f"  {status} {name}")

    print("=" * 60)
    print(f"TOTAL: {passed}/{total} passed")
    if passed < total:
        print("\nFailed tests need investigation!")
    print("=" * 60)

    return passed == total


def main():
    """Run all tests."""
    print("=" * 60)
    print("BlitzInfer API Test Suite")
    print(f"Target: {BASE_URL}")
    print("=" * 60 + "\n")

    # Check server is up first
    if not test_health_endpoint():
        print("\n❌ Server not responding, aborting tests")
        return 1

    print()

    # Run tests
    test_models_list()
    test_disabled_model_404()
    print()

    test_basic_chat_gptoss()
    test_streaming_chat()
    print()

    test_tool_call_non_streaming()
    test_tool_call_streaming()
    test_tool_result_conversation()
    print()

    # Model switch is slow, run last
    # test_model_switch()

    all_passed = print_summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
