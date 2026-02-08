#!/usr/bin/env python3
"""
Comprehensive Test Suite for BlitzInfer API

Tests all available models with:
1. First load: Basic "HI" test (warm-up, verify model loads)
2. Second load: Full edge case test suite

Run with: python tests/test_comprehensive.py

Success criteria: 100% pass rate after single server start.
"""

import base64
import json
import os
import requests
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configuration
BASE_URL = "http://192.168.2.90:8000"
TIMEOUT = 180  # seconds per request
TEST_IMAGE_PATH = Path(__file__).parent / "test_image.png"

# Results tracking
@dataclass
class TestResult:
    model: str
    test_name: str
    passed: bool
    message: str = ""
    duration: float = 0.0

results: List[TestResult] = []
current_model: str = ""


def log_result(test_name: str, passed: bool, message: str = "", duration: float = 0.0):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append(TestResult(current_model, test_name, passed, message, duration))
    print(f"  {status}: {test_name}")
    if message:
        # Truncate long messages
        msg = message[:200] + "..." if len(message) > 200 else message
        print(f"         {msg}")


def make_chat_request(
    model: str,
    messages: List[Dict],
    max_tokens: int = 200,
    stream: bool = False,
    tools: Optional[List[Dict]] = None,
    temperature: float = 0.7,
) -> Tuple[int, Any]:
    """Make a chat completion request."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
            stream=stream,
        )
        duration = time.time() - start

        if stream:
            return resp.status_code, resp, duration
        return resp.status_code, resp.json() if resp.status_code == 200 else resp.text, duration
    except Exception as e:
        return 0, str(e), time.time() - start


def make_vision_request(
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int = 300,
) -> Tuple[int, Any, float]:
    """Make a vision request with an image."""
    # Encode image as base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            }
        ]
    }]

    return make_chat_request(model, messages, max_tokens=max_tokens)


def get_content(response: Any) -> str:
    """Extract content from response."""
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ""


def stream_to_content(response) -> Tuple[str, bool, Optional[str]]:
    """Parse streaming response to content."""
    content = ""
    has_done = False
    finish_reason = None

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    has_done = True
                    continue
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content += delta["content"]
                    fr = chunk["choices"][0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                except json.JSONDecodeError:
                    pass

    return content, has_done, finish_reason


# =============================================================================
# Model-Specific Test Suites
# =============================================================================

def test_basic_hi(model: str) -> bool:
    """Basic HI test - verify model loads and responds."""
    status, resp, dur = make_chat_request(model, [{"role": "user", "content": "Hi"}])
    if status != 200:
        log_result("basic_hi", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if not content or len(content) < 2:
        log_result("basic_hi", False, "Empty or too short response", dur)
        return False

    log_result("basic_hi", True, f"response={content[:50]}...", dur)
    return True


# -----------------------------------------------------------------------------
# GPT-OSS-120B Edge Cases
# -----------------------------------------------------------------------------

def test_gptoss_harmony_reasoning(model: str) -> bool:
    """Test that GPT-OSS uses Harmony format with proper reasoning."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What is 15 * 17? Show your work."}],
        max_tokens=500,
    )
    if status != 200:
        log_result("harmony_reasoning", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should have the answer (255)
    if "255" not in content:
        log_result("harmony_reasoning", False, f"Wrong answer: {content[:100]}", dur)
        return False

    # Should NOT have raw Harmony markers
    corruption = ["assistantanalysis", "assistantfinal", "<|channel|>", "<|message|>"]
    for pattern in corruption:
        if pattern in content:
            log_result("harmony_reasoning", False, f"Harmony corruption: {pattern}", dur)
            return False

    log_result("harmony_reasoning", True, f"Correct: 255 in response", dur)
    return True


def test_gptoss_tool_calling(model: str) -> bool:
    """Test GPT-OSS tool calling."""
    tools = [{
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
    }]

    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        max_tokens=300,
    )
    if status != 200:
        log_result("tool_calling", False, f"HTTP {status}", dur)
        return False

    message = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls")

    if not tool_calls:
        log_result("tool_calling", False, "No tool calls returned", dur)
        return False

    tc = tool_calls[0]
    if tc.get("function", {}).get("name") != "get_weather":
        log_result("tool_calling", False, f"Wrong function: {tc}", dur)
        return False

    # Check arguments contain Tokyo
    args = tc.get("function", {}).get("arguments", "")
    if "Tokyo" not in args and "tokyo" not in args.lower():
        log_result("tool_calling", False, f"Tokyo not in args: {args}", dur)
        return False

    log_result("tool_calling", True, f"tool={tc['function']['name']}", dur)
    return True


def test_gptoss_streaming(model: str) -> bool:
    """Test GPT-OSS streaming output."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Count from 1 to 10."}],
        max_tokens=200,
        stream=True,
    )
    if status != 200:
        log_result("streaming", False, f"HTTP {status}", dur)
        return False

    content, has_done, finish_reason = stream_to_content(resp)

    if not has_done:
        log_result("streaming", False, "Missing [DONE] marker", dur)
        return False

    # Check numbers are present
    found_count = sum(1 for i in range(1, 11) if str(i) in content)
    if found_count < 8:
        log_result("streaming", False, f"Only found {found_count}/10 numbers", dur)
        return False

    log_result("streaming", True, f"Found {found_count}/10 numbers", dur)
    return True


def test_gptoss_multi_turn(model: str) -> bool:
    """Test GPT-OSS multi-turn conversation."""
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]

    status, resp, dur = make_chat_request(model, messages, max_tokens=100)
    if status != 200:
        log_result("multi_turn", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    if "alice" not in content:
        log_result("multi_turn", False, f"Didn't remember name: {content[:100]}", dur)
        return False

    log_result("multi_turn", True, "Remembered name", dur)
    return True


def test_gptoss_unicode(model: str) -> bool:
    """Test GPT-OSS with unicode/special characters."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Translate 'hello' to Japanese, Chinese, and Korean."}],
        max_tokens=200,
    )
    if status != 200:
        log_result("unicode", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should contain some CJK characters
    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af'
                  or '\u3040' <= c <= '\u30ff' for c in content)
    if not has_cjk:
        log_result("unicode", False, f"No CJK chars: {content[:100]}", dur)
        return False

    log_result("unicode", True, f"Contains CJK characters", dur)
    return True


# -----------------------------------------------------------------------------
# Qwen3-32B Edge Cases (Text Model, FP8)
# -----------------------------------------------------------------------------

def test_qwen_basic_chat(model: str) -> bool:
    """Test Qwen basic chat."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("basic_chat", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    if "paris" not in content:
        log_result("basic_chat", False, f"Wrong answer: {content[:100]}", dur)
        return False

    log_result("basic_chat", True, "Correct: Paris", dur)
    return True


def test_qwen_math(model: str) -> bool:
    """Test Qwen math capabilities (FP8 precision test)."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Calculate: 123 + 456 + 789 = ?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("math", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if "1368" not in content:
        log_result("math", False, f"Wrong: {content[:100]} (expected 1368)", dur)
        return False

    log_result("math", True, "Correct: 1368", dur)
    return True


def test_qwen_code(model: str) -> bool:
    """Test Qwen code generation."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Write a Python function to check if a number is prime. Just the function, no explanation."}],
        max_tokens=300,
    )
    if status != 200:
        log_result("code", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should contain function definition
    if "def " not in content or "prime" not in content.lower():
        log_result("code", False, f"Not a function: {content[:100]}", dur)
        return False

    log_result("code", True, "Generated prime function", dur)
    return True


def test_qwen_streaming(model: str) -> bool:
    """Test Qwen streaming."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "List 5 colors."}],
        max_tokens=100,
        stream=True,
    )
    if status != 200:
        log_result("streaming", False, f"HTTP {status}", dur)
        return False

    content, has_done, _ = stream_to_content(resp)

    if not has_done:
        log_result("streaming", False, "Missing [DONE]", dur)
        return False

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "pink"]
    found = sum(1 for c in colors if c in content.lower())
    if found < 3:
        log_result("streaming", False, f"Only {found} colors found", dur)
        return False

    log_result("streaming", True, f"Found {found} colors", dur)
    return True


def test_qwen_long_context(model: str) -> bool:
    """Test Qwen with longer context."""
    # Create a moderately long context
    context = "The following is a list of items: " + ", ".join([f"item{i}" for i in range(100)])
    context += ". How many items are in the list?"

    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": context}],
        max_tokens=100,
    )
    if status != 200:
        log_result("long_context", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if "100" not in content:
        log_result("long_context", False, f"Wrong count: {content[:100]}", dur)
        return False

    log_result("long_context", True, "Counted 100 items", dur)
    return True


# -----------------------------------------------------------------------------
# Mistral-Small-24B Edge Cases
# -----------------------------------------------------------------------------

def test_mistral_basic_chat(model: str) -> bool:
    """Test Mistral basic chat."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What year did World War 2 end?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("basic_chat", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if "1945" not in content:
        log_result("basic_chat", False, f"Wrong: {content[:100]}", dur)
        return False

    log_result("basic_chat", True, "Correct: 1945", dur)
    return True


def test_mistral_reasoning(model: str) -> bool:
    """Test Mistral reasoning."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"}],
        max_tokens=200,
    )
    if status != 200:
        log_result("reasoning", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    # Should recognize this is a logical fallacy
    if "no" not in content and "cannot" not in content and "can't" not in content:
        log_result("reasoning", False, f"Didn't recognize fallacy: {content[:100]}", dur)
        return False

    log_result("reasoning", True, "Recognized logical issue", dur)
    return True


def test_mistral_streaming(model: str) -> bool:
    """Test Mistral streaming."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Name 5 planets."}],
        max_tokens=100,
        stream=True,
    )
    if status != 200:
        log_result("streaming", False, f"HTTP {status}", dur)
        return False

    content, has_done, _ = stream_to_content(resp)
    if not has_done:
        log_result("streaming", False, "Missing [DONE]", dur)
        return False

    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    found = sum(1 for p in planets if p in content.lower())
    if found < 4:
        log_result("streaming", False, f"Only {found} planets", dur)
        return False

    log_result("streaming", True, f"Found {found} planets", dur)
    return True


def test_mistral_format(model: str) -> bool:
    """Test Mistral formatting output."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Create a markdown table with 3 columns: Name, Age, City. Add 2 example rows."}],
        max_tokens=200,
    )
    if status != 200:
        log_result("format", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should contain table elements
    if "|" not in content or "---" not in content:
        log_result("format", False, f"No table: {content[:100]}", dur)
        return False

    log_result("format", True, "Created markdown table", dur)
    return True


def test_mistral_multilingual(model: str) -> bool:
    """Test Mistral multilingual capability."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Say 'thank you' in Spanish, French, and German."}],
        max_tokens=150,
    )
    if status != 200:
        log_result("multilingual", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    # Check for translations
    found = 0
    if "gracias" in content: found += 1
    if "merci" in content: found += 1
    if "danke" in content: found += 1

    if found < 2:
        log_result("multilingual", False, f"Only {found}/3 translations", dur)
        return False

    log_result("multilingual", True, f"Found {found}/3 translations", dur)
    return True


# -----------------------------------------------------------------------------
# Llama-3.1-70B Edge Cases (AWQ Quantized)
# -----------------------------------------------------------------------------

def test_llama_basic_chat(model: str) -> bool:
    """Test Llama basic chat."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("basic_chat", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    if "shakespeare" not in content:
        log_result("basic_chat", False, f"Wrong: {content[:100]}", dur)
        return False

    log_result("basic_chat", True, "Correct: Shakespeare", dur)
    return True


def test_llama_math_awq(model: str) -> bool:
    """Test Llama math with AWQ quantization."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What is 7 * 8 * 9?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("math_awq", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if "504" not in content:
        log_result("math_awq", False, f"Wrong: {content[:100]} (expected 504)", dur)
        return False

    log_result("math_awq", True, "Correct: 504", dur)
    return True


def test_llama_streaming(model: str) -> bool:
    """Test Llama streaming."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "List the days of the week."}],
        max_tokens=100,
        stream=True,
    )
    if status != 200:
        log_result("streaming", False, f"HTTP {status}", dur)
        return False

    content, has_done, _ = stream_to_content(resp)
    if not has_done:
        log_result("streaming", False, "Missing [DONE]", dur)
        return False

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    found = sum(1 for d in days if d in content.lower())
    if found < 5:
        log_result("streaming", False, f"Only {found} days", dur)
        return False

    log_result("streaming", True, f"Found {found}/7 days", dur)
    return True


def test_llama_instruction_following(model: str) -> bool:
    """Test Llama instruction following."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Reply with exactly one word: the color of grass."}],
        max_tokens=50,
    )
    if status != 200:
        log_result("instruction", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).strip().lower()
    # Should be a short response with "green"
    if "green" not in content:
        log_result("instruction", False, f"Wrong: {content[:50]}", dur)
        return False

    log_result("instruction", True, "Followed instruction", dur)
    return True


def test_llama_json_output(model: str) -> bool:
    """Test Llama JSON output."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Output a JSON object with keys 'name' and 'age'. Use name='Test' and age=25. Only output the JSON, nothing else."}],
        max_tokens=100,
        temperature=0.1,  # Low temp for consistent output
    )
    if status != 200:
        log_result("json_output", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).strip()
    # Try to parse as JSON
    try:
        # Find JSON in response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)
            if "name" in data and "age" in data:
                log_result("json_output", True, f"Valid JSON: {json_str[:50]}", dur)
                return True
        log_result("json_output", False, f"Invalid JSON: {content[:100]}", dur)
        return False
    except json.JSONDecodeError:
        log_result("json_output", False, f"JSON parse error: {content[:100]}", dur)
        return False


# -----------------------------------------------------------------------------
# Qwen2.5-72B Edge Cases
# -----------------------------------------------------------------------------

def test_qwen25_basic_chat(model: str) -> bool:
    """Test Qwen2.5 basic chat."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What is the speed of light?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("basic_chat", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    # Should mention speed of light value
    if "300" not in content and "299" not in content:
        log_result("basic_chat", False, f"No speed value: {content[:100]}", dur)
        return False

    log_result("basic_chat", True, "Mentioned speed of light", dur)
    return True


def test_qwen25_complex_math(model: str) -> bool:
    """Test Qwen2.5 complex math."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "What is the square root of 144?"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("complex_math", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    if "12" not in content:
        log_result("complex_math", False, f"Wrong: {content[:100]}", dur)
        return False

    log_result("complex_math", True, "Correct: 12", dur)
    return True


def test_qwen25_streaming(model: str) -> bool:
    """Test Qwen2.5 streaming."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "List 5 fruits."}],
        max_tokens=100,
        stream=True,
    )
    if status != 200:
        log_result("streaming", False, f"HTTP {status}", dur)
        return False

    content, has_done, _ = stream_to_content(resp)
    if not has_done:
        log_result("streaming", False, "Missing [DONE]", dur)
        return False

    fruits = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "strawberry"]
    found = sum(1 for f in fruits if f in content.lower())
    if found < 3:
        log_result("streaming", False, f"Only {found} fruits", dur)
        return False

    log_result("streaming", True, f"Found {found} fruits", dur)
    return True


def test_qwen25_chinese(model: str) -> bool:
    """Test Qwen2.5 Chinese capability."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "用中文回答：中国的首都是哪里？"}],
        max_tokens=100,
    )
    if status != 200:
        log_result("chinese", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should mention Beijing in Chinese
    if "北京" not in content:
        log_result("chinese", False, f"No Beijing: {content[:100]}", dur)
        return False

    log_result("chinese", True, "Correct: 北京", dur)
    return True


def test_qwen25_long_output(model: str) -> bool:
    """Test Qwen2.5 longer output generation."""
    status, resp, dur = make_chat_request(
        model,
        [{"role": "user", "content": "Write a short story in exactly 3 sentences about a robot."}],
        max_tokens=300,
    )
    if status != 200:
        log_result("long_output", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp)
    # Should have substantial content
    if len(content) < 50:
        log_result("long_output", False, f"Too short: {len(content)} chars", dur)
        return False

    if "robot" not in content.lower():
        log_result("long_output", False, f"No robot: {content[:100]}", dur)
        return False

    log_result("long_output", True, f"Generated {len(content)} chars", dur)
    return True


# -----------------------------------------------------------------------------
# Vision Model Tests (Qwen3-VL-32B, Kimi-VL)
# -----------------------------------------------------------------------------

def test_vision_basic(model: str) -> bool:
    """Test vision model can see the image."""
    if not TEST_IMAGE_PATH.exists():
        log_result("vision_basic", False, f"Test image not found: {TEST_IMAGE_PATH}")
        return False

    status, resp, dur = make_vision_request(
        model,
        "What do you see in this image? Describe it briefly.",
        TEST_IMAGE_PATH,
    )
    if status != 200:
        log_result("vision_basic", False, f"HTTP {status}: {resp}", dur)
        return False

    content = get_content(resp).lower()
    # Image has: red rectangle, blue circle, green triangle, text
    if len(content) < 20:
        log_result("vision_basic", False, f"Response too short: {content}", dur)
        return False

    log_result("vision_basic", True, f"Got description: {content[:80]}...", dur)
    return True


def test_vision_shapes(model: str) -> bool:
    """Test vision model identifies shapes."""
    if not TEST_IMAGE_PATH.exists():
        log_result("vision_shapes", False, "Test image not found")
        return False

    status, resp, dur = make_vision_request(
        model,
        "What geometric shapes do you see? List them.",
        TEST_IMAGE_PATH,
    )
    if status != 200:
        log_result("vision_shapes", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    shapes = ["rectangle", "circle", "triangle", "square"]
    found = sum(1 for s in shapes if s in content)

    if found < 2:
        log_result("vision_shapes", False, f"Only {found} shapes: {content[:100]}", dur)
        return False

    log_result("vision_shapes", True, f"Found {found} shapes", dur)
    return True


def test_vision_colors(model: str) -> bool:
    """Test vision model identifies colors."""
    if not TEST_IMAGE_PATH.exists():
        log_result("vision_colors", False, "Test image not found")
        return False

    status, resp, dur = make_vision_request(
        model,
        "What colors are the shapes in this image?",
        TEST_IMAGE_PATH,
    )
    if status != 200:
        log_result("vision_colors", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    colors = ["red", "blue", "green"]
    found = sum(1 for c in colors if c in content)

    if found < 2:
        log_result("vision_colors", False, f"Only {found}/3 colors: {content[:100]}", dur)
        return False

    log_result("vision_colors", True, f"Found {found}/3 colors", dur)
    return True


def test_vision_text_reading(model: str) -> bool:
    """Test vision model reads text in image."""
    if not TEST_IMAGE_PATH.exists():
        log_result("vision_text", False, "Test image not found")
        return False

    status, resp, dur = make_vision_request(
        model,
        "Can you read any text in this image? What does it say?",
        TEST_IMAGE_PATH,
    )
    if status != 200:
        log_result("vision_text", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    # Image contains "Test Image" text
    if "test" not in content:
        log_result("vision_text", False, f"Didn't read text: {content[:100]}", dur)
        return False

    log_result("vision_text", True, "Read text from image", dur)
    return True


def test_vision_counting(model: str) -> bool:
    """Test vision model counts objects."""
    if not TEST_IMAGE_PATH.exists():
        log_result("vision_counting", False, "Test image not found")
        return False

    status, resp, dur = make_vision_request(
        model,
        "How many distinct geometric shapes are in this image? Count them.",
        TEST_IMAGE_PATH,
    )
    if status != 200:
        log_result("vision_counting", False, f"HTTP {status}", dur)
        return False

    content = get_content(resp).lower()
    # Should mention 3 shapes
    if "3" in content or "three" in content:
        log_result("vision_counting", True, "Counted 3 shapes", dur)
        return True

    log_result("vision_counting", False, f"Wrong count: {content[:100]}", dur)
    return False


# =============================================================================
# Model Test Configurations
# =============================================================================

MODEL_TESTS = {
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (Harmony)",
        "tests": [
            test_gptoss_harmony_reasoning,
            test_gptoss_tool_calling,
            test_gptoss_streaming,
            test_gptoss_multi_turn,
            test_gptoss_unicode,
        ],
    },
    "qwen3-32b": {
        "name": "Qwen3-32B (FP8)",
        "tests": [
            test_qwen_basic_chat,
            test_qwen_math,
            test_qwen_code,
            test_qwen_streaming,
            test_qwen_long_context,
        ],
    },
    "mistral-small-24b": {
        "name": "Mistral-Small-24B",
        "tests": [
            test_mistral_basic_chat,
            test_mistral_reasoning,
            test_mistral_streaming,
            test_mistral_format,
            test_mistral_multilingual,
        ],
    },
    "llama-3.1-70b": {
        "name": "Llama-3.1-70B (AWQ)",
        "tests": [
            test_llama_basic_chat,
            test_llama_math_awq,
            test_llama_streaming,
            test_llama_instruction_following,
            test_llama_json_output,
        ],
    },
    "qwen2.5-72b": {
        "name": "Qwen2.5-72B",
        "tests": [
            test_qwen25_basic_chat,
            test_qwen25_complex_math,
            test_qwen25_streaming,
            test_qwen25_chinese,
            test_qwen25_long_output,
        ],
    },
    "qwen3-vl-32b-thinking": {
        "name": "Qwen3-VL-32B (Vision, FP8)",
        "tests": [
            test_vision_basic,
            test_vision_shapes,
            test_vision_colors,
            test_vision_text_reading,
            test_vision_counting,
        ],
    },
    "kimi-vl": {
        "name": "Kimi-VL (Vision)",
        "tests": [
            test_vision_basic,
            test_vision_shapes,
            test_vision_colors,
            test_vision_text_reading,
            test_vision_counting,
        ],
    },
}


# =============================================================================
# Main Test Runner
# =============================================================================

def check_server() -> bool:
    """Check if server is running."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Server healthy, current model: {data.get('current_model')}")
            return True
    except Exception as e:
        print(f"Server check failed: {e}")
    return False


def run_model_tests(model_id: str, model_config: dict) -> bool:
    """Run all tests for a model (2 loads: basic HI, then full suite)."""
    global current_model
    current_model = model_id

    print(f"\n{'='*70}")
    print(f"MODEL: {model_config['name']} ({model_id})")
    print(f"{'='*70}")

    # First load: Basic HI test
    print(f"\n--- First Load: Basic HI Test ---")
    if not test_basic_hi(model_id):
        print(f"  ⚠️  Basic HI test failed, skipping edge cases for {model_id}")
        return False

    # Second load: Full test suite
    print(f"\n--- Second Load: Edge Case Tests ---")
    # Make a simple request to ensure model is still loaded
    status, _, _ = make_chat_request(model_id, [{"role": "user", "content": "Ready"}], max_tokens=10)
    if status != 200:
        print(f"  ⚠️  Second load failed for {model_id}")
        return False

    all_passed = True
    for test_func in model_config["tests"]:
        try:
            if not test_func(model_id):
                all_passed = False
        except Exception as e:
            log_result(test_func.__name__, False, f"Exception: {e}")
            all_passed = False

    return all_passed


def print_summary() -> Tuple[int, int]:
    """Print test summary and return (passed, total)."""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*70}")

    # Group by model
    by_model: Dict[str, List[TestResult]] = {}
    for r in results:
        if r.model not in by_model:
            by_model[r.model] = []
        by_model[r.model].append(r)

    total_passed = 0
    total_tests = 0

    for model_id, model_results in by_model.items():
        model_name = MODEL_TESTS.get(model_id, {}).get("name", model_id)
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        total_passed += passed
        total_tests += total

        status = "✅" if passed == total else "❌"
        print(f"\n{status} {model_name}: {passed}/{total} passed")
        for r in model_results:
            s = "  ✅" if r.passed else "  ❌"
            print(f"  {s} {r.test_name}")

    print(f"\n{'='*70}")
    pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    status = "✅ SUCCESS" if total_passed == total_tests else "❌ FAILURE"
    print(f"{status}: {total_passed}/{total_tests} tests passed ({pct:.1f}%)")
    print(f"{'='*70}")

    return total_passed, total_tests


def main():
    """Main test runner."""
    print("="*70)
    print("BlitzInfer Comprehensive Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print("="*70)

    # Check server
    if not check_server():
        print("\n❌ Server not responding, aborting tests")
        return 1

    # Check test image
    if not TEST_IMAGE_PATH.exists():
        print(f"\n⚠️  Test image not found at {TEST_IMAGE_PATH}")
        print("   Vision tests will fail. Create the image first.")

    # Run tests for each model
    for model_id, model_config in MODEL_TESTS.items():
        try:
            run_model_tests(model_id, model_config)
        except Exception as e:
            print(f"\n❌ Exception testing {model_id}: {e}")
            log_result("model_test", False, f"Exception: {e}")

    # Print summary
    passed, total = print_summary()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
