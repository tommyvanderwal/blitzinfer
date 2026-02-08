#!/usr/bin/env python3
"""8-switch stress test for BlitzInfer server."""

import requests
import json
import time
import sys

URL = "http://192.168.2.90:8000"
TIMEOUT = 300  # 5 minutes per request

MODELS = [
    "gpt-oss-120b",       # 1: already loaded (or first switch)
    "qwen3-32b",          # 2: switch
    "gpt-oss-120b",       # 3: switch back
    "kimi-vl",            # 4: switch to vision
    "gpt-oss-120b",       # 5: switch back
    "qwen3-coder-next",   # 6: switch to coder
    "gpt-oss-120b",       # 7: switch back
    "qwen3-32b",          # 8: final switch
]

def health_check():
    try:
        r = requests.get(f"{URL}/health", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def status_check():
    try:
        r = requests.get(f"{URL}/status", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def send_request(model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 100,
        "temperature": 0.1,
    }
    try:
        r = requests.post(
            f"{URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        return r.status_code, r.json()
    except requests.exceptions.Timeout:
        return 0, {"error": "timeout after 300s"}
    except Exception as e:
        return 0, {"error": str(e)}

def main():
    print("=== BlitzInfer 8-Switch Stress Test ===")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {URL}")
    print()

    # Pre-test health
    print("--- Pre-test health check ---")
    h = health_check()
    print(json.dumps(h, indent=2))
    print()

    results = []

    for i, model in enumerate(MODELS):
        num = i + 1
        print(f"=== Request {num}/8: {model} ===")

        start = time.time()
        code, data = send_request(model)
        elapsed = time.time() - start

        if code == 200:
            try:
                content = data["choices"][0]["message"].get("content", None)
                if content is None:
                    content = "(no content)"
                content_preview = content[:120].replace("\n", " ")
            except (KeyError, IndexError, TypeError):
                content_preview = "(no content in response)"
            print(f"  STATUS: PASS (HTTP {code}, {elapsed:.1f}s)")
            print(f"  CONTENT: {content_preview}")
            results.append(("PASS", elapsed))
        else:
            error_msg = str(data.get("detail", data.get("error", str(data))))[:200]
            print(f"  STATUS: FAIL (HTTP {code}, {elapsed:.1f}s)")
            print(f"  ERROR: {error_msg}")
            results.append((f"FAIL:{code}", elapsed))

        # Health check
        h = health_check()
        if "error" not in h:
            print(f"  HEALTH: model={h.get('current_model')}, "
                  f"switches={h.get('switch_count')}, "
                  f"state={h.get('queue_state')}, "
                  f"failed={h.get('failed_models', {})}")
        else:
            print(f"  HEALTH: unreachable ({h['error']})")

        # GPU memory
        s = status_check()
        if "error" not in s:
            used = s.get('gpu_memory_used_gb', 0)
            total = s.get('gpu_memory_total_gb', 0)
            if isinstance(used, (int, float)) and isinstance(total, (int, float)):
                print(f"  GPU: {used:.1f}/{total:.1f} GB")
            else:
                print(f"  GPU: {used}/{total}")
        else:
            print(f"  STATUS: unreachable ({s['error']})")

        print()

    # Summary
    print("=== SUMMARY ===")
    print(f"{'#':<4} {'Model':<25} {'Result':<12} {'Time':>8}")
    print("-" * 53)
    for i, (model, (result, elapsed)) in enumerate(zip(MODELS, results)):
        print(f"{i+1:<4} {model:<25} {result:<12} {elapsed:>7.1f}s")
    print()

    pass_count = sum(1 for r, _ in results if r == "PASS")
    fail_count = len(results) - pass_count
    print(f"Result: {pass_count}/8 passed, {fail_count}/8 failed")

    # Final health
    print()
    print("--- Final health check ---")
    h = health_check()
    print(json.dumps(h, indent=2))

    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
