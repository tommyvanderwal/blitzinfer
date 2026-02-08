#!/usr/bin/env python3
"""Test model loading in different orders to isolate crash pattern.

Logs every action to disk with immediate flush to capture state
milliseconds before any crash.
"""

import os
import sys
import time
import requests
from datetime import datetime

# Configuration
BASE_URL = "http://192.168.2.90:8000"
TIMEOUT = 300
LOG_FILE = "/tmp/model_order_test.log"

# All models in original order
MODELS_ORIGINAL = [
    "gpt-oss-120b",
    "qwen3-32b",
    "mistral-small-24b",
    "llama-3.1-70b",
]

# Reverse order to test if it's position vs model specific
MODELS_REVERSE = list(reversed(MODELS_ORIGINAL))

# Skip mistral to test if mistral cleanup is the issue
MODELS_NO_MISTRAL = [
    "gpt-oss-120b",
    "qwen3-32b",
    "llama-3.1-70b",  # Now 3rd instead of 4th
]

# Llama first to test if llama is the issue
MODELS_LLAMA_FIRST = [
    "llama-3.1-70b",
    "gpt-oss-120b",
    "qwen3-32b",
    "mistral-small-24b",
]


def log(msg: str):
    """Log with timestamp and immediate disk flush."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())  # Force write to disk


def get_memory():
    """Get memory state from remote."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024  # MB
    except:
        pass
    return 0


def test_model(model: str, index: int, total: int) -> bool:
    """Test a single model with detailed logging."""
    log(f"")
    log(f"{'='*60}")
    log(f"MODEL {index+1}/{total}: {model}")
    log(f"{'='*60}")

    # Request
    log(f"SENDING REQUEST for {model}...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say exactly: TEST OK"}],
        "max_tokens": 20,
        "temperature": 0,
    }

    try:
        log(f"POST /v1/chat/completions starting...")
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start
        log(f"POST completed in {elapsed:.1f}s, status={resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content'][:50]
            log(f"RESPONSE: {content}")
            log(f"SUCCESS: {model}")
            return True
        else:
            log(f"FAILED: HTTP {resp.status_code}")
            log(f"Response: {resp.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        log(f"TIMEOUT after {TIMEOUT}s")
        return False
    except requests.exceptions.ConnectionError as e:
        log(f"CONNECTION ERROR: {e}")
        log(f">>> LIKELY CRASH - server not responding <<<")
        return False
    except Exception as e:
        log(f"EXCEPTION: {type(e).__name__}: {e}")
        return False


def run_test_sequence(name: str, models: list):
    """Run a sequence of model tests."""
    log(f"")
    log(f"{'#'*70}")
    log(f"# TEST SEQUENCE: {name}")
    log(f"# Models: {models}")
    log(f"{'#'*70}")

    # Check server is up
    log(f"Checking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        health = resp.json()
        log(f"Server healthy, current_model={health.get('current_model')}")
    except Exception as e:
        log(f"Server not responding: {e}")
        return False

    results = []
    for i, model in enumerate(models):
        success = test_model(model, i, len(models))
        results.append((model, success))

        if not success:
            log(f"STOPPING - {model} failed")
            break

        # Brief pause between models
        log(f"Pause 2s before next model...")
        time.sleep(2)

    # Summary
    log(f"")
    log(f"{'='*60}")
    log(f"SEQUENCE RESULTS: {name}")
    log(f"{'='*60}")
    for model, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        log(f"  {status}: {model}")

    passed = sum(1 for _, s in results if s)
    log(f"Total: {passed}/{len(results)} passed")

    return all(s for _, s in results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=["original", "reverse", "no-mistral", "llama-first"],
                       default="original", help="Model order to test")
    args = parser.parse_args()

    # Clear log
    with open(LOG_FILE, "w") as f:
        f.write(f"=== MODEL ORDER TEST: {args.order} ===\n")
        f.write(f"Started: {datetime.now().isoformat()}\n\n")

    orders = {
        "original": ("Original Order", MODELS_ORIGINAL),
        "reverse": ("Reverse Order", MODELS_REVERSE),
        "no-mistral": ("Skip Mistral", MODELS_NO_MISTRAL),
        "llama-first": ("Llama First", MODELS_LLAMA_FIRST),
    }

    name, models = orders[args.order]
    log(f"Testing: {name}")
    log(f"Log file: {LOG_FILE}")

    success = run_test_sequence(name, models)

    log(f"")
    log(f"TEST COMPLETE: {'PASS' if success else 'FAIL'}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
