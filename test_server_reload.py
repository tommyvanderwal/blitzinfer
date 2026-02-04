#!/usr/bin/env python3
"""Test server.py _switch_model with reload_model integration.

Verifies that the API server uses in-process reload_model() for fast
model switching instead of full Engine restart.

Run on remote RTX PRO 6000:
    cd ~/pythonprojects/blitzinfer
    venv/bin/python test_server_reload.py
"""

import asyncio
import gc
import os
import sys
import time
import logging

import torch

# Add sglang source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sglang", "python"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("test_server_reload")


def get_gpu_mem():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, free / 1024**3, total / 1024**3


async def test_server_reload():
    """Test _switch_model with reload_model integration."""
    # Import server module
    from blitzinfer.api.server import (
        ServerState, AVAILABLE_MODELS, SglEngine,
    )

    log.info("=" * 70)
    log.info("TEST: Server _switch_model with reload_model()")
    log.info("=" * 70)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Create minimal server state (skip StandbyManager for faster test)
    state = ServerState()

    # Patch out StandbyManager to skip 80GB pinned arena allocation
    class DummyStandby:
        def get_state(self):
            class S:
                name = "IDLE"
            return S()
        def start_prefetch(self, path): pass
        def get_standby_model(self): return None
        def is_ready(self, m): return False
    state.standby = DummyStandby()

    # Create queues
    for model_id in AVAILABLE_MODELS:
        state.queues[model_id] = asyncio.Queue(maxsize=64)

    # Phase 1: Load initial model (creates Engine)
    initial = "qwen2.5-7b"
    log.info(f"\n--- PHASE 1: Load initial model {initial} ---")

    t0 = time.perf_counter()
    await state._load_engine(initial)
    t_init = time.perf_counter() - t0
    log.info(f"Initial load: {t_init:.1f}s")

    # Quick generation test
    out = await state.engine.async_generate(
        prompt="What is 2+2? Answer with just the number.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[{initial}] Output: {out['text'][:80]}")

    # Phase 2: Switch via reload_model
    switches = [
        ("qwen2.5-7b", "llama-3.1-70b"),
        ("llama-3.1-70b", "qwen3-32b"),
        ("qwen3-32b", "qwen2.5-7b"),
    ]

    results = []
    for from_model, to_model in switches:
        log.info(f"\n{'='*60}")
        log.info(f"SWITCH: {from_model} -> {to_model}")
        log.info(f"{'='*60}")

        t0 = time.perf_counter()
        await state._switch_model(to_model)
        t_switch = time.perf_counter() - t0

        log.info(f"Switch time: {t_switch:.1f}s")
        assert state.current_model == to_model, f"Expected {to_model}, got {state.current_model}"

        # Verify generation
        out = await state.engine.async_generate(
            prompt="What is 2+2? Answer with just the number.",
            sampling_params={"max_new_tokens": 15, "temperature": 0.0},
        )
        text = out['text'][:80]
        log.info(f"[{to_model}] Output: {text}")

        used, _, _ = get_gpu_mem()
        log.info(f"GPU: {used:.1f}GB")

        results.append({
            "switch": f"{from_model} -> {to_model}",
            "time": t_switch,
            "output": text,
        })

    # Shutdown
    log.info("\n--- SHUTDOWN ---")
    state.engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: Server _switch_model with reload_model()")
    log.info("=" * 70)
    log.info(f"{'Switch':<35} {'Time':>8}")
    log.info("-" * 45)
    for r in results:
        log.info(f"{r['switch']:<35} {r['time']:>6.1f}s")

    total = sum(r['time'] for r in results)
    log.info("-" * 45)
    log.info(f"{'TOTAL':<35} {total:>6.1f}s")
    log.info(f"\nAll {len(results)} switches used reload_model (no Engine restart)")


if __name__ == "__main__":
    asyncio.run(test_server_reload())
