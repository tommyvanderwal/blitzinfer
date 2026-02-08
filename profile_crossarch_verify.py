#!/usr/bin/env python3
"""Profile cross-architecture reload: qwen3-32b <-> kimi-vl.

Tests that preloaded switching works across different architectures
and that generation quality is preserved.
"""

import logging
import os
import sys
import time
import glob as _glob

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("profile_cross")


def create_engine(model_path, **extra_kwargs):
    from sglang.srt.entrypoints.engine import Engine
    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        fp8_gemm_runner_backend="triton",
        log_level="info",
    )
    kwargs.update(extra_kwargs)
    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    logger.info(f"Engine created in {time.perf_counter() - t0:.1f}s")
    return engine


def drop_page_cache(model_path):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        model_dir = os.path.join(model_cache, "snapshots", snap)
    else:
        return
    st_files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    total = 0
    for fpath in st_files:
        real = os.path.realpath(fpath)
        sz = os.path.getsize(real)
        fd = os.open(real, os.O_RDONLY)
        os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        total += sz
    if total:
        logger.info(f"Dropped page cache: {total/1024**3:.1f}GB")


def test_model(engine, prompt, expect):
    params = {"temperature": 0, "max_new_tokens": 50}
    result = engine.generate(prompt=prompt, sampling_params=params)
    text = result.get("text", "")
    passed = expect.lower() in text.lower()
    return passed, text[:60]


def main():
    QWEN = "Qwen/Qwen3-32B-FP8"
    KIMI = "moonshotai/Kimi-VL-A3B-Thinking-2506"

    print("=" * 70)
    print("CROSS-ARCHITECTURE RELOAD PROFILE")
    print(f"  qwen3-32b <-> kimi-vl")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")

    engine = create_engine(QWEN)

    # Verify initial
    passed, text = test_model(engine, "What is 2 + 3? Answer with just the number.", "5")
    print(f"\nInitial (qwen3-32b): {'PASS' if passed else 'FAIL'} -> {text!r}")

    results = []

    # Switch sequence: qwen -> kimi -> qwen -> kimi -> qwen
    switches = [
        ("kimi-vl", KIMI, {}),
        ("qwen3-32b", QWEN, {}),
        ("kimi-vl", KIMI, {}),
        ("qwen3-32b", QWEN, {}),
    ]

    for i, (name, path, overrides) in enumerate(switches):
        print(f"\n{'='*70}")
        print(f"SWITCH {i+1}: -> {name}")
        print(f"{'='*70}")

        # Drop page cache and preload
        drop_page_cache(path)
        engine.preload_for_reload(path)
        # Wait proportional to model size
        wait = 15 if "Kimi" in path else 45
        logger.info(f"Waiting {wait}s for preload...")
        time.sleep(wait)

        # Record memory
        free_before, _ = torch.cuda.mem_get_info(0)

        # Reload
        t0 = time.perf_counter()
        success, msg = engine.reload_model(model_path=path, server_args_overrides=overrides)
        reload_time = time.perf_counter() - t0

        free_after, _ = torch.cuda.mem_get_info(0)

        # Verify
        if success:
            passed, text = test_model(
                engine,
                "What is 2 + 3? Answer with just the number.",
                "5",
            )
            passed2, text2 = test_model(
                engine,
                "What is the capital of France? Answer in one word.",
                "Paris",
            )
        else:
            passed = passed2 = False
            text = text2 = f"FAILED: {msg[:60]}"

        result = {
            "switch": i + 1,
            "target": name,
            "reload_time": reload_time,
            "success": success,
            "test1": passed,
            "test2": passed2,
            "text1": text,
            "text2": text2,
            "mem_drift_gb": (free_before - free_after) / 1024**3,
        }
        results.append(result)

        status = "PASS" if (passed and passed2) else "FAIL"
        print(f"  Reload: {reload_time:.3f}s | Gen: {status}")
        print(f"  2+3: {'PASS' if passed else 'FAIL'} -> {text[:50]!r}")
        print(f"  Capital: {'PASS' if passed2 else 'FAIL'} -> {text2[:50]!r}")

    engine.shutdown()

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':>3} {'Target':<15} {'Reload':>8} {'Test1':>6} {'Test2':>6}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['switch']:>3} {r['target']:<15} "
            f"{r['reload_time']:>7.3f}s "
            f"{'PASS' if r['test1'] else 'FAIL':>6} "
            f"{'PASS' if r['test2'] else 'FAIL':>6}"
        )

    reload_times = [r["reload_time"] for r in results if r["success"]]
    all_pass = all(r["test1"] and r["test2"] for r in results if r["success"])
    if reload_times:
        print(f"\nAvg reload: {sum(reload_times)/len(reload_times):.3f}s")
        print(f"Max reload: {max(reload_times):.3f}s")
    print(f"All tests pass: {'YES' if all_pass else 'NO'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
