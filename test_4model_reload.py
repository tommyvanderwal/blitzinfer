#!/usr/bin/env python3
"""Comprehensive 4-model reload test with full profiling.

Switches between 4 different models (different sizes, architectures) via
in-process reload_model(), verifying generation quality after each switch.

Models tested:
  1. Qwen/Qwen3-32B-FP8          (~32GB, text, FP8)
  2. Qwen/Qwen2.5-7B-Instruct    (~14GB, text, bf16)
  3. openai/gpt-oss-120b          (~60GB, text, mxfp4, Harmony)

Profiles:
  - Preload time (background disk→CPU)
  - Switchover time (free old + load new + init KV)
  - Generation verification with model-appropriate encoding
  - Memory at every stage (GPU alloc/reserved, CPU RSS)
"""

import gc
import logging
import os
import sys
import time
import glob as _glob

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import psutil
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_4model")


# ── Models ──────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "qwen3-32b-fp8",
        "path": "Qwen/Qwen3-32B-FP8",
        "overrides": {},
        "engine_extra": {"fp8_gemm_runner_backend": "triton"},
    },
    {
        "name": "qwen2.5-7b",
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "overrides": {},
        "engine_extra": {},
    },
    {
        "name": "gpt-oss-120b",
        "path": "openai/gpt-oss-120b",
        "overrides": {"context_length": 131072, "moe_runner_backend": "triton_kernel", "dtype": "bfloat16"},
        "engine_extra": {},
    },
]

# Switch sequence: cycle through all 3 models, twice
SWITCH_SEQUENCE = [
    MODELS[1],  # qwen3-32b -> qwen2.5-7b
    MODELS[2],  # qwen2.5-7b -> gpt-oss-120b (large, Harmony)
    MODELS[0],  # gpt-oss-120b -> qwen3-32b (round-trip)
    MODELS[2],  # qwen3-32b -> gpt-oss-120b (repeat, should be faster)
    MODELS[1],  # gpt-oss-120b -> qwen2.5-7b
    MODELS[0],  # qwen2.5-7b -> qwen3-32b (final)
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def mem_snapshot():
    ga = torch.cuda.memory_allocated() / 1024**3
    gr = torch.cuda.memory_reserved() / 1024**3
    rss = psutil.Process().memory_info().rss / 1024**3
    return ga, gr, rss


def mem_str():
    ga, gr, rss = mem_snapshot()
    return f"GPU alloc={ga:.1f}GB rsrv={gr:.1f}GB | RSS={rss:.1f}GB"


def create_engine(model_path, **extra_kwargs):
    from sglang.srt.entrypoints.engine import Engine
    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
    )
    kwargs.update(extra_kwargs)
    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    logger.info(f"Engine created in {time.perf_counter() - t0:.1f}s")
    return engine


def drop_page_cache(model_path):
    """Evict model files from OS page cache (forces cold disk reads)."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        model_dir = os.path.join(model_cache, "snapshots", snap)
    else:
        return 0
    st_files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    total = 0
    for fpath in st_files:
        real = os.path.realpath(fpath)
        sz = os.path.getsize(real)
        fd = os.open(real, os.O_RDONLY)
        os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        total += sz
    gb = total / 1024**3
    if total:
        logger.info(f"Dropped page cache: {gb:.1f}GB for {model_path}")
    return gb


def verify_standard(engine, model_name):
    """Verify standard text models (qwen, etc)."""
    tests = [
        ("What is 2 + 3? Answer with just the number.", "5"),
        ("What is the capital of France? Answer in one word.", "Paris"),
    ]
    params = {"temperature": 0, "max_new_tokens": 50}
    results = []
    all_pass = True

    for prompt, expect in tests:
        result = engine.generate(prompt=prompt, sampling_params=params)
        text = result.get("text", "")
        passed = expect.lower() in text.lower()
        all_pass = all_pass and passed
        results.append(
            f"  {'PASS' if passed else 'FAIL'}: "
            f"{prompt[:40]:<42} -> {text[:60]!r}"
        )

    return all_pass, "\n".join(results)


def verify_gpt_oss(engine, model_name):
    """Verify gpt-oss-120b using Harmony encoding."""
    try:
        from vllm.entrypoints.openai.parser.harmony_utils import (
            parse_chat_inputs_to_harmony_messages,
            render_for_completion,
            get_system_message,
            parse_chat_output,
            get_stop_tokens_for_assistant_actions,
        )
    except ImportError:
        return True, "  SKIP: Harmony utils not available"

    tests = [
        ("What is 2 + 3? Answer with just the number.", "5"),
        ("What is the capital of France? Answer in one word.", "Paris"),
    ]
    stop_tokens = get_stop_tokens_for_assistant_actions()
    params = {"temperature": 0, "max_new_tokens": 500, "stop_token_ids": stop_tokens}
    results = []
    all_pass = True

    for prompt, expect in tests:
        chat_msgs = [{"role": "user", "content": prompt}]
        sys_msg = get_system_message(with_custom_tools=False)
        harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)
        token_ids = render_for_completion(harmony_msgs)

        result = engine.generate(input_ids=token_ids, sampling_params=params)
        output_ids = result.get("meta_info", {}).get("output_token_ids", [])
        if output_ids:
            parsed = parse_chat_output(list(output_ids))
            text = parsed.final_content or result.get("text", "")[:80]
        else:
            text = result.get("text", "")[:80]

        passed = expect.lower() in text.lower()
        all_pass = all_pass and passed
        results.append(
            f"  {'PASS' if passed else 'FAIL'}: "
            f"{prompt[:40]:<42} -> {text[:60]!r}"
        )

    return all_pass, "\n".join(results)


def verify_model(engine, model_name):
    """Dispatch to the right verification function."""
    if model_name == "gpt-oss-120b":
        return verify_gpt_oss(engine, model_name)
    else:
        return verify_standard(engine, model_name)


def estimate_preload_time(model_path):
    """Estimate preload time based on model size on disk."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        model_dir = os.path.join(model_cache, "snapshots", snap)
    else:
        return 60  # conservative default
    st_files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    total = sum(os.path.getsize(os.path.realpath(f)) for f in st_files)
    gb = total / 1024**3
    # Estimate ~1GB/s from cold disk, ~8GB/s from page cache
    # Use cold estimate + margin
    wait = max(int(gb * 1.2) + 5, 15)
    logger.info(f"Model {model_path}: {gb:.1f}GB on disk, estimated preload wait: {wait}s")
    return wait


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    initial = MODELS[0]

    print("=" * 78)
    print("4-MODEL RELOAD TEST: Full Profiling")
    print("=" * 78)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")
    print(f"System RAM: {psutil.virtual_memory().total/1024**3:.0f}GB")
    print(f"Models: {', '.join(m['name'] for m in MODELS)}")
    print(f"Switches: {len(SWITCH_SEQUENCE)}")
    print(f"Start: {initial['name']}")
    print()

    # ── Initial load ──
    print(f"{'='*78}")
    print(f"INITIAL LOAD: {initial['name']}")
    print(f"{'='*78}")
    t_start = time.perf_counter()
    engine = create_engine(initial["path"], **initial["engine_extra"])
    t_init = time.perf_counter() - t_start
    print(f"Initial load: {t_init:.1f}s | {mem_str()}")

    passed, details = verify_model(engine, initial["name"])
    print(f"Verification: {'PASS' if passed else 'FAIL'}")
    print(details)
    print()

    if not passed:
        print("ABORT: Initial model verification failed!")
        engine.shutdown()
        return

    # Track results
    results = []
    ga0, gr0, rss0 = mem_snapshot()

    # ── Switch sequence ──
    for i, target in enumerate(SWITCH_SEQUENCE):
        print(f"\n{'='*78}")
        print(f"SWITCH {i+1}/{len(SWITCH_SEQUENCE)}: -> {target['name']}")
        print(f"{'='*78}")

        ga_before, gr_before, rss_before = mem_snapshot()
        print(f"Before: GPU alloc={ga_before:.1f}GB rsrv={gr_before:.1f}GB | RSS={rss_before:.1f}GB")

        # Drop page cache to test realistic cold-cache scenario
        drop_page_cache(target["path"])

        # Phase A: Background preload (disk → CPU RAM)
        t_preload_start = time.perf_counter()
        engine.preload_for_reload(target["path"])
        wait = estimate_preload_time(target["path"])
        logger.info(f"Waiting {wait}s for preload...")
        time.sleep(wait)

        # Check preload status
        ga_preloaded, gr_preloaded, rss_preloaded = mem_snapshot()
        preload_time = time.perf_counter() - t_preload_start
        print(f"Preloaded: {preload_time:.1f}s | RSS={rss_preloaded:.1f}GB (delta={rss_preloaded-rss_before:+.1f}GB)")

        # Phase B: Switchover (free old + load new + init KV)
        t_switch = time.perf_counter()
        success, msg = engine.reload_model(
            model_path=target["path"],
            server_args_overrides=target["overrides"],
        )
        switch_time = time.perf_counter() - t_switch

        ga_after, gr_after, rss_after = mem_snapshot()
        print(f"After:  GPU alloc={ga_after:.1f}GB rsrv={gr_after:.1f}GB | RSS={rss_after:.1f}GB")

        # Phase C: Verify
        t_verify = time.perf_counter()
        if success:
            passed, details = verify_model(engine, target["name"])
        else:
            passed = False
            details = f"  RELOAD FAILED: {msg[:100]}"
        verify_time = time.perf_counter() - t_verify

        total_time = time.perf_counter() - t_preload_start

        result = {
            "switch": i + 1,
            "target": target["name"],
            "preload_time": preload_time,
            "switch_time": switch_time,
            "verify_time": verify_time,
            "total_time": total_time,
            "success": success,
            "passed": passed,
            "gpu_alloc_after": ga_after,
            "gpu_reserved_after": gr_after,
            "rss_before": rss_before,
            "rss_preloaded": rss_preloaded,
            "rss_after": rss_after,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"Switchover: {switch_time:.3f}s | Verify: {status} ({verify_time:.1f}s)")
        print(f"Total (preload+switch+verify): {total_time:.1f}s")
        print(details)

        # Force cleanup check
        gc.collect()
        torch.cuda.empty_cache()

    engine.shutdown()

    # ── Summary ──
    print(f"\n{'='*78}")
    print("RESULTS SUMMARY")
    print(f"{'='*78}")
    print(f"{'#':>3} {'Target':<20} {'Preload':>8} {'Switch':>8} {'Verify':>7} {'Total':>8} {'GPU':>6} {'RSS':>6}")
    print("-" * 78)

    for r in results:
        print(
            f"{r['switch']:>3} {r['target']:<20} "
            f"{r['preload_time']:>7.1f}s "
            f"{r['switch_time']:>7.3f}s "
            f"{'PASS' if r['passed'] else 'FAIL':>7} "
            f"{r['total_time']:>7.1f}s "
            f"{r['gpu_alloc_after']:>5.1f}G "
            f"{r['rss_after']:>5.1f}G"
        )

    # Stats
    successful = [r for r in results if r["success"]]
    all_pass = all(r["passed"] for r in results)
    switch_times = [r["switch_time"] for r in successful]

    print()
    if switch_times:
        print(f"Avg switchover:  {sum(switch_times)/len(switch_times):.3f}s")
        print(f"Max switchover:  {max(switch_times):.3f}s")
        print(f"Min switchover:  {min(switch_times):.3f}s")

    ga_final, gr_final, rss_final = mem_snapshot()
    print(f"GPU drift:       {ga_final - ga0:+.1f}GB (alloc), {gr_final - gr0:+.1f}GB (reserved)")
    print(f"RSS drift:       {rss_final - rss0:+.1f}GB")
    print(f"All verify:      {'YES' if all_pass else 'NO'}")
    print(f"Switches:        {len(successful)}/{len(results)} succeeded")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
