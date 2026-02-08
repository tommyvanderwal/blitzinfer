#!/usr/bin/env python3
"""Profile reload with preload, verify correct generation.

gpt-oss-120b -> qwen3-32b (with pinned preload)
Tests that the optimized reload produces a working model.
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
logger = logging.getLogger("profile_verify")


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
    logger.info(f"Dropped page cache: {total/1024**3:.1f}GB")


def test_generation(engine, model_name):
    """Run multiple generation tests to verify model quality."""
    tests = [
        ("What is 2 + 3? Answer with just the number.", "5"),
        ("What is the capital of France? Answer in one word.", "Paris"),
        ("What color is the sky on a clear day? Answer in one word.", "blue"),
    ]
    params = {"temperature": 0, "max_new_tokens": 50}

    if model_name == "gpt-oss-120b":
        # Use Harmony encoding
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import (
                parse_chat_inputs_to_harmony_messages,
                render_for_completion,
                get_system_message,
                parse_chat_output,
                get_stop_tokens_for_assistant_actions,
            )
        except ImportError:
            logger.warning("Harmony utils not available, skipping gpt-oss verification")
            return True, "Skipped (no harmony utils)"

        stop_tokens = get_stop_tokens_for_assistant_actions()
        harmony_params = {"temperature": 0, "max_new_tokens": 500, "stop_token_ids": stop_tokens}

        passed_all = True
        outputs = []
        for prompt, expect in tests:
            chat_msgs = [{"role": "user", "content": prompt}]
            sys_msg = get_system_message(with_custom_tools=False)
            harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)
            token_ids = render_for_completion(harmony_msgs)

            result = engine.generate(input_ids=token_ids, sampling_params=harmony_params)
            output_ids = result.get("meta_info", {}).get("output_token_ids", [])
            if output_ids:
                parsed = parse_chat_output(list(output_ids))
                text = parsed.final_content or result.get("text", "")[:50]
            else:
                text = result.get("text", "")[:50]

            passed = expect.lower() in text.lower()
            passed_all = passed_all and passed
            outputs.append(f"  Q: {prompt[:40]} -> {text[:40]!r} {'PASS' if passed else 'FAIL'}")

        return passed_all, "\n".join(outputs)
    else:
        # Standard text generation
        passed_all = True
        outputs = []
        for prompt, expect in tests:
            result = engine.generate(prompt=prompt, sampling_params=params)
            text = result.get("text", "")
            passed = expect.lower() in text.lower()
            passed_all = passed_all and passed
            outputs.append(f"  Q: {prompt[:40]} -> {text[:40]!r} {'PASS' if passed else 'FAIL'}")

        return passed_all, "\n".join(outputs)


def main():
    INITIAL = "Qwen/Qwen3-32B-FP8"
    TARGET = "Qwen/Qwen3-32B-FP8"  # Same model to verify correctness without AWQ issues

    # Also test a real cross-arch switch
    CROSS_TARGET = "openai/gpt-oss-120b"
    CROSS_OVERRIDES = {"context_length": 131072}

    print("=" * 70)
    print("VERIFICATION PROFILE: Preload + Reload with Correctness Checks")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")

    # ---- Test 1: qwen3-32b -> qwen3-32b (same model reload) ----
    print(f"\n{'='*70}")
    print("TEST 1: Same-model reload (Qwen3-32B -> Qwen3-32B)")
    print(f"{'='*70}")

    engine = create_engine(INITIAL)

    # Verify initial model
    passed, details = test_generation(engine, "qwen3-32b")
    print(f"\nInitial model verification: {'PASS' if passed else 'FAIL'}")
    print(details)

    # Drop page cache and preload
    drop_page_cache(TARGET)
    engine.preload_for_reload(TARGET)
    logger.info("Waiting 45s for preload...")
    time.sleep(45)

    # Reload same model
    t0 = time.perf_counter()
    success, msg = engine.reload_model(model_path=TARGET)
    reload_time = time.perf_counter() - t0
    print(f"\nReload time: {reload_time:.3f}s, success={success}")

    if success:
        passed, details = test_generation(engine, "qwen3-32b")
        print(f"Post-reload verification: {'PASS' if passed else 'FAIL'}")
        print(details)
    else:
        print(f"Reload FAILED: {msg}")

    # ---- Test 2: qwen3-32b -> gpt-oss-120b (cross-architecture) ----
    print(f"\n{'='*70}")
    print("TEST 2: Cross-arch reload (Qwen3-32B -> GPT-OSS-120B)")
    print(f"{'='*70}")

    drop_page_cache(CROSS_TARGET)
    engine.preload_for_reload(CROSS_TARGET)
    logger.info("Waiting 90s for gpt-oss-120b preload (60GB)...")
    time.sleep(90)

    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=CROSS_TARGET,
        server_args_overrides=CROSS_OVERRIDES,
    )
    reload_time2 = time.perf_counter() - t0
    print(f"\nReload time: {reload_time2:.3f}s, success={success}")

    if success:
        passed, details = test_generation(engine, "gpt-oss-120b")
        print(f"Post-reload verification: {'PASS' if passed else 'FAIL'}")
        print(details)

        # ---- Test 3: gpt-oss-120b -> qwen3-32b (back to smaller model) ----
        print(f"\n{'='*70}")
        print("TEST 3: Cross-arch back (GPT-OSS-120B -> Qwen3-32B)")
        print(f"{'='*70}")

        drop_page_cache(INITIAL)
        engine.preload_for_reload(INITIAL)
        logger.info("Waiting 45s for qwen3-32b preload...")
        time.sleep(45)

        t0 = time.perf_counter()
        success, msg = engine.reload_model(model_path=INITIAL)
        reload_time3 = time.perf_counter() - t0
        print(f"\nReload time: {reload_time3:.3f}s, success={success}")

        if success:
            passed, details = test_generation(engine, "qwen3-32b")
            print(f"Post-reload verification: {'PASS' if passed else 'FAIL'}")
            print(details)
        else:
            print(f"Reload FAILED: {msg}")
            reload_time3 = None
    else:
        print(f"Reload FAILED: {msg}")
        reload_time2 = None
        reload_time3 = None

    engine.shutdown()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Test 1 (qwen->qwen, 32GB):     {reload_time:.3f}s")
    if reload_time2:
        print(f"  Test 2 (qwen->gpt-oss, 60GB):  {reload_time2:.3f}s")
    if reload_time3:
        print(f"  Test 3 (gpt-oss->qwen, 32GB):  {reload_time3:.3f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
