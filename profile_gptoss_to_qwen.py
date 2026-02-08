#!/usr/bin/env python3
"""Profile: gpt-oss-120b -> qwen3-32b with preload.

Start with the larger model (fits in VRAM), switch to smaller one.
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
logger = logging.getLogger("profile_g2q")


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


def test_qwen(engine):
    tests = [
        ("What is 2 + 3? Answer with just the number.", "5"),
        ("What is the capital of France? Answer in one word.", "Paris"),
    ]
    params = {"temperature": 0, "max_new_tokens": 50}
    passed_all = True
    for prompt, expect in tests:
        result = engine.generate(prompt=prompt, sampling_params=params)
        text = result.get("text", "")
        passed = expect.lower() in text.lower()
        passed_all = passed_all and passed
        print(f"  {'PASS' if passed else 'FAIL'}: {prompt[:40]} -> {text[:50]!r}")
    return passed_all


def test_gptoss(engine):
    try:
        from vllm.entrypoints.openai.parser.harmony_utils import (
            parse_chat_inputs_to_harmony_messages,
            render_for_completion,
            get_system_message,
            parse_chat_output,
            get_stop_tokens_for_assistant_actions,
        )
    except ImportError:
        print("  SKIP: Harmony utils not available")
        return True

    tests = [
        ("What is 2 + 3? Answer with just the number.", "5"),
        ("What is the capital of France? Answer in one word.", "Paris"),
    ]
    stop_tokens = get_stop_tokens_for_assistant_actions()
    params = {"temperature": 0, "max_new_tokens": 500, "stop_token_ids": stop_tokens}

    passed_all = True
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
        passed_all = passed_all and passed
        print(f"  {'PASS' if passed else 'FAIL'}: {prompt[:40]} -> {text[:50]!r}")
    return passed_all


def main():
    GPT_OSS = "openai/gpt-oss-120b"
    QWEN = "Qwen/Qwen3-32B-FP8"

    print("=" * 70)
    print("PROFILE: gpt-oss-120b -> qwen3-32b -> gpt-oss-120b")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")

    # Start with gpt-oss-120b (large model, sets VRAM)
    engine = create_engine(GPT_OSS, context_length=131072)

    print("\n--- Verify gpt-oss-120b initial load ---")
    test_gptoss(engine)

    # Preload qwen3-32b
    print(f"\n--- Preloading {QWEN} ---")
    drop_page_cache(QWEN)
    engine.preload_for_reload(QWEN)
    logger.info("Waiting 45s for preload (~32GB)...")
    time.sleep(45)

    # Switch to qwen3-32b
    print(f"\n--- Switch 1: gpt-oss-120b -> qwen3-32b ---")
    t0 = time.perf_counter()
    success, msg = engine.reload_model(model_path=QWEN)
    t1 = time.perf_counter() - t0
    print(f"Reload time: {t1:.3f}s, success={success}")

    if success:
        test_qwen(engine)
    else:
        print(f"FAILED: {msg}")

    # Preload gpt-oss-120b back
    print(f"\n--- Preloading {GPT_OSS} ---")
    drop_page_cache(GPT_OSS)
    engine.preload_for_reload(GPT_OSS)
    logger.info("Waiting 90s for preload (~60GB)...")
    time.sleep(90)

    # Switch back to gpt-oss-120b
    print(f"\n--- Switch 2: qwen3-32b -> gpt-oss-120b ---")
    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=GPT_OSS,
        server_args_overrides={"context_length": 131072},
    )
    t2 = time.perf_counter() - t0
    print(f"Reload time: {t2:.3f}s, success={success}")

    if success:
        test_gptoss(engine)

        # Switch back to qwen one more time
        print(f"\n--- Preloading {QWEN} ---")
        drop_page_cache(QWEN)
        engine.preload_for_reload(QWEN)
        logger.info("Waiting 45s for preload...")
        time.sleep(45)

        print(f"\n--- Switch 3: gpt-oss-120b -> qwen3-32b ---")
        t0 = time.perf_counter()
        success, msg = engine.reload_model(model_path=QWEN)
        t3 = time.perf_counter() - t0
        print(f"Reload time: {t3:.3f}s, success={success}")
        if success:
            test_qwen(engine)
        else:
            print(f"FAILED: {msg}")
    else:
        print(f"FAILED: {msg}")
        t3 = None

    engine.shutdown()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Switch 1 (gpt-oss->qwen, 32GB):  {t1:.3f}s")
    print(f"  Switch 2 (qwen->gpt-oss, 60GB):   {t2:.3f}s")
    if t3 is not None:
        print(f"  Switch 3 (gpt-oss->qwen, 32GB):  {t3:.3f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
