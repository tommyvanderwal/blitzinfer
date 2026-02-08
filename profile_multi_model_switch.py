#!/usr/bin/env python3
"""Profile multi-model switching with preload optimization.

Tests switching between multiple models with pinned memory preloading.
Verifies generation quality for each model after reload.
"""

import gc
import glob as _glob
import logging
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("profile_multi")

# Model definitions
MODELS = {
    "qwen3-32b": {
        "path": "Qwen/Qwen3-32B-FP8",
        "overrides": {},
        "test_prompt": "What is 2 + 3? Answer with just the number.",
        "expect_contains": "5",
    },
    "gpt-oss-120b": {
        "path": "openai/gpt-oss-120b",
        "overrides": {"context_length": 131072},
        "test_prompt": "What is the capital of France? Answer in one word.",
        "expect_contains": "Paris",
    },
    "kimi-vl": {
        "path": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "overrides": {},
        "test_prompt": "What is 7 * 8? Answer with just the number.",
        "expect_contains": "56",
    },
}


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
    """Drop OS page cache for a model's safetensors files."""
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
    return total


def verify_generation(engine, prompt, expect_contains, model_name, use_harmony=False):
    """Generate text and check output quality."""
    params = {"temperature": 0, "max_new_tokens": 100}

    if use_harmony:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import (
                parse_chat_inputs_to_harmony_messages,
                render_for_completion,
                get_system_message,
                parse_chat_output,
                get_stop_tokens_for_assistant_actions,
            )
            chat_msgs = [{"role": "user", "content": prompt}]
            sys_msg = get_system_message(with_custom_tools=False)
            harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)
            prompt_token_ids = render_for_completion(harmony_msgs)
            stop_tokens = get_stop_tokens_for_assistant_actions()
            params["stop_token_ids"] = stop_tokens
            params["max_new_tokens"] = 500

            result = engine.generate(
                input_ids=prompt_token_ids,
                sampling_params=params,
            )
            text = result.get("text", "")
            # Parse Harmony output
            output_ids = result.get("meta_info", {}).get("output_token_ids", [])
            if output_ids:
                parsed = parse_chat_output(list(output_ids))
                final = parsed.final_content or ""
            else:
                final = text
            return expect_contains.lower() in final.lower(), final[:100]
        except Exception as e:
            logger.error(f"Harmony generation failed: {e}")
            return False, str(e)[:100]
    else:
        try:
            result = engine.generate(prompt=prompt, sampling_params=params)
            text = result.get("text", "")
            passed = expect_contains.lower() in text.lower()
            return passed, text[:100]
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return False, str(e)[:100]


def main():
    # Define the switch sequence
    INITIAL = "qwen3-32b"
    TARGETS = ["gpt-oss-120b", "qwen3-32b", "gpt-oss-120b", "qwen3-32b"]

    print("=" * 70)
    print("MULTI-MODEL SWITCHING PROFILE (with pinned preload)")
    print(f"  Initial: {INITIAL}")
    print(f"  Switches: {' -> '.join(TARGETS)}")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")

    initial_model = MODELS[INITIAL]
    results = []

    # Step 1: Create initial engine
    engine = create_engine(initial_model["path"], **initial_model.get("overrides", {}))

    # Verify initial model
    use_harmony = (INITIAL == "gpt-oss-120b")
    passed, text = verify_generation(
        engine, initial_model["test_prompt"],
        initial_model["expect_contains"], INITIAL, use_harmony=use_harmony,
    )
    logger.info(f"Initial model ({INITIAL}) verification: {'PASS' if passed else 'FAIL'} - {text!r}")

    for i, target_name in enumerate(TARGETS):
        target = MODELS[target_name]
        target_path = target["path"]
        target_overrides = target.get("overrides", {})

        print(f"\n{'='*70}")
        print(f"SWITCH {i+1}: -> {target_name}")
        print(f"{'='*70}")

        # Drop page cache for target
        dropped = drop_page_cache(target_path)
        if dropped:
            logger.info(f"Dropped page cache: {dropped/1024**3:.1f}GB")

        # Start preload
        t_pre = time.perf_counter()
        success, msg = engine.preload_for_reload(target_path)
        logger.info(f"preload_for_reload started in {time.perf_counter() - t_pre:.3f}s")

        # Wait for preload to complete
        logger.info("Waiting for preload...")
        time.sleep(70)  # Allow enough time for large models

        # Record GPU mem before
        free_before, _ = torch.cuda.mem_get_info(0)

        # Reload
        t0 = time.perf_counter()
        success, msg = engine.reload_model(
            model_path=target_path,
            server_args_overrides=target_overrides,
        )
        reload_time = time.perf_counter() - t0

        # Record GPU mem after
        free_after, _ = torch.cuda.mem_get_info(0)

        if success:
            # Verify generation
            use_harmony = (target_name == "gpt-oss-120b")
            passed, text = verify_generation(
                engine, target["test_prompt"],
                target["expect_contains"], target_name, use_harmony=use_harmony,
            )
        else:
            passed = False
            text = f"RELOAD FAILED: {msg}"

        result = {
            "switch": i + 1,
            "target": target_name,
            "reload_time": reload_time,
            "success": success,
            "gen_passed": passed,
            "gen_text": text,
            "mem_before": free_before / 1024**3,
            "mem_after": free_after / 1024**3,
        }
        results.append(result)

        logger.info(
            f"Switch {i+1} to {target_name}: "
            f"reload={reload_time:.3f}s, gen={'PASS' if passed else 'FAIL'}, "
            f"text={text[:60]!r}"
        )

    engine.shutdown()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'#':>3} {'Model':<20} {'Reload':>8} {'Gen':>5} {'Text':<40}")
    print("-" * 70)
    for r in results:
        gen_status = "PASS" if r["gen_passed"] else "FAIL"
        print(
            f"{r['switch']:>3} {r['target']:<20} "
            f"{r['reload_time']:>7.3f}s {gen_status:>5} "
            f"{r['gen_text'][:40]}"
        )

    avg_reload = sum(r["reload_time"] for r in results) / len(results) if results else 0
    all_gen_pass = all(r["gen_passed"] for r in results)
    print(f"\nAvg reload time: {avg_reload:.3f}s")
    print(f"All generation tests: {'PASS' if all_gen_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
