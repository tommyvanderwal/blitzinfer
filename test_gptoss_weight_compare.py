#!/usr/bin/env python3
"""Compare gpt-oss-120b model state: fresh load vs reload.

Diagnostic: compute checksums of all model params/buffers/attrs after fresh load,
then after reload from Qwen, and compare. This tells us if weights are corrupted.
"""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import gc
import hashlib
import time
import torch

GPT_OSS = "openai/gpt-oss-120b"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GPT_OSS_OVERRIDES = {
    "context_length": 131072,
    "moe_runner_backend": "triton_kernel",
    "dtype": "bfloat16",
}


def snapshot_model_state(engine):
    """Snapshot all model params, buffers, and triton wrapper tensors.

    Returns dict of {name: (shape, dtype, checksum)} for comparison.
    """
    # Access the model through the scheduler -> tp_worker -> model_runner
    # In single-process mode with InprocClient, we need to navigate:
    # engine.engine_core (InprocClient) -> .engine_core (EngineCore) -> ...
    try:
        core = engine.engine_core
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        model_runner = core.scheduler.tp_worker.model_runner
        model = model_runner.model
    except Exception as e:
        print(f"  Cannot access model internals: {e}")
        return {}

    state = {}

    # 1. Named parameters
    for name, param in model.named_parameters():
        t = param.data
        if t.numel() > 0:
            # Use sum as a quick checksum (faster than full hash for large tensors)
            cksum = t.float().sum().item()
            state[f"param:{name}"] = (tuple(t.shape), str(t.dtype), cksum)

    # 2. Named buffers
    for name, buf in model.named_buffers():
        if buf.numel() > 0:
            cksum = buf.float().sum().item()
            state[f"buffer:{name}"] = (tuple(buf.shape), str(buf.dtype), cksum)

    # 3. Triton wrapper tensors (mxfp4 swizzled weights) - walk __dict__
    n_triton = 0
    visited = set()

    def walk_for_tensors(obj, prefix, depth=0):
        nonlocal n_triton
        obj_id = id(obj)
        if obj_id in visited or depth > 5:
            return
        visited.add(obj_id)

        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict is None or not isinstance(obj_dict, dict):
            return

        for attr_name, val in obj_dict.items():
            if val is None or isinstance(val, (str, int, float, bool, type, bytes,
                                                torch.nn.Module, torch.dtype, torch.device)):
                continue

            # Direct tensor
            if isinstance(val, torch.Tensor) and val.numel() > 0:
                key = f"attr:{prefix}.{attr_name}"
                cksum = val.float().sum().item()
                state[key] = (tuple(val.shape), str(val.dtype), cksum)
                n_triton += 1
                continue

            # Check for triton wrapper with .data
            val_dict = getattr(val, '__dict__', None)
            if val_dict and isinstance(val_dict, dict):
                data_attr = val_dict.get('data')
                if isinstance(data_attr, torch.Tensor) and data_attr.numel() > 0:
                    key = f"triton:{prefix}.{attr_name}.data"
                    cksum = data_attr.float().sum().item()
                    state[key] = (tuple(data_attr.shape), str(data_attr.dtype), cksum)
                    n_triton += 1
                # Recurse
                walk_for_tensors(val, f"{prefix}.{attr_name}", depth + 1)

    for mod_name, module in model.named_modules():
        walk_for_tensors(module, mod_name)

    print(f"  Snapshot: {len(state)} entries ({n_triton} triton wrappers)")
    return state


def compare_snapshots(snap_a, snap_b, label_a="Fresh", label_b="Reload"):
    """Compare two model snapshots. Print differences."""
    all_keys = set(snap_a.keys()) | set(snap_b.keys())

    only_a = set(snap_a.keys()) - set(snap_b.keys())
    only_b = set(snap_b.keys()) - set(snap_a.keys())

    if only_a:
        print(f"\n  Only in {label_a} ({len(only_a)} entries):")
        for k in sorted(only_a)[:20]:
            print(f"    {k}: {snap_a[k][:2]}")
        if len(only_a) > 20:
            print(f"    ... and {len(only_a) - 20} more")

    if only_b:
        print(f"\n  Only in {label_b} ({len(only_b)} entries):")
        for k in sorted(only_b)[:20]:
            print(f"    {k}: {snap_b[k][:2]}")
        if len(only_b) > 20:
            print(f"    ... and {len(only_b) - 20} more")

    common = set(snap_a.keys()) & set(snap_b.keys())
    mismatches = []
    for k in sorted(common):
        shape_a, dtype_a, cksum_a = snap_a[k]
        shape_b, dtype_b, cksum_b = snap_b[k]

        if shape_a != shape_b:
            mismatches.append((k, f"shape: {shape_a} vs {shape_b}"))
        elif dtype_a != dtype_b:
            mismatches.append((k, f"dtype: {dtype_a} vs {dtype_b}"))
        elif abs(cksum_a - cksum_b) > 1e-3:
            mismatches.append((k, f"checksum: {cksum_a:.6f} vs {cksum_b:.6f}"))

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)} entries):")
        for k, diff in mismatches[:30]:
            print(f"    {k}: {diff}")
        if len(mismatches) > 30:
            print(f"    ... and {len(mismatches) - 30} more")
    else:
        print(f"\n  All {len(common)} common entries MATCH!")

    return len(mismatches), len(only_a), len(only_b)


def test_generate(engine, label):
    """Quick generation test."""
    result = engine.generate(
        prompt="What is 2+3? Answer with just the number.",
        sampling_params={"temperature": 0, "max_new_tokens": 50},
    )
    text = result.get("text", "")
    print(f"  [{label}] text: {text[:100]!r}")
    return text


def main():
    from sglang.srt.entrypoints.engine import Engine

    # ========================================
    # Step 1: Fresh load gpt-oss-120b
    # ========================================
    print("\n=== STEP 1: Fresh load gpt-oss-120b ===")
    t0 = time.perf_counter()
    engine = Engine(
        model_path=GPT_OSS,
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
        context_length=131072,
        dtype="bfloat16",
        moe_runner_backend="triton_kernel",
    )
    print(f"  Fresh load took {time.perf_counter()-t0:.1f}s")

    # Test generation
    text_fresh = test_generate(engine, "fresh")

    # Snapshot weights
    print("\n  Taking snapshot of fresh model...")
    snap_fresh = snapshot_model_state(engine)

    # ========================================
    # Step 2: Reload to Qwen
    # ========================================
    print("\n=== STEP 2: Reload to Qwen ===")
    t0 = time.perf_counter()
    ok, msg = engine.reload_model(QWEN)
    print(f"  Reload to Qwen: ok={ok}, time={time.perf_counter()-t0:.1f}s")
    if not ok:
        print(f"  Error: {msg[:200]}")
        engine.shutdown()
        return

    text_qwen = test_generate(engine, "qwen")

    # ========================================
    # Step 3: Reload back to gpt-oss
    # ========================================
    print("\n=== STEP 3: Reload back to gpt-oss ===")
    t0 = time.perf_counter()
    ok, msg = engine.reload_model(
        GPT_OSS,
        server_args_overrides=GPT_OSS_OVERRIDES,
    )
    print(f"  Reload to gpt-oss: ok={ok}, time={time.perf_counter()-t0:.1f}s")
    if not ok:
        print(f"  Error: {msg[:200]}")
        engine.shutdown()
        return

    # Test generation
    text_reload = test_generate(engine, "reload")

    # Snapshot weights
    print("\n  Taking snapshot of reloaded model...")
    snap_reload = snapshot_model_state(engine)

    # ========================================
    # Step 4: Compare snapshots
    # ========================================
    print("\n=== STEP 4: Compare fresh vs reload weights ===")
    n_mismatch, n_only_fresh, n_only_reload = compare_snapshots(snap_fresh, snap_reload)

    print(f"\n=== SUMMARY ===")
    print(f"  Fresh output: {text_fresh[:80]!r}")
    print(f"  Reload output: {text_reload[:80]!r}")
    print(f"  Outputs match: {text_fresh == text_reload}")
    print(f"  Weight mismatches: {n_mismatch}")
    print(f"  Only in fresh: {n_only_fresh}")
    print(f"  Only in reload: {n_only_reload}")

    engine.shutdown()
    print("\nDone")


if __name__ == "__main__":
    main()
