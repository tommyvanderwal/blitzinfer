#!/usr/bin/env python3
"""Debug which weights are skipped during injection."""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import gc
import time
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

# Initialize CUDA
_ = torch.randn(1000, device='cuda')
gc.collect()
torch.cuda.empty_cache()

from blitzinfer.memory import (
    PinnedMemoryArena,
    load_model_to_arena,
    get_model_size,
)

model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
model_path = Path(snapshot_download(model_name, local_files_only=True))
model_size = get_model_size(str(model_path))
model_gb = model_size / 1e9

print(f"Model: {model_name}")
print(f"Size: {model_gb:.1f} GB")

# Load into arena
arena_size_gb = model_gb + 5
arena = PinnedMemoryArena(arena_size_gb)
load_model_to_arena(str(model_path), arena, model_name)

# Get pinned tensors
pinned_tensors = arena.get_all_tensors(model_name)
print(f"\nPinned tensors: {len(pinned_tensors)}")

# Load vLLM model with dummy weights
from vllm import LLM

print("\nLoading vLLM with dummy weights...")
llm = LLM(
    model=model_name,
    load_format="dummy",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    max_num_seqs=4,
    enforce_eager=True,
    trust_remote_code=True,
)

# Get model parameters
model_params = {}
try:
    engine_core = llm.llm_engine.engine_core
    if hasattr(engine_core, 'engine_core'):
        core = engine_core.engine_core
    else:
        core = engine_core

    if hasattr(core, 'model_executor'):
        executor = core.model_executor
        if hasattr(executor, 'driver_worker'):
            worker = executor.driver_worker
            if hasattr(worker, 'worker') and worker.worker is not None:
                model_runner = getattr(worker.worker, 'model_runner', None)
                if model_runner and hasattr(model_runner, 'model'):
                    for name, param in model_runner.model.named_parameters():
                        model_params[name] = {
                            'shape': tuple(param.shape),
                            'dtype': param.dtype,
                        }
except Exception as e:
    print(f"Error getting model params: {e}")

print(f"vLLM parameters: {len(model_params)}")

# Now try the matching logic from pinned_loader
import re
from typing import Dict, Optional

def normalize_weight_name(name: str) -> str:
    transformations = [
        ("model.language_model.", "language_model.model."),
        ("language_model.model.", "model.language_model."),
        ("model.", ""),
    ]
    for old_prefix, new_prefix in transformations:
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]
    return name

def same_layer(name1: str, name2: str) -> bool:
    pattern = r'(?:layers|blocks)\.(\d+)\.'
    match1 = re.search(pattern, name1)
    match2 = re.search(pattern, name2)
    if match1 and match2:
        return match1.group(1) == match2.group(1)
    return False

# Build normalized preloaded
normalized_preloaded = {}
for name, tensor in pinned_tensors.items():
    norm_name = normalize_weight_name(name)
    normalized_preloaded[norm_name] = (name, tensor)
    normalized_preloaded[name] = (name, tensor)

# Check each vLLM param
matched = []
merged = []
skipped = []

merge_patterns = [
    ("qkv_proj", ["q_proj", "k_proj", "v_proj"], 0),
    ("gate_up_proj", ["gate_proj", "up_proj"], 0),
]

for param_name, param_info in model_params.items():
    weight = None

    # Try direct match
    norm_param_name = normalize_weight_name(param_name)
    if norm_param_name in normalized_preloaded:
        matched.append(param_name)
        continue
    if param_name in normalized_preloaded:
        matched.append(param_name)
        continue

    # Try merge patterns
    found_merge = False
    for merged_part, components, concat_dim in merge_patterns:
        if merged_part not in param_name:
            continue

        merged_idx = param_name.find(merged_part)
        base_path = param_name[:merged_idx]
        suffix = param_name[merged_idx + len(merged_part):]

        all_found = True
        for comp in components:
            comp_name = base_path + comp + suffix
            norm_comp_name = normalize_weight_name(comp_name)

            if norm_comp_name not in normalized_preloaded and comp_name not in normalized_preloaded:
                # Try finding in preloaded
                found = False
                for pname in pinned_tensors:
                    if comp in pname and suffix in pname:
                        if same_layer(param_name, pname):
                            found = True
                            break
                if not found:
                    all_found = False
                    break

        if all_found:
            merged.append(param_name)
            found_merge = True
            break

    if not found_merge:
        skipped.append(param_name)

print(f"\n=== MATCHING RESULTS ===")
print(f"Direct matched: {len(matched)}")
print(f"Merged: {len(merged)}")
print(f"Skipped: {len(skipped)}")

if skipped:
    print(f"\nSkipped params:")
    for p in sorted(skipped):
        info = model_params[p]
        print(f"  {p}: {info['shape']} {info['dtype']}")

        # Try to find similar names in preloaded
        p_parts = p.split('.')
        suffix = p_parts[-1]
        print(f"    Looking for '{suffix}' in preloaded...")
        similar = [n for n in pinned_tensors if suffix in n][:3]
        for s in similar:
            t = pinned_tensors[s]
            print(f"      {s}: {t.shape} {t.dtype}")

# Verify some merged weights
print("\n=== MERGED WEIGHT VERIFICATION ===")
for param_name in merged[:5]:
    for merged_part, components, concat_dim in merge_patterns:
        if merged_part in param_name:
            merged_idx = param_name.find(merged_part)
            base_path = param_name[:merged_idx]
            suffix = param_name[merged_idx + len(merged_part):]

            print(f"\n{param_name}")
            print(f"  vLLM shape: {model_params[param_name]['shape']}")

            comp_shapes = []
            for comp in components:
                comp_name = base_path + comp + suffix
                norm_comp = normalize_weight_name(comp_name)
                if norm_comp in normalized_preloaded:
                    _, t = normalized_preloaded[norm_comp]
                    comp_shapes.append(t.shape)
                    print(f"  {comp}: {t.shape}")

            if comp_shapes:
                # Calculate merged shape
                total_dim0 = sum(s[0] for s in comp_shapes)
                print(f"  Expected merged: ({total_dim0}, {comp_shapes[0][1]})")
            break

del llm
arena.clear()
print("\nDone")
