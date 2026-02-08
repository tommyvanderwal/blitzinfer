#!/usr/bin/env python3
"""Debug FP8 weight name and shape mismatches between safetensor and vLLM."""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from collections import defaultdict

from blitzinfer.memory.fast_loader import parse_safetensor_header, get_tensor_info, get_safetensor_files

model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
model_path = Path(snapshot_download(model_name, local_files_only=True))

print("=" * 80)
print("SAFETENSOR WEIGHT ANALYSIS")
print("=" * 80)

# Parse safetensor headers
sf_weights = {}
sf_files = get_safetensor_files(str(model_path))
for sf_file in sf_files:
    _, header = parse_safetensor_header(str(sf_file))
    tensor_info = get_tensor_info(header)
    for name, info in tensor_info.items():
        sf_weights[name] = {
            'shape': info['shape'],
            'dtype': info['dtype'],
        }

print(f"Total safetensor weights: {len(sf_weights)}")

# Categorize by suffix
by_suffix = defaultdict(list)
for name in sf_weights:
    suffix = name.split('.')[-1]
    by_suffix[suffix].append(name)

print("\nWeight types by suffix:")
for suffix, names in sorted(by_suffix.items(), key=lambda x: -len(x[1])):
    ex = names[0]
    info = sf_weights[ex]
    print(f"  {suffix}: {len(names)} weights (e.g., {ex}: {info['shape']} {info['dtype']})")

# Load vLLM model with dummy weights to get parameter structure
print("\n" + "=" * 80)
print("VLLM MODEL PARAMETER ANALYSIS")
print("=" * 80)

from vllm import LLM

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

print(f"Total vLLM parameters: {len(model_params)}")

# Categorize by suffix
by_suffix = defaultdict(list)
for name in model_params:
    suffix = name.split('.')[-1]
    by_suffix[suffix].append(name)

print("\nParameter types by suffix:")
for suffix, names in sorted(by_suffix.items(), key=lambda x: -len(x[1])):
    ex = names[0]
    info = model_params[ex]
    print(f"  {suffix}: {len(names)} params (e.g., {ex}: {info['shape']} {info['dtype']})")

# Compare a specific layer
print("\n" + "=" * 80)
print("LAYER 0 COMPARISON")
print("=" * 80)

print("\nSafetensor layer 0 weights:")
for name in sorted(sf_weights.keys()):
    if 'layers.0.' in name and 'language_model' in name:
        info = sf_weights[name]
        print(f"  {name}: {info['shape']} {info['dtype']}")

print("\nvLLM layer 0 parameters:")
for name in sorted(model_params.keys()):
    if 'layers.0.' in name and 'language_model' in name:
        info = model_params[name]
        print(f"  {name}: {info['shape']} {info['dtype']}")

# Try to find direct matches
print("\n" + "=" * 80)
print("MATCHING ANALYSIS")
print("=" * 80)

matches = 0
mismatches = 0
unmatched_sf = []
unmatched_vllm = []

sf_set = set(sf_weights.keys())
model_set = set(model_params.keys())

# Direct matches
direct = sf_set & model_set
for name in direct:
    sf_info = sf_weights[name]
    m_info = model_params[name]
    if sf_info['shape'] == m_info['shape']:
        matches += 1
    else:
        mismatches += 1
        if mismatches <= 10:
            print(f"Shape mismatch: {name}")
            print(f"  SF: {sf_info['shape']} {sf_info['dtype']}")
            print(f"  vLLM: {m_info['shape']} {m_info['dtype']}")

print(f"\nDirect name matches: {len(direct)}")
print(f"  Shape matches: {matches}")
print(f"  Shape mismatches: {mismatches}")

# Check model. prefix
model_prefixed = set('model.' + n for n in sf_weights.keys())
with_prefix = model_prefixed & model_set
print(f"\nWith 'model.' prefix: {len(with_prefix)} matches")

# FP8 specific: check if vLLM has separate weight/scale params
print("\n" + "=" * 80)
print("FP8-SPECIFIC ANALYSIS")
print("=" * 80)

# Check FP8 weight tensors in safetensor
fp8_weights = [(n, sf_weights[n]) for n in sf_weights if sf_weights[n]['dtype'] == torch.float8_e4m3fn]
print(f"\nFP8 weights in safetensor: {len(fp8_weights)}")
if fp8_weights:
    for name, info in fp8_weights[:5]:
        print(f"  {name}: {info['shape']}")

# Check corresponding vLLM params
print("\nCorresponding vLLM params for layer 0 MLP:")
mlp_params = [(n, model_params[n]) for n in model_params if 'layers.0.mlp' in n]
for name, info in sorted(mlp_params):
    print(f"  {name}: {info['shape']} {info['dtype']}")

# Check for weight vs weight_packed naming
print("\nLooking for 'weight' vs 'weight_packed' patterns:")
for name in sorted(model_params.keys()):
    if 'weight' in name and 'layers.0.' in name and 'mlp' in name:
        info = model_params[name]
        print(f"  vLLM: {name}: {info['shape']}")

        # Look for corresponding SF weight
        base_name = name.replace('model.', '').replace('language_model.', '')
        for sf_name in sf_weights:
            if base_name in sf_name or sf_name in base_name:
                sf_info = sf_weights[sf_name]
                print(f"    SF match?: {sf_name}: {sf_info['shape']}")

del llm
print("\nDone")
