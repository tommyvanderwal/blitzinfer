#!/usr/bin/env python3
"""Debug weight name mismatches between safetensor and vLLM."""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from blitzinfer.memory import PinnedMemoryArena, load_model_to_arena

model_name = "openai/gpt-oss-120b"
model_path = Path(snapshot_download(model_name, local_files_only=True))

# Get first few tensor names from arena
print("Loading minimal amount into arena...")
arena = PinnedMemoryArena(5)  # Just 5GB for metadata

# Parse safetensor headers to get tensor names without loading
from blitzinfer.memory.fast_loader import parse_safetensor_header, get_safetensor_files

sf_files = get_safetensor_files(str(model_path))
safetensor_names = []
for sf_file in sf_files[:2]:  # Just first 2 files
    header_size, header = parse_safetensor_header(str(sf_file))
    for name in header:
        if name != '__metadata__':
            safetensor_names.append(name)

print(f"\nSafetensor tensor names (first 20):")
for name in safetensor_names[:20]:
    print(f"  {name}")

# Get vLLM model parameter names
print("\n\nLoading vLLM model with dummy weights...")
from vllm import LLM

llm = LLM(
    model=model_name,
    load_format="dummy",  # Don't load real weights
    dtype="bfloat16",
    max_model_len=512,
    gpu_memory_utilization=0.10,
    max_num_seqs=1,
    enforce_eager=True,
    trust_remote_code=True,
)

# Get model parameters
engine_core = llm.llm_engine.engine_core
if hasattr(engine_core, 'engine_core'):
    core = engine_core.engine_core
else:
    core = engine_core

model_params = []
if hasattr(core, 'model_executor'):
    executor = core.model_executor
    if hasattr(executor, 'driver_worker'):
        worker = executor.driver_worker
        if hasattr(worker, 'worker') and worker.worker is not None:
            model_runner = getattr(worker.worker, 'model_runner', None)
            if model_runner and hasattr(model_runner, 'model'):
                for name, param in model_runner.model.named_parameters():
                    model_params.append((name, param.shape, param.dtype))

print(f"\nvLLM model parameter names (first 20):")
for name, shape, dtype in model_params[:20]:
    print(f"  {name} {shape} {dtype}")

# Find common patterns
print("\n\nComparing naming patterns...")
sf_set = set(safetensor_names)
model_set = set(n for n, s, d in model_params)

# Direct matches
direct_matches = sf_set & model_set
print(f"Direct matches: {len(direct_matches)}")

# Try adding 'model.' prefix
model_prefixed = set('model.' + n for n in safetensor_names)
with_prefix_matches = model_prefixed & model_set
print(f"With 'model.' prefix: {len(with_prefix_matches)}")

# Show sample mismatches
print("\nSample safetensor names without match:")
unmatched_sf = [n for n in safetensor_names[:50] if n not in model_set and 'model.' + n not in model_set]
for n in unmatched_sf[:10]:
    print(f"  SF: {n}")

print("\nSample vLLM names without match:")
unmatched_model = [n for n, s, d in model_params[:50] if n not in sf_set and n.replace('model.', '') not in sf_set]
for n in unmatched_model[:10]:
    print(f"  vLLM: {n}")

del llm
print("\nDone")
