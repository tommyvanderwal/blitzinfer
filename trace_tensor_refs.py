#!/usr/bin/env python3
"""
Trace what's holding GPU tensors after LLM deletion.
"""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

import sys
import types
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import gc
import torch

print("="*70)
print("TRACING TENSOR REFERENCES")
print("="*70)

# Import and load
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

print("\n>>> Loading model...")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.65,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=4 * 1024**3,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
)

# Quick inference
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
print(f"Output: {outputs[0].outputs[0].text}")

# Delete
print("\n>>> Deleting LLM...")
del llm
gc.collect()

# Cleanup
print(">>> Running vLLM cleanup...")
try:
    cleanup_dist_env_and_memory(shutdown_ray=False)
except Exception as e:
    print(f"  Warning: {e}")
gc.collect()
torch.cuda.empty_cache()

# Find GPU tensors
print("\n>>> Finding GPU tensors and their referrers...")
gpu_tensors = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            gpu_tensors.append(obj)
    except:
        pass

print(f"Found {len(gpu_tensors)} GPU tensors")

# For the largest tensors, find what's referencing them
print("\n>>> Tracing references for largest tensors...")
gpu_tensors.sort(key=lambda t: t.numel(), reverse=True)

for i, tensor in enumerate(gpu_tensors[:5]):
    print(f"\n--- Tensor {i+1}: shape={tuple(tensor.shape)}, size={tensor.numel() * tensor.element_size() / (1024**2):.1f} MB ---")

    referrers = gc.get_referrers(tensor)
    print(f"  {len(referrers)} direct referrers:")

    for j, ref in enumerate(referrers[:10]):
        ref_type = type(ref).__name__

        if isinstance(ref, dict):
            # Find keys that point to this tensor
            keys = [k for k, v in ref.items() if v is tensor]
            if keys:
                print(f"    {j+1}. dict with keys: {keys[:3]}...")
            else:
                print(f"    {j+1}. dict (indirect reference)")

        elif isinstance(ref, (list, tuple)):
            print(f"    {j+1}. {ref_type} of length {len(ref)}")

        elif hasattr(ref, '__class__'):
            module_name = ref.__class__.__module__
            class_name = ref.__class__.__name__
            print(f"    {j+1}. {module_name}.{class_name}")

            # If it's a torch.nn.Module, print more info
            if isinstance(ref, torch.nn.Module):
                print(f"        Module type: {type(ref)}")

        else:
            print(f"    {j+1}. {ref_type}")

# Check for vLLM global state
print("\n" + "="*70)
print("CHECKING VLLM GLOBAL STATE")
print("="*70)

# Check parallel_state
print("\n>>> Checking vllm.distributed.parallel_state...")
from vllm.distributed import parallel_state as ps
attrs_to_check = [
    '_world_size', '_rank', '_local_rank',
    '_TP_GROUP', '_PP_GROUP', '_DP_GROUP',
    '_DEVICE_WORLD_GROUP', '_CPU_WORLD_GROUP',
]
for attr in dir(ps):
    if not attr.startswith('__'):
        try:
            val = getattr(ps, attr)
            if val is not None and not callable(val):
                if isinstance(val, (int, str, bool)):
                    print(f"  {attr} = {val}")
                elif torch.is_tensor(val):
                    print(f"  {attr} = tensor {tuple(val.shape)}")
                else:
                    print(f"  {attr} = {type(val).__name__}")
        except:
            pass

# Check for model registry
print("\n>>> Checking for model registries...")
try:
    from vllm.model_executor.models import _MODELS
    print(f"  _MODELS registry: {len(_MODELS)} entries")
except:
    pass

# Check compiled model caches
print("\n>>> Checking torch._dynamo state...")
try:
    import torch._dynamo
    if hasattr(torch._dynamo, 'reset'):
        print("  Resetting torch._dynamo...")
        torch._dynamo.reset()
except Exception as e:
    print(f"  {e}")

# Check triton caches
print("\n>>> Checking triton caches...")
try:
    import triton
    import triton.compiler
    if hasattr(triton.compiler, 'CompiledKernel'):
        print(f"  triton.compiler.CompiledKernel exists")
except Exception as e:
    print(f"  {e}")

# Final attempt: look at all modules for CUDA tensors
print("\n>>> Scanning all loaded modules for GPU tensors...")
cuda_holding_modules = []
for name, module in sys.modules.items():
    if module is None:
        continue
    try:
        for attr in dir(module):
            try:
                val = getattr(module, attr)
                if torch.is_tensor(val) and val.is_cuda:
                    cuda_holding_modules.append((name, attr, tuple(val.shape)))
            except:
                pass
    except:
        pass

if cuda_holding_modules:
    print(f"  Found {len(cuda_holding_modules)} module-level GPU tensors:")
    for name, attr, shape in cuda_holding_modules[:20]:
        print(f"    {name}.{attr}: {shape}")
else:
    print("  No module-level GPU tensors found")

# Check what Python objects are holding the most memory
print("\n>>> Top object types by count in gc...")
type_counts = {}
for obj in gc.get_objects():
    t = type(obj).__name__
    type_counts[t] = type_counts.get(t, 0) + 1

sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
print("  Top 20 types:")
for t, count in sorted_types[:20]:
    print(f"    {t}: {count}")
