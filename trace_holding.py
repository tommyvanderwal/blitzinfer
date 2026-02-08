#!/usr/bin/env python3
"""
Trace what's holding tensors after del llm but before cleanup.
"""

import os
import sys
import gc
import time
import types

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def count_tensors():
    count = 0
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                total += obj.numel() * obj.element_size()
        except:
            pass
    return count, total / (1024**3)


print("=" * 70)
print("TRACING TENSOR REFERENCES")
print("=" * 70)

from vllm import LLM, SamplingParams

initial = get_mem()
print(f"\nInitial: {initial:.2f} GB")

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.30,
    max_model_len=512,
    kv_cache_memory_bytes=2 * 1024**3,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
)
out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
print(f"Output: {out[0].outputs[0].text}")

after_load = get_mem()
print(f"\nAfter load: {after_load:.2f} GB (used {initial - after_load:.2f} GB)")

count, size = count_tensors()
print(f"Tensors in gc: {count}, {size:.2f} GB")

# Delete LLM
print("\n>>> del llm")
del llm
del out

count, size = count_tensors()
print(f"Tensors in gc after del: {count}, {size:.2f} GB")
print(f"Memory: {get_mem():.2f} GB")

# gc.collect
print("\n>>> gc.collect()")
gc.collect()

count, size = count_tensors()
print(f"Tensors in gc after gc.collect: {count}, {size:.2f} GB")
print(f"Memory: {get_mem():.2f} GB")

# What's holding them?
print("\n>>> Finding what's holding tensors...")

# Find the largest tensor
largest = None
largest_size = 0
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
            size = obj.numel() * obj.element_size()
            if size > largest_size:
                largest_size = size
                largest = obj
    except:
        pass

if largest is not None:
    print(f"\nLargest tensor: {tuple(largest.shape)}, {largest_size / (1024**2):.1f} MB")

    refs = gc.get_referrers(largest)
    print(f"Referrers: {len(refs)}")

    for i, ref in enumerate(refs[:10]):
        print(f"\n  Ref {i+1}:")
        if isinstance(ref, dict):
            keys = [k for k, v in list(ref.items())[:50] if v is largest]
            print(f"    dict with keys pointing to tensor: {keys}")
            # What holds this dict?
            dict_refs = gc.get_referrers(ref)
            print(f"    dict is held by {len(dict_refs)} objects:")
            for j, dr in enumerate(dict_refs[:5]):
                if isinstance(dr, dict):
                    print(f"      {j+1}. another dict")
                elif hasattr(dr, '__class__'):
                    cn = f"{dr.__class__.__module__}.{dr.__class__.__name__}"
                    print(f"      {j+1}. {cn}")
                    # Is this an nn.Module?
                    if isinstance(dr, torch.nn.Module):
                        print(f"         Module type: {type(dr).__name__}")
                else:
                    print(f"      {j+1}. {type(dr)}")
        elif isinstance(ref, list):
            print(f"    list of length {len(ref)}")
        elif hasattr(ref, '__class__'):
            cn = f"{ref.__class__.__module__}.{ref.__class__.__name__}"
            print(f"    {cn}")
        else:
            print(f"    {type(ref)}")

# Now try vLLM cleanup
print("\n>>> vLLM cleanup_dist_env_and_memory()")
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
cleanup_dist_env_and_memory(shutdown_ray=False)

count, size = count_tensors()
print(f"Tensors in gc after vLLM cleanup: {count}, {size:.2f} GB")
print(f"Memory: {get_mem():.2f} GB")

# What's STILL holding them?
if count > 0:
    print("\n>>> What's still holding tensors?")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                refs = gc.get_referrers(obj)
                print(f"\nTensor {tuple(obj.shape)}: {len(refs)} referrers")
                for ref in refs[:3]:
                    if isinstance(ref, dict):
                        keys = list(ref.keys())[:5]
                        print(f"  dict with keys: {keys}")
                    elif hasattr(ref, '__class__'):
                        print(f"  {ref.__class__.__module__}.{ref.__class__.__name__}")
                break  # Just show one
        except:
            pass
