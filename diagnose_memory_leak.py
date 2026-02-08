#!/usr/bin/env python3
"""
Diagnose what's holding GPU memory after LLM deletion.

Goal: Find the ~20GB leak that prevents in-process model switching.
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
import time
import torch


def get_memory_info():
    """Get detailed GPU memory info."""
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "free_gb": free / (1024**3),
        "total_gb": total / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "used_gb": (total - free) / (1024**3),
    }


def print_memory(label):
    """Print memory state with label."""
    m = get_memory_info()
    print(f"\n[{label}]")
    print(f"  Free:      {m['free_gb']:.2f} GB")
    print(f"  Used:      {m['used_gb']:.2f} GB")
    print(f"  Allocated: {m['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {m['reserved_gb']:.2f} GB")
    return m


def count_tensors():
    """Count GPU tensors in memory."""
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                total_bytes += obj.numel() * obj.element_size()
        except:
            pass
    return count, total_bytes / (1024**3)


print("="*70)
print("GPU MEMORY LEAK DIAGNOSIS")
print("="*70)

# Initial state
initial = print_memory("Initial (before any imports)")
tensor_count, tensor_gb = count_tensors()
print(f"  GPU tensors: {tensor_count} ({tensor_gb:.2f} GB)")

# Import vLLM
print("\n>>> Importing vLLM...")
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

after_import = print_memory("After vLLM import")
tensor_count, tensor_gb = count_tensors()
print(f"  GPU tensors: {tensor_count} ({tensor_gb:.2f} GB)")

# Load model
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

after_load = print_memory("After model load")
tensor_count, tensor_gb = count_tensors()
print(f"  GPU tensors: {tensor_count} ({tensor_gb:.2f} GB)")

# Quick inference to warm up
print("\n>>> Running inference to warm up...")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
print(f"  Output: {outputs[0].outputs[0].text}")

after_inference = print_memory("After inference")

# Now try to clean up
print("\n" + "="*70)
print("CLEANUP ATTEMPTS")
print("="*70)

# Step 1: Delete LLM reference
print("\n>>> Step 1: del llm")
del llm
gc.collect()
step1 = print_memory("After del llm + gc.collect()")
tensor_count, tensor_gb = count_tensors()
print(f"  GPU tensors: {tensor_count} ({tensor_gb:.2f} GB)")

# Step 2: vLLM cleanup
print("\n>>> Step 2: cleanup_dist_env_and_memory()")
try:
    cleanup_dist_env_and_memory(shutdown_ray=False)
except Exception as e:
    print(f"  Warning: {e}")
gc.collect()
step2 = print_memory("After vLLM cleanup")
tensor_count, tensor_gb = count_tensors()
print(f"  GPU tensors: {tensor_count} ({tensor_gb:.2f} GB)")

# Step 3: PyTorch cache clearing
print("\n>>> Step 3: torch.cuda.empty_cache()")
torch.cuda.empty_cache()
step3 = print_memory("After empty_cache()")

# Step 4: Reset peak stats and synchronize
print("\n>>> Step 4: synchronize + reset_peak_memory_stats()")
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
gc.collect()
step4 = print_memory("After sync + reset")

# Step 5: Check for remaining CUDA tensors
print("\n>>> Step 5: Hunting for remaining GPU tensors...")
remaining_tensors = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            remaining_tensors.append({
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "bytes": obj.numel() * obj.element_size(),
                "device": str(obj.device),
            })
    except:
        pass

if remaining_tensors:
    print(f"  Found {len(remaining_tensors)} GPU tensors still in memory:")
    # Sort by size
    remaining_tensors.sort(key=lambda x: x["bytes"], reverse=True)
    total_remaining = sum(t["bytes"] for t in remaining_tensors)
    print(f"  Total size: {total_remaining / (1024**3):.2f} GB")
    print("  Top 10 largest:")
    for i, t in enumerate(remaining_tensors[:10]):
        print(f"    {i+1}. {t['shape']} {t['dtype']} = {t['bytes']/(1024**2):.1f} MB")
else:
    print("  No GPU tensors found in gc.get_objects()")

# Step 6: Try to clear Triton cache
print("\n>>> Step 6: Clear Triton cache...")
try:
    import triton
    if hasattr(triton, 'runtime') and hasattr(triton.runtime, 'cache'):
        triton.runtime.cache.clear()
        print("  Cleared triton.runtime.cache")
except Exception as e:
    print(f"  Could not clear Triton cache: {e}")

gc.collect()
torch.cuda.empty_cache()
step6 = print_memory("After Triton cache clear")

# Step 7: Check torch.cuda allocator state
print("\n>>> Step 7: PyTorch CUDA allocator stats...")
try:
    stats = torch.cuda.memory_stats()
    print(f"  Allocated blocks: {stats.get('allocated_bytes.all.current', 0) / (1024**3):.2f} GB")
    print(f"  Reserved blocks:  {stats.get('reserved_bytes.all.current', 0) / (1024**3):.2f} GB")
    print(f"  Active blocks:    {stats.get('active_bytes.all.current', 0) / (1024**3):.2f} GB")
except Exception as e:
    print(f"  Could not get stats: {e}")

# Step 8: Try IPC handle cleanup (for shared memory)
print("\n>>> Step 8: Check for IPC handles...")
try:
    # This is a hack to try to release any shared memory handles
    import torch.multiprocessing as mp
    # Nothing specific to clean here in main process
    print("  No specific IPC cleanup available")
except Exception as e:
    print(f"  {e}")

# Final aggressive cleanup
print("\n>>> Step 9: Final aggressive cleanup...")
gc.collect()
gc.collect()
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # Clean up IPC handles
final = print_memory("Final state")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Initial free:     {initial['free_gb']:.2f} GB")
print(f"After load:       {after_load['free_gb']:.2f} GB (used {initial['free_gb'] - after_load['free_gb']:.2f} GB)")
print(f"Final free:       {final['free_gb']:.2f} GB")
print(f"Memory leaked:    {initial['free_gb'] - final['free_gb']:.2f} GB")
print(f"Recovery rate:    {(final['free_gb'] - after_load['free_gb']) / (initial['free_gb'] - after_load['free_gb']) * 100:.1f}%")

if initial['free_gb'] - final['free_gb'] > 1.0:
    print("\n⚠️  SIGNIFICANT MEMORY LEAK DETECTED")
    print("   This prevents in-process model switching.")
else:
    print("\n✓ Memory properly released - in-process switching should work!")
