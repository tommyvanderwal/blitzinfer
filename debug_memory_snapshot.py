#!/usr/bin/env python3
"""
Debug memory issues using PyTorch's memory snapshot tools.
This helps visualize where GPU memory is being held.

Usage:
    python3 debug_memory_snapshot.py
    # Then open https://pytorch.org/memory_viz and drag the .pickle file
"""

import os
import sys
import gc
import time
import types

# Environment setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_memory_gb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3), total / (1024**3)


def main():
    print("=" * 70)
    print("MEMORY SNAPSHOT DEBUGGING")
    print("=" * 70)

    # Enable memory history recording
    print("\n>>> Enabling memory history recording...")
    try:
        torch.cuda.memory._record_memory_history(
            enabled='all',
            context='all',
            stacks='all',
            max_entries=100000  # Limit entries to avoid huge files
        )
        print("Memory history recording enabled")
    except Exception as e:
        print(f"Warning: Could not enable memory history: {e}")
        print("Continuing without detailed traces...")

    initial_free, total = get_memory_gb()
    print(f"\nInitial: {initial_free:.2f} GB free / {total:.2f} GB total")

    # Import and load vLLM
    print("\n>>> Loading vLLM model...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.50,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=4 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    after_load, _ = get_memory_gb()
    print(f"After load: {after_load:.2f} GB free (used {initial_free - after_load:.2f} GB)")

    # Take snapshot before deletion
    print("\n>>> Taking snapshot before deletion...")
    try:
        torch.cuda.memory._dump_snapshot("memory_before_delete.pickle")
        print("Saved: memory_before_delete.pickle")
    except Exception as e:
        print(f"Warning: {e}")

    # Run inference
    print("\n>>> Running inference...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {outputs[0].outputs[0].text}")

    # Delete model
    print("\n>>> Deleting model...")
    del llm
    del outputs
    gc.collect()

    # Cleanup
    print("\n>>> Running cleanup...")
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    time.sleep(1.0)
    gc.collect()
    torch.cuda.empty_cache()

    after_cleanup, _ = get_memory_gb()
    leaked = initial_free - after_cleanup
    print(f"\nAfter cleanup: {after_cleanup:.2f} GB free")
    print(f"Memory leaked: {leaked:.2f} GB")

    # Take snapshot after deletion
    print("\n>>> Taking snapshot after deletion...")
    try:
        torch.cuda.memory._dump_snapshot("memory_after_delete.pickle")
        print("Saved: memory_after_delete.pickle")
    except Exception as e:
        print(f"Warning: {e}")

    # Stop recording
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except:
        pass

    # Analyze remaining tensors
    print("\n>>> Analyzing remaining GPU tensors...")
    gpu_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                gpu_tensors.append(obj)
        except:
            pass

    if gpu_tensors:
        print(f"Found {len(gpu_tensors)} GPU tensors still in memory")
        total_bytes = sum(t.numel() * t.element_size() for t in gpu_tensors)
        print(f"Total size: {total_bytes / (1024**3):.2f} GB")

        # Group by shape
        shape_counts = {}
        for t in gpu_tensors:
            shape = tuple(t.shape)
            if shape not in shape_counts:
                shape_counts[shape] = {'count': 0, 'bytes': 0}
            shape_counts[shape]['count'] += 1
            shape_counts[shape]['bytes'] += t.numel() * t.element_size()

        print("\nTop tensor shapes by size:")
        sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1]['bytes'], reverse=True)
        for shape, info in sorted_shapes[:10]:
            mb = info['bytes'] / (1024**2)
            print(f"  {shape}: {info['count']} tensors, {mb:.1f} MB")

        # Trace referrers for largest tensors
        print("\n>>> Tracing referrers for largest tensors...")
        gpu_tensors.sort(key=lambda t: t.numel(), reverse=True)

        for i, tensor in enumerate(gpu_tensors[:3]):
            print(f"\nTensor {i+1}: shape={tuple(tensor.shape)}, "
                  f"size={tensor.numel() * tensor.element_size() / (1024**2):.1f} MB")

            referrers = gc.get_referrers(tensor)
            print(f"  Referrers: {len(referrers)}")

            for j, ref in enumerate(referrers[:5]):
                ref_type = type(ref).__name__
                if isinstance(ref, dict):
                    keys = [k for k, v in list(ref.items())[:100] if v is tensor]
                    print(f"    {j+1}. dict, keys pointing to tensor: {keys[:3]}")
                elif isinstance(ref, (list, tuple)):
                    print(f"    {j+1}. {ref_type} of length {len(ref)}")
                elif hasattr(ref, '__class__'):
                    print(f"    {j+1}. {ref.__class__.__module__}.{ref.__class__.__name__}")
                else:
                    print(f"    {j+1}. {ref_type}")

    else:
        print("No GPU tensors found in gc.get_objects()")
        print("Memory may be held at the HIP/ROCR driver level")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Open https://pytorch.org/memory_viz in your browser
2. Drag and drop memory_before_delete.pickle and memory_after_delete.pickle
3. Compare the memory allocations to identify what's not being freed

If no .pickle files were created, the issue is at the driver level,
not the PyTorch level.
    """)


if __name__ == '__main__':
    main()
