#!/usr/bin/env python3
"""
Memory cleanup fix for vLLM on ROCm 780M - Version 3.

This version directly resizes tensor storage to 0 to force GPU memory release.
"""

import gc
import sys
import torch


def get_memory_gb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3)


def force_release_gpu_tensors():
    """
    Force release GPU memory by resizing tensor storage to 0.
    This is aggressive but necessary when references can't be broken.
    """
    released = 0
    total_bytes = 0

    for _ in range(5):  # Multiple passes
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                    tensors.append(obj)
            except:
                pass

        if not tensors:
            break

        for tensor in tensors:
            try:
                bytes_before = tensor.numel() * tensor.element_size()
                # Resize storage to 0 - this should release GPU memory
                tensor.storage().resize_(0)
                total_bytes += bytes_before
                released += 1
            except Exception as e:
                # Some tensors may not allow resizing
                pass

        gc.collect()
        torch.cuda.empty_cache()

    return released, total_bytes / (1024**3)


def full_cleanup_v3():
    """
    Complete cleanup with forced tensor release.
    """
    # Step 1: Standard vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass

    gc.collect()

    # Step 2: Reset torch dynamo
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except:
        pass

    gc.collect()

    # Step 3: Force release GPU tensors
    released, gb_released = force_release_gpu_tensors()

    # Step 4: Clear CUDA caches
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    gc.collect()

    return released, gb_released


def test_forced_release():
    """Test the forced release approach."""
    import os
    import time
    import types

    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    fake_meta = types.ModuleType('torchvision._meta_registrations')
    sys.modules['torchvision._meta_registrations'] = fake_meta
    sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

    print("=" * 70)
    print("TESTING FORCED TENSOR RELEASE")
    print("=" * 70)

    initial = get_memory_gb()
    print(f"\nInitial: {initial:.2f} GB free")

    # First model
    print("\n>>> Loading first model...")
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

    after_load = get_memory_gb()
    print(f"After load: {after_load:.2f} GB free (used {initial - after_load:.2f} GB)")

    # Generate
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Delete first
    print("\n>>> Deleting first model...")
    del llm
    del out
    gc.collect()

    # Force cleanup
    print(">>> Running forced tensor release...")
    released, gb = full_cleanup_v3()
    print(f"Force-released {released} tensors ({gb:.2f} GB)")

    time.sleep(1.0)
    gc.collect()
    torch.cuda.empty_cache()

    after_cleanup = get_memory_gb()
    leaked = initial - after_cleanup
    print(f"\nAfter cleanup: {after_cleanup:.2f} GB free")
    print(f"Memory leaked: {leaked:.2f} GB")

    # Check remaining tensors
    remaining = sum(1 for obj in gc.get_objects()
                    if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0)
    print(f"Remaining GPU tensors with data: {remaining}")

    if leaked < 2.0:
        print("\n>>> Cleanup SUCCESS! Trying second model...")

        llm2 = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            gpu_memory_utilization=0.50,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            kv_cache_memory_bytes=4 * 1024**3,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )

        out2 = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {out2[0].outputs[0].text}")

        print("\n" + "=" * 70)
        print("SUCCESS! Model switching works with forced release!")
        print("=" * 70)

        del llm2
        del out2
        gc.collect()
    else:
        print(f"\n>>> Cleanup FAILED - still leaking {leaked:.2f} GB")
        print("\nDIAGNOSTIC: Checking what's holding memory...")

        # Check if there are tensors with 0 storage but allocated memory
        zero_storage = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    if obj.storage().size() == 0:
                        zero_storage += 1
            except:
                pass
        print(f"Tensors with zero storage: {zero_storage}")

        # The issue is likely at HIP/ROCR level
        print("\nThe GPU memory is held at the HIP/ROCR driver level.")
        print("This is a ROCm bug, not a Python/vLLM issue.")


if __name__ == '__main__':
    test_forced_release()
