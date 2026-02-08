#!/usr/bin/env python3
"""
Aggressive cleanup to release GPU memory after LLM deletion.
Attempts to find and break all circular references holding model weights.
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


def get_memory():
    """Get free GPU memory in GB."""
    return torch.cuda.mem_get_info()[0] / (1024**3)


def aggressive_cleanup():
    """
    Aggressively clean up all vLLM-related GPU tensors.
    This goes beyond vLLM's built-in cleanup.
    """
    print("\n>>> Starting aggressive cleanup...")

    # Step 1: Standard vLLM cleanup
    print("  Step 1: vLLM cleanup_dist_env_and_memory...")
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception as e:
        print(f"    Warning: {e}")
    gc.collect()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 2: Find and delete all GPU tensors
    print("  Step 2: Finding all GPU tensors in gc...")
    deleted = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                # Try to delete the tensor's storage
                obj.storage().resize_(0)
                deleted += 1
        except:
            pass
    print(f"    Resized storage of {deleted} tensors")
    gc.collect()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 3: Clear all module-level caches in vLLM
    print("  Step 3: Clearing vLLM module caches...")
    vllm_modules = [name for name in sys.modules if 'vllm' in name]
    cleared = 0
    for name in vllm_modules:
        module = sys.modules.get(name)
        if module is None:
            continue
        try:
            for attr in list(dir(module)):
                if attr.startswith('_') and not attr.startswith('__'):
                    try:
                        val = getattr(module, attr)
                        if isinstance(val, (dict, list)):
                            if hasattr(val, 'clear'):
                                val.clear()
                                cleared += 1
                    except:
                        pass
        except:
            pass
    print(f"    Cleared {cleared} caches")
    gc.collect()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 4: Reset torch._dynamo (compilation cache)
    print("  Step 4: Reset torch._dynamo...")
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception as e:
        print(f"    {e}")
    gc.collect()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 5: Clear PyTorch's CUDA caching allocator
    print("  Step 5: Clear PyTorch CUDA allocator...")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 6: Try to delete all nn.Module instances
    print("  Step 6: Delete all nn.Module instances...")
    deleted_modules = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                # Clear the module's parameters
                for param in obj.parameters():
                    param.data = torch.empty(0)
                for buf in obj.buffers():
                    buf.data = torch.empty(0)
                deleted_modules += 1
        except:
            pass
    print(f"    Cleared {deleted_modules} modules")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    Free: {get_memory():.2f} GB")

    # Step 7: Nuclear option - break all referrer chains
    print("  Step 7: Breaking referrer chains for GPU tensors...")
    iterations = 0
    while iterations < 5:
        found_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                    found_tensors.append(obj)
            except:
                pass

        if not found_tensors:
            break

        print(f"    Iteration {iterations + 1}: {len(found_tensors)} tensors remaining")

        for tensor in found_tensors:
            # Get all objects referring to this tensor
            referrers = gc.get_referrers(tensor)
            for ref in referrers:
                try:
                    if isinstance(ref, dict):
                        # Find and remove keys pointing to tensor
                        keys_to_del = [k for k, v in ref.items() if v is tensor]
                        for k in keys_to_del:
                            del ref[k]
                    elif isinstance(ref, list):
                        # Remove tensor from list
                        while tensor in ref:
                            ref.remove(tensor)
                except:
                    pass

            # Try to zero out the tensor
            try:
                tensor.storage().resize_(0)
            except:
                pass

        gc.collect()
        iterations += 1

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"    Free: {get_memory():.2f} GB")

    print("  Aggressive cleanup complete!")
    return get_memory()


def test_aggressive_cleanup():
    """Test if aggressive cleanup can recover GPU memory."""
    print("="*70)
    print("AGGRESSIVE CLEANUP TEST")
    print("="*70)

    initial_free = get_memory()
    print(f"\nInitial free memory: {initial_free:.2f} GB")

    # Load model
    print("\n>>> Loading model...")
    from vllm import LLM, SamplingParams

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

    after_load = get_memory()
    print(f"\nAfter load free: {after_load:.2f} GB (used {initial_free - after_load:.2f} GB)")

    # Delete LLM
    print("\n>>> Deleting LLM...")
    del llm
    del outputs
    gc.collect()

    after_del = get_memory()
    print(f"After del free: {after_del:.2f} GB")

    # Aggressive cleanup
    final_free = aggressive_cleanup()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Initial free:     {initial_free:.2f} GB")
    print(f"After load:       {after_load:.2f} GB")
    print(f"After del:        {after_del:.2f} GB")
    print(f"After cleanup:    {final_free:.2f} GB")
    print(f"Memory recovered: {final_free - after_load:.2f} GB")
    print(f"Memory leaked:    {initial_free - final_free:.2f} GB")

    if initial_free - final_free < 2.0:
        print("\n✓ SUCCESS! Memory properly cleaned up!")
        print("  In-process model switching should work!")
        return True
    else:
        print(f"\n⚠️  Still leaking {initial_free - final_free:.2f} GB")
        return False


if __name__ == '__main__':
    test_aggressive_cleanup()
