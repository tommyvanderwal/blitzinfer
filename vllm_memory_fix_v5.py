#!/usr/bin/env python3
"""
Memory cleanup fix for vLLM on ROCm 780M - Version 5.

Targeted cleanup: only clear model weight dicts, preserve caches.
"""

import gc
import sys
import torch


def get_memory_gb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3)


def targeted_weight_cleanup():
    """
    Clear only dicts containing model weights, preserve caches.
    """
    # Find all GPU tensors and their referrer dicts
    weight_dicts = []
    tensor_dicts = []

    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict):
                # Check if this is a weight dict (from nn.Module._parameters)
                if 'weight' in obj:
                    val = obj.get('weight')
                    if torch.is_tensor(val) and val.is_cuda:
                        weight_dicts.append(obj)
                # Check if this is a tensor storage dict (from safetensors)
                if 'tensor' in obj:
                    val = obj.get('tensor')
                    if torch.is_tensor(val) and val.is_cuda:
                        tensor_dicts.append(obj)
        except:
            pass

    print(f"  Found {len(weight_dicts)} weight dicts, {len(tensor_dicts)} tensor dicts")

    # Clear weight dicts (these are nn.Module._parameters)
    for d in weight_dicts:
        try:
            for key in list(d.keys()):
                val = d.get(key)
                if torch.is_tensor(val) and val.is_cuda:
                    d[key] = None
        except:
            pass

    # Clear tensor dicts (these are safetensors loading dicts)
    for d in tensor_dicts:
        try:
            for key in list(d.keys()):
                val = d.get(key)
                if torch.is_tensor(val) and val.is_cuda:
                    d[key] = None
        except:
            pass

    gc.collect()

    return len(weight_dicts), len(tensor_dicts)


def cleanup_without_corrupting():
    """
    Cleanup that tries to preserve shared caches (rotary embedding etc).
    """
    # Step 1: Standard vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass
    gc.collect()

    # Step 2: Targeted weight cleanup
    w, t = targeted_weight_cleanup()

    # Step 3: Now force release remaining large tensors (>1MB)
    # Skip small tensors that might be caches
    released = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                size_mb = obj.numel() * obj.element_size() / (1024**2)
                if size_mb > 1:  # Release tensors > 1MB (model weights + activations)
                    obj.storage().resize_(0)
                    released += 1
        except:
            pass

    gc.collect()

    # Step 4: Reset torch dynamo
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except:
        pass

    # Step 5: Clear CUDA caches
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return w, t, released


def test_targeted_cleanup():
    """Test targeted cleanup approach."""
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
    print("TESTING TARGETED CLEANUP (preserve caches)")
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
    used = initial - after_load
    print(f"After load: {after_load:.2f} GB free (used {used:.2f} GB)")

    # Generate
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Delete
    print("\n>>> Deleting model...")
    del llm
    del out
    gc.collect()

    # Targeted cleanup
    print(">>> Running targeted cleanup...")
    start = time.perf_counter()
    w, t, r = cleanup_without_corrupting()
    cleanup_time = time.perf_counter() - start
    print(f"  Cleared {w} weight dicts, {t} tensor dicts, released {r} large tensors")
    print(f"  Cleanup time: {cleanup_time:.2f}s")

    time.sleep(0.5)

    after_cleanup = get_memory_gb()
    leaked = initial - after_cleanup
    print(f"\nAfter cleanup: {after_cleanup:.2f} GB free (leaked {leaked:.2f} GB)")

    # 4GB KV cache + some overhead is expected
    if leaked < 6.0 and after_cleanup > 50.0:
        print(f"\n>>> Acceptable leak ({leaked:.1f} GB). Loading second model...")

        start = time.perf_counter()
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
        load2_time = time.perf_counter() - start
        print(f"Second model loaded in {load2_time:.2f}s")

        out2 = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {out2[0].outputs[0].text}")

        total_switch = cleanup_time + load2_time
        print(f"\n" + "=" * 70)
        print(f"SUCCESS!")
        print(f"=" * 70)
        print(f"Cleanup: {cleanup_time:.2f}s")
        print(f"Load:    {load2_time:.2f}s")
        print(f"TOTAL:   {total_switch:.2f}s")

        del llm2
        del out2
        gc.collect()

        return True, total_switch
    else:
        print(f"\n>>> FAILED - still leaking {leaked:.2f} GB")
        return False, 0


if __name__ == '__main__':
    success, switch_time = test_targeted_cleanup()
    if success:
        print(f"\nModel switching works! Total time: {switch_time:.2f}s")
    else:
        print("\nModel switching still blocked.")
