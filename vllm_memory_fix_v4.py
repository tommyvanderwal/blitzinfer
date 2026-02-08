#!/usr/bin/env python3
"""
Memory cleanup fix for vLLM on ROCm 780M - Version 4.

Combines forced tensor release with module reload to get clean state.
"""

import gc
import sys
import torch


def get_memory_gb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3)


def force_release_and_reload():
    """
    Force release GPU tensors and reload vLLM modules for clean state.
    """
    # Step 1: Standard vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass
    gc.collect()

    # Step 2: Force release ALL GPU tensors
    released = 0
    total_bytes = 0
    for _ in range(3):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                    bytes_before = obj.numel() * obj.element_size()
                    obj.storage().resize_(0)
                    total_bytes += bytes_before
                    released += 1
            except:
                pass
        gc.collect()

    # Step 3: Reset torch dynamo
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except:
        pass

    # Step 4: Unload ALL vLLM modules so next import is fresh
    vllm_modules = [name for name in list(sys.modules.keys())
                    if 'vllm' in name.lower()]
    for name in vllm_modules:
        try:
            del sys.modules[name]
        except:
            pass

    # Step 5: Clear CUDA caches
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return released, total_bytes / (1024**3)


def test_combined_approach():
    """Test forced release + module reload."""
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
    print("TESTING FORCED RELEASE + MODULE RELOAD")
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

    # Delete and cleanup
    print("\n>>> Deleting and force-releasing...")
    del llm
    del out
    gc.collect()

    start = time.perf_counter()
    released, gb = force_release_and_reload()
    cleanup_time = time.perf_counter() - start
    print(f"Released {released} tensors ({gb:.2f} GB) in {cleanup_time:.2f}s")

    time.sleep(0.5)

    after_cleanup = get_memory_gb()
    leaked = initial - after_cleanup
    print(f"After cleanup: {after_cleanup:.2f} GB free (leaked {leaked:.2f} GB)")

    if leaked < 2.0:
        print("\n>>> SUCCESS! Loading second model with fresh vLLM...")

        # Re-import vLLM (fresh modules)
        start = time.perf_counter()
        from vllm import LLM, SamplingParams

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

        after_load2 = get_memory_gb()
        print(f"Second model loaded in {load2_time:.2f}s")
        print(f"Memory: {after_load2:.2f} GB free")

        out2 = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {out2[0].outputs[0].text}")

        total_switch = cleanup_time + load2_time
        print(f"\n" + "=" * 70)
        print(f"SUCCESS! Model switch completed!")
        print(f"=" * 70)
        print(f"Cleanup time: {cleanup_time:.2f}s")
        print(f"Second load:  {load2_time:.2f}s")
        print(f"TOTAL SWITCH: {total_switch:.2f}s")
        print(f"=" * 70)

        del llm2
        del out2
        gc.collect()

        return True, total_switch
    else:
        print(f"\n>>> FAILED - still leaking {leaked:.2f} GB")
        return False, 0


if __name__ == '__main__':
    success, switch_time = test_combined_approach()
    if success:
        print(f"\nModel switching is working! Switch time: {switch_time:.2f}s")
    else:
        print("\nModel switching still blocked by memory leak.")
