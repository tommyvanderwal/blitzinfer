#!/usr/bin/env python3
"""
ACCURATE REPRODUCTION: vLLM Model Switching Memory Leak on ROCm 780M

This test reproduces the exact scenario where model switching fails:
1. Load vLLM model (uses ~15GB)
2. Run inference
3. Delete model and cleanup
4. Try to load second model → FAILS due to unreleased memory

This is the blocker for fast model switching in BlitzInfer.

Environment:
  - AMD Radeon 780M (gfx1103)
  - ROCm 7.2, PyTorch 2.6+
  - 96GB unified memory
"""

import os
import sys
import gc
import time
import types

# Environment setup - must be before torch import
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # In-process mode
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module (vLLM import workaround)
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_memory_gb():
    """Get free GPU memory in GB."""
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3), total / (1024**3)


def count_gpu_tensors():
    """Count GPU tensors in gc."""
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


def full_cleanup():
    """Standard vLLM cleanup procedure."""
    gc.collect()

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception as e:
        pass

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def test_model_switch():
    """
    Reproduce the model switching failure scenario.
    """
    print("=" * 70)
    print("vLLM MODEL SWITCHING MEMORY LEAK REPRODUCTION")
    print("=" * 70)

    # System info
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    initial_free, total = get_memory_gb()
    print(f"\nInitial state: {initial_free:.2f} GB free / {total:.2f} GB total")

    # Import vLLM
    print("\n>>> Importing vLLM...")
    from vllm import LLM, SamplingParams

    after_import_free, _ = get_memory_gb()
    print(f"After import: {after_import_free:.2f} GB free")

    # Model config - conservative settings
    model_config = dict(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.50,  # 50% to leave room for second model
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=4 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    # =========================================================
    # FIRST MODEL
    # =========================================================
    print("\n" + "=" * 70)
    print("LOADING FIRST MODEL")
    print("=" * 70)

    start = time.perf_counter()
    llm1 = LLM(**model_config)
    load1_time = time.perf_counter() - start

    after_load1_free, _ = get_memory_gb()
    memory_used = initial_free - after_load1_free
    print(f"\nFirst model loaded in {load1_time:.2f}s")
    print(f"Memory: {after_load1_free:.2f} GB free (used {memory_used:.2f} GB)")

    # Run inference
    print("\n>>> Running inference...")
    outputs = llm1.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {outputs[0].outputs[0].text}")

    count1, size1 = count_gpu_tensors()
    print(f"\nGPU tensors: {count1} tensors, {size1:.2f} GB")

    # =========================================================
    # CLEANUP FIRST MODEL
    # =========================================================
    print("\n" + "=" * 70)
    print("CLEANING UP FIRST MODEL")
    print("=" * 70)

    print(">>> Deleting LLM instance...")
    del llm1
    del outputs

    print(">>> Running full cleanup...")
    full_cleanup()

    # Wait for driver to process
    time.sleep(2.0)
    full_cleanup()

    after_cleanup_free, _ = get_memory_gb()
    count2, size2 = count_gpu_tensors()

    memory_recovered = after_cleanup_free - after_load1_free
    memory_leaked = initial_free - after_cleanup_free

    print(f"\nAfter cleanup: {after_cleanup_free:.2f} GB free")
    print(f"Memory recovered: {memory_recovered:.2f} GB")
    print(f"Memory leaked: {memory_leaked:.2f} GB")
    print(f"GPU tensors still in gc: {count2} tensors, {size2:.2f} GB")

    # Check if we have enough memory for second model
    if memory_leaked > 10.0:
        print(f"\n*** BUG CONFIRMED: {memory_leaked:.2f} GB not released ***")

    # =========================================================
    # SECOND MODEL - This should work but fails
    # =========================================================
    print("\n" + "=" * 70)
    print("LOADING SECOND MODEL")
    print("=" * 70)

    print(f"Available memory: {after_cleanup_free:.2f} GB")
    print(f"Required memory: ~{memory_used:.2f} GB")

    if after_cleanup_free < memory_used:
        print(f"\n*** INSUFFICIENT MEMORY: {after_cleanup_free:.2f} < {memory_used:.2f} GB ***")
        print("Cannot load second model due to memory leak.")
        return False

    print("\n>>> Loading second model...")
    start = time.perf_counter()

    try:
        llm2 = LLM(**model_config)
        load2_time = time.perf_counter() - start

        after_load2_free, _ = get_memory_gb()
        print(f"\nSecond model loaded in {load2_time:.2f}s")
        print(f"Memory: {after_load2_free:.2f} GB free")

        # Run inference
        outputs = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {outputs[0].outputs[0].text}")

        print("\n" + "=" * 70)
        print("SUCCESS: Both models loaded and ran!")
        print("=" * 70)
        print(f"Switch time would be: {load2_time:.2f}s")

        del llm2
        del outputs
        gc.collect()

        return True

    except Exception as e:
        load2_time = time.perf_counter() - start
        print(f"\n*** FAILED after {load2_time:.2f}s ***")
        print(f"Error: {e}")

        print("\n" + "=" * 70)
        print("BUG REPRODUCED: Model switching failed")
        print("=" * 70)
        print(f"Cause: {memory_leaked:.2f} GB memory not released after first model deletion")
        print("This blocks in-process model switching on ROCm 780M.")

        import traceback
        traceback.print_exc()

        return False


def summarize():
    """Print summary and diagnosis."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print("""
The bug occurs because:

1. vLLM creates complex object graphs with circular references
2. When del llm is called, Python's gc cannot break all cycles
3. GPU tensors remain referenced by these circular structures
4. torch.cuda.empty_cache() only frees unreferenced tensors
5. Memory stays allocated in HIP/ROCR/kernel driver

On discrete GPUs or with subprocess isolation, process termination
forces the kernel to reclaim all memory. On 780M with unified memory
and in-process mode, this cleanup path is broken.

Workaround: Use vLLM multiprocessing mode (subprocess per model)
  - Cost: ~30-40 seconds per model switch
  - Target: <3 seconds for 7B models

Fix required in: ROCR-Runtime or HIP Runtime memory management
    """)


if __name__ == '__main__':
    print("Testing vLLM model switching on ROCm 780M (gfx1103)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    success = test_model_switch()
    summarize()

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {'PASS' if success else 'FAIL'}")
    print("=" * 70)
