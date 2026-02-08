#!/usr/bin/env python3
"""Deep dive into why model parameters aren't being freed.

Key question: Why does param.data = torch.empty(0, device='cpu') not free GPU memory?
"""

import os
import gc
import sys

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
    }


def print_gpu_memory(label=""):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Alloc: {m['allocated_gb']:.2f}GB, "
          f"Reserved: {m['reserved_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")


def get_engine_internals(llm):
    """Get deep references into vLLM internals."""
    engine = llm.llm_engine

    # V1 engine uses InprocClient which wraps EngineCore
    engine_core = engine.engine_core

    # InprocClient has engine_core attribute that is the actual EngineCore
    actual_core = getattr(engine_core, 'engine_core', engine_core)

    executor = getattr(actual_core, 'model_executor', None)
    if executor is None:
        print("No model_executor found in engine_core")
        print(f"  engine_core type: {type(actual_core)}")
        print(f"  engine_core attrs: {[a for a in dir(actual_core) if not a.startswith('_')]}")
        return None, None, None, None

    driver_worker = getattr(executor, 'driver_worker', None)
    worker = getattr(driver_worker, 'worker', None) if driver_worker else None
    model_runner = getattr(worker, 'model_runner', None) if worker else None
    model = getattr(model_runner, 'model', None) if model_runner else None

    return executor, worker, model_runner, model


def test_param_clearing():
    """Test if clearing a parameter's data actually frees GPU memory."""
    print("\n=== TEST: Does param.data = empty free GPU memory? ===\n")

    print_gpu_memory("before tensor")

    # Create a large tensor
    t = torch.randn(1000, 1000, 1000, device='cuda', dtype=torch.float32)  # 4GB
    print_gpu_memory("after 4GB tensor")

    # Method 1: Set data to empty CPU tensor
    print("\n--- Method 1: param.data = empty CPU tensor ---")
    t.data = torch.empty(0, device='cpu')
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after data=empty CPU")

    # Clean up
    del t
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del t")

    # Create another tensor
    print("\n--- Method 2: Direct deletion ---")
    t2 = torch.randn(1000, 1000, 1000, device='cuda', dtype=torch.float32)  # 4GB
    print_gpu_memory("after 4GB tensor t2")

    del t2
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del t2 + empty_cache")


def test_nn_parameter_clearing():
    """Test clearing nn.Parameter specifically."""
    print("\n=== TEST: Does clearing nn.Parameter work? ===\n")

    print_gpu_memory("before param")

    # Create an nn.Parameter (like model weights)
    p = torch.nn.Parameter(torch.randn(1000, 1000, 1000, device='cuda', dtype=torch.float32))
    print_gpu_memory("after 4GB nn.Parameter")

    # Check reference count
    print(f"Parameter refcount: {sys.getrefcount(p)}")
    print(f"Parameter.data refcount: {sys.getrefcount(p.data)}")

    # Method: Set data to empty
    p.data = torch.empty(0, device='cpu')
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after data=empty CPU")
    print(f"Parameter refcount after: {sys.getrefcount(p)}")

    # Delete parameter
    del p
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del p")


def investigate_model_weights(llm):
    """Look at model weights and how they're structured."""
    print("\n=== INVESTIGATING MODEL WEIGHTS ===\n")

    executor, worker, model_runner, model = get_engine_internals(llm)

    if model is None:
        print("Could not get model reference!")
        return

    print(f"Model type: {type(model)}")

    # Get first few parameters
    params = list(model.named_parameters())
    print(f"Total parameters: {len(params)}")

    # Look at structure of first parameter
    if params:
        name, param = params[0]
        print(f"\nFirst parameter: {name}")
        print(f"  Type: {type(param)}")
        print(f"  Shape: {param.shape}")
        print(f"  Device: {param.device}")
        print(f"  Dtype: {param.dtype}")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Is leaf: {param.is_leaf}")

        # Check for any views or shared storage
        print(f"  Data ptr: {param.data_ptr()}")
        print(f"  Storage size: {param.storage().size()}")
        print(f"  Storage data ptr: {param.storage().data_ptr()}")

        # Check refcounts
        print(f"  Param refcount: {sys.getrefcount(param)}")
        print(f"  Data refcount: {sys.getrefcount(param.data)}")

    # Sample a few more parameters to see if they share storage
    print("\n--- Checking for shared storage ---")
    storage_ptrs = set()
    shared_count = 0
    for name, param in params[:100]:
        ptr = param.storage().data_ptr()
        if ptr in storage_ptrs:
            shared_count += 1
        storage_ptrs.add(ptr)
    print(f"First 100 params: {shared_count} share storage with earlier params")


def test_real_model_cleanup():
    """Actually load a model and test cleanup."""
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("\n" + "=" * 70)
    print("REAL MODEL CLEANUP TEST")
    print("=" * 70)

    print_gpu_memory("baseline")

    # Load model
    print("\n=== Loading model ===")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print_gpu_memory("after load")

    # Investigate weights
    investigate_model_weights(llm)

    # Get internals
    executor, worker, model_runner, model = get_engine_internals(llm)

    print("\n=== CLEANUP ATTEMPTS ===\n")

    # Attempt 1: Try to set each parameter to None
    print("--- Attempt 1: Set param.data to None ---")
    if model:
        for name, param in model.named_parameters():
            if param.device.type == 'cuda':
                # Try different approaches
                try:
                    param.data = torch.tensor([], device='cpu')
                except:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after set data to []")

    # Attempt 2: Manually delete the tensor's storage
    print("\n--- Attempt 2: Clear storage ---")
    if model:
        for name, param in model.named_parameters():
            try:
                # This might actually free memory
                param.data.storage().resize_(0)
            except Exception as e:
                print(f"  Error on {name}: {e}")
                break
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after storage.resize_(0)")

    # Attempt 3: Use torch.cuda.memory._record_memory_history
    print("\n--- Recording memory history ---")
    try:
        torch.cuda.memory._record_memory_history(enabled=True)
        gc.collect()
        torch.cuda.empty_cache()
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=False)

        # Analyze snapshot
        if snapshot and 'segments' in snapshot:
            print(f"Memory segments: {len(snapshot['segments'])}")
            total_active = 0
            for seg in snapshot['segments']:
                if seg.get('allocated_size', 0) > 0:
                    total_active += seg['allocated_size']
            print(f"Total active allocations: {total_active / 1024**3:.2f}GB")
    except Exception as e:
        print(f"Memory history failed: {e}")

    # Attempt 4: Delete model directly
    print("\n--- Attempt 4: Delete model object ---")
    if model_runner:
        model_runner.model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del model")

    # Attempt 5: Clear entire model_runner
    print("\n--- Attempt 5: Clear model_runner ---")
    if worker:
        worker.model_runner = None
    del model_runner
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del model_runner")

    # Attempt 6: Clear worker
    print("\n--- Attempt 6: Clear worker ---")
    if executor and hasattr(executor, 'driver_worker') and executor.driver_worker:
        executor.driver_worker.worker = None
    del worker
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del worker")

    # Attempt 7: Shutdown engine core
    print("\n--- Attempt 7: engine_core.shutdown() ---")
    engine_core = llm.llm_engine.engine_core
    actual_core = getattr(engine_core, 'engine_core', engine_core)
    try:
        actual_core.shutdown()
    except Exception as e:
        print(f"  shutdown error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after shutdown")

    # Attempt 8: Delete LLM
    print("\n--- Attempt 8: del llm ---")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del llm")

    # Attempt 9: Clear NCCL process group
    print("\n--- Attempt 9: Destroy process group ---")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            print("  Process group is initialized, destroying...")
            dist.destroy_process_group()
        else:
            print("  Process group not initialized")
    except Exception as e:
        print(f"  Error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after destroy_process_group")

    # Attempt 10: Reset CUDA device
    print("\n--- Attempt 10: torch.cuda.reset_peak_memory_stats ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after reset stats")

    # Final check
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print_gpu_memory("final")


def main():
    # First run simple tests
    test_param_clearing()
    test_nn_parameter_clearing()

    # Then test with real model
    test_real_model_cleanup()


if __name__ == "__main__":
    main()
