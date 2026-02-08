#!/usr/bin/env python3
"""Test prefetch system with a real model."""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch

from blitzinfer.memory import PinnedMemoryArena, ModelPrefetcher, PrefetchStatus
from blitzinfer.memory import load_model_to_arena, get_model_size, get_safetensor_files

# Model to test
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def get_model_path(model_name: str) -> str:
    """Get the local path for a HuggingFace model."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name, local_files_only=True)


def test_arena_load():
    """Test loading a model directly into the arena."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct Arena Load (SSD → Pinned RAM)")
    print("=" * 60)

    model_path = get_model_path(MODEL_NAME)
    print(f"Model path: {model_path}")

    # Get model size
    model_size = get_model_size(model_path)
    print(f"Model size: {model_size / 1024**3:.2f} GB")

    # List safetensor files
    sf_files = get_safetensor_files(model_path)
    print(f"Safetensor files: {len(sf_files)}")
    for f in sf_files:
        print(f"  - {f.name}: {os.path.getsize(f) / 1024**3:.2f} GB")

    # Create arena (add some headroom)
    arena_size_gb = (model_size / 1024**3) + 2
    print(f"\nAllocating {arena_size_gb:.1f} GB arena...")

    t0 = time.time()
    arena = PinnedMemoryArena(arena_size_gb)
    arena_alloc_time = time.time() - t0
    print(f"Arena allocated in {arena_alloc_time:.2f}s")

    # Load model into arena
    print(f"\nLoading model into arena...")
    t0 = time.time()
    tensors = load_model_to_arena(model_path, arena, MODEL_NAME)
    load_time = time.time() - t0

    load_speed = (model_size / 1024**3) / load_time
    print(f"Loaded {len(tensors)} tensors in {load_time:.2f}s ({load_speed:.1f} GB/s)")

    # Print arena stats
    stats = arena.get_stats()
    print(f"\nArena stats:")
    print(f"  Total: {stats['total_gb']:.2f} GB")
    print(f"  Used: {stats['used_gb']:.2f} GB")
    print(f"  Available: {stats['available_gb']:.2f} GB")
    print(f"  Models: {stats['num_models']}")

    return arena, tensors


def test_gpu_transfer(arena, model_name: str):
    """Test transferring tensors from arena to GPU."""
    print("\n" + "=" * 60)
    print("TEST 2: GPU Transfer (Pinned RAM → GPU)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU transfer test")
        return

    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_mem / 1024**3:.1f} GB")

    # Get tensors from arena
    print(f"\nGetting tensor views from arena...")
    t0 = time.time()
    tensors = arena.get_all_tensors(model_name)
    get_time = time.time() - t0
    print(f"Got {len(tensors)} tensor views in {get_time:.3f}s")

    # Calculate total size
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"Total tensor data: {total_bytes / 1024**3:.2f} GB")

    # Transfer to GPU (non-blocking)
    print(f"\nTransferring to GPU...")
    torch.cuda.synchronize()

    t0 = time.time()
    gpu_tensors = {}
    for name, tensor in tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    transfer_time = time.time() - t0

    transfer_speed = (total_bytes / 1024**3) / transfer_time
    print(f"Transferred {total_bytes / 1024**3:.2f} GB in {transfer_time:.2f}s ({transfer_speed:.1f} GB/s)")

    # Clear GPU memory
    del gpu_tensors
    torch.cuda.empty_cache()

    return transfer_time, transfer_speed


def test_prefetcher():
    """Test the full prefetcher workflow."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Prefetcher Workflow")
    print("=" * 60)

    model_path = get_model_path(MODEL_NAME)
    model_size = get_model_size(model_path)

    # Create arena with headroom
    arena_size_gb = (model_size / 1024**3) + 2
    print(f"Creating {arena_size_gb:.1f} GB arena...")
    arena = PinnedMemoryArena(arena_size_gb)

    # Create prefetcher
    prefetcher = ModelPrefetcher(arena)
    prefetcher.register_model(MODEL_NAME, model_path)

    print(f"Initial status: {prefetcher.get_status(MODEL_NAME).name}")

    # Start prefetch
    print(f"\nStarting prefetch...")
    t0 = time.time()
    success = prefetcher.start_prefetch(MODEL_NAME)
    print(f"Prefetch started: {success}")
    print(f"Status: {prefetcher.get_status(MODEL_NAME).name}")

    # Wait for completion
    print(f"Waiting for prefetch to complete...")
    prefetcher.wait_for_ready(MODEL_NAME, timeout=120)
    prefetch_time = time.time() - t0

    print(f"Prefetch completed in {prefetch_time:.2f}s")
    print(f"Final status: {prefetcher.get_status(MODEL_NAME).name}")

    prefetch_speed = (model_size / 1024**3) / prefetch_time
    print(f"Prefetch speed: {prefetch_speed:.1f} GB/s")

    # Test getting tensors for GPU
    if prefetcher.is_ready(MODEL_NAME):
        print(f"\nGetting tensors for GPU transfer...")
        t0 = time.time()
        tensors = prefetcher.get_tensors_for_gpu(MODEL_NAME)
        print(f"Got {len(tensors)} tensors in {time.time() - t0:.3f}s")
        print(f"Status after get: {prefetcher.get_status(MODEL_NAME).name}")

        # Mark transfer complete
        prefetcher.mark_transfer_complete(MODEL_NAME)
        print(f"Status after mark_transfer_complete: {prefetcher.get_status(MODEL_NAME).name}")

    # Print stats
    stats = prefetcher.get_stats()
    print(f"\nPrefetcher stats:")
    print(f"  Ready count: {stats['ready_count']}")
    print(f"  Loading count: {stats['loading_count']}")
    print(f"  Arena used: {stats['arena']['used_gb']:.2f} GB")

    # Cleanup
    prefetcher.shutdown()

    return prefetch_time, prefetch_speed


def test_cold_vs_prefetch_comparison():
    """Compare cold load vs prefetched load times."""
    print("\n" + "=" * 60)
    print("TEST 4: Cold Load vs Prefetched Load Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping comparison test")
        return

    model_path = get_model_path(MODEL_NAME)
    model_size = get_model_size(model_path)

    # Test 1: Measure cold load (standard vLLM would do this)
    print(f"\n--- Simulating Cold Load (SSD → GPU directly) ---")
    print("(This simulates what happens without prefetch)")

    # We can't easily test vLLM's cold load without actually running it,
    # so we'll simulate by measuring sequential file read + GPU transfer
    from blitzinfer.memory.fast_loader import parse_safetensor_header, get_tensor_info

    sf_files = get_safetensor_files(model_path)
    total_bytes = 0
    tensors_loaded = {}

    torch.cuda.synchronize()
    t0_cold = time.time()

    for sf_file in sf_files:
        # Read file (simulates disk read in cold load)
        with open(sf_file, 'rb') as f:
            data = f.read()

        # Parse header
        header_size, header = parse_safetensor_header(str(sf_file))
        tensor_info = get_tensor_info(header)

        # Create tensors and move to GPU (simulates vLLM weight loading)
        data_offset = 8 + header_size
        for name, info in tensor_info.items():
            start, end = info['data_offsets']
            tensor_bytes = data[data_offset + start:data_offset + end]

            # Create tensor from bytes
            tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=info['dtype']).view(info['shape'])
            # Move to GPU
            tensors_loaded[name] = tensor.to('cuda', non_blocking=True)
            total_bytes += len(tensor_bytes)

    torch.cuda.synchronize()
    cold_load_time = time.time() - t0_cold
    cold_speed = (total_bytes / 1024**3) / cold_load_time

    print(f"Cold load: {total_bytes / 1024**3:.2f} GB in {cold_load_time:.2f}s ({cold_speed:.1f} GB/s)")

    # Clear GPU
    del tensors_loaded
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Test 2: Prefetched load
    print(f"\n--- Prefetched Load (Arena → GPU) ---")

    # Pre-load into arena
    arena_size_gb = (model_size / 1024**3) + 2
    arena = PinnedMemoryArena(arena_size_gb)

    print("Pre-loading into arena (this happens in background while serving)...")
    t0 = time.time()
    load_model_to_arena(model_path, arena, MODEL_NAME)
    prefetch_time = time.time() - t0
    print(f"Prefetch completed in {prefetch_time:.2f}s")

    # Now measure just the GPU transfer (this is what matters for switch time)
    print("\nMeasuring GPU transfer (what user waits for during switch)...")
    tensors = arena.get_all_tensors(MODEL_NAME)

    torch.cuda.synchronize()
    t0_hot = time.time()

    gpu_tensors = {}
    for name, tensor in tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
    torch.cuda.synchronize()

    hot_load_time = time.time() - t0_hot
    hot_speed = (total_bytes / 1024**3) / hot_load_time

    print(f"Hot load (prefetched): {total_bytes / 1024**3:.2f} GB in {hot_load_time:.2f}s ({hot_speed:.1f} GB/s)")

    # Summary
    print(f"\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Cold load time (user waits):     {cold_load_time:.2f}s @ {cold_speed:.1f} GB/s")
    print(f"Hot load time (with prefetch):   {hot_load_time:.2f}s @ {hot_speed:.1f} GB/s")
    print(f"Speedup:                         {cold_load_time / hot_load_time:.1f}x faster")
    print(f"Time saved per switch:           {cold_load_time - hot_load_time:.2f}s")

    # Cleanup
    del gpu_tensors
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("BLITZINFER PREFETCH SYSTEM TEST")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run tests
    try:
        # Test 1: Direct arena load
        arena, tensors = test_arena_load()

        # Test 2: GPU transfer
        if torch.cuda.is_available():
            test_gpu_transfer(arena, MODEL_NAME)

        # Clean up arena from test 1
        del arena
        import gc
        gc.collect()

        # Test 3: Full prefetcher workflow
        test_prefetcher()

        # Test 4: Comparison
        if torch.cuda.is_available():
            test_cold_vs_prefetch_comparison()

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
