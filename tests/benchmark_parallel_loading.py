#!/usr/bin/env python3
"""Benchmark chunk-parallel loading speed for BlitzInfer.

Compares different chunk sizes and worker counts for loading models
from SSD into pinned memory arena.

Drops page cache before each test for cold-start accuracy.

Usage:
    # Full benchmark (requires sudo for cache drop)
    sudo python tests/benchmark_parallel_loading.py

    # Without cache dropping (warm cache)
    python tests/benchmark_parallel_loading.py --no-drop-cache

    # Specific model
    python tests/benchmark_parallel_loading.py --model /path/to/model

    # Quick (fewer configurations)
    python tests/benchmark_parallel_loading.py --quick
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blitzinfer.memory.arena import PinnedMemoryArena
from blitzinfer.memory.fast_loader import (
    DEFAULT_READ_CHUNK_BYTES,
    get_model_size,
    get_safetensor_files,
    load_model_to_arena,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)


# Well-known model paths on the RTX PRO 6000 system
DEFAULT_MODELS = {
    "qwen3-32b": os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots"),
    "gpt-oss-120b": os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots"),
}

# Test configurations
CHUNK_SIZES_GB = [0.5, 1.0, 2.0, 4.0]
WORKER_COUNTS = [4, 8, 16]

QUICK_CHUNK_SIZES_GB = [2.0]
QUICK_WORKER_COUNTS = [4, 16]


def find_snapshot_dir(base_path: str) -> str:
    """Find the actual snapshot directory under HuggingFace cache."""
    p = Path(base_path)
    if not p.exists():
        return base_path

    # Look for subdirectories (HF snapshot hashes)
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if subdirs:
        # Use the most recently modified
        subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return str(subdirs[0])

    return base_path


def drop_page_cache():
    """Drop OS page cache for cold-start measurements."""
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        time.sleep(0.5)  # Let the kernel flush
        return True
    except (PermissionError, OSError):
        return False


def run_benchmark(
    model_path: str,
    model_name: str,
    chunk_sizes_gb: list,
    worker_counts: list,
    drop_cache: bool,
    arena_size_gb: float = 88.0,
):
    """Run loading benchmarks for one model with various configurations."""
    model_size = get_model_size(model_path)
    sf_files = get_safetensor_files(model_path)
    model_size_gb = model_size / 1024**3

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Size: {model_size_gb:.2f}GB")
    print(f"  Files: {len(sf_files)}")
    print(f"  Drop cache: {drop_cache}")
    print(f"{'='*70}")

    # Allocate arena once (reused across tests)
    print(f"\nAllocating {arena_size_gb}GB pinned arena...")
    arena = PinnedMemoryArena(arena_size_gb, pin_memory=True, chunk_size_gb=8.0)
    print(f"Arena ready: {arena.size_gb:.1f}GB")

    results = []

    for chunk_gb in chunk_sizes_gb:
        for workers in worker_counts:
            chunk_bytes = int(chunk_gb * 1024**3)
            num_chunks = max(1, int(model_size / chunk_bytes) + 1)

            # Clear arena for this test
            arena.clear()

            # Drop page cache for cold-start measurement
            if drop_cache:
                if not drop_page_cache():
                    print("  WARNING: Could not drop page cache (run as root)")
                    drop_cache = False

            gc.collect()

            config = f"chunk={chunk_gb:.1f}GB, workers={workers}"
            print(f"\n  [{config}] chunks={num_chunks} ...", end=" ", flush=True)

            t0 = time.time()
            try:
                tensors = load_model_to_arena(
                    model_path, arena,
                    model_name=model_name,
                    parallel_workers=workers,
                    read_chunk_bytes=chunk_bytes,
                )
                elapsed = time.time() - t0
                speed = model_size_gb / elapsed
                print(f"{elapsed:.2f}s ({speed:.1f} GB/s) - {len(tensors)} tensors")

                results.append({
                    "chunk_gb": chunk_gb,
                    "workers": workers,
                    "elapsed": elapsed,
                    "speed_gbs": speed,
                    "num_chunks": num_chunks,
                    "num_tensors": len(tensors),
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.2f}s): {e}")
                results.append({
                    "chunk_gb": chunk_gb,
                    "workers": workers,
                    "elapsed": elapsed,
                    "speed_gbs": 0,
                    "error": str(e),
                })

    # Print summary table
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name} ({model_size_gb:.2f}GB)")
    print(f"{'='*70}")
    print(f"  {'Chunk':>8s}  {'Workers':>8s}  {'Time':>8s}  {'Speed':>10s}  {'Chunks':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    best = None
    for r in results:
        if "error" in r:
            print(f"  {r['chunk_gb']:>7.1f}GB  {r['workers']:>8d}  FAILED")
            continue
        speed_str = f"{r['speed_gbs']:.1f} GB/s"
        print(f"  {r['chunk_gb']:>7.1f}GB  {r['workers']:>8d}  {r['elapsed']:>7.2f}s  {speed_str:>10s}  {r['num_chunks']:>8d}")
        if best is None or r['speed_gbs'] > best['speed_gbs']:
            best = r

    if best:
        print(f"\n  BEST: chunk={best['chunk_gb']:.1f}GB, workers={best['workers']}, "
              f"speed={best['speed_gbs']:.1f} GB/s")

    # Cleanup
    del arena
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark chunk-parallel loading")
    parser.add_argument("--model", help="Path to specific model directory")
    parser.add_argument("--no-drop-cache", action="store_true",
                        help="Don't drop page cache (warm cache benchmark)")
    parser.add_argument("--quick", action="store_true",
                        help="Run fewer configurations")
    parser.add_argument("--arena-size", type=float, default=88.0,
                        help="Arena size in GB")
    args = parser.parse_args()

    drop_cache = not args.no_drop_cache
    chunk_sizes = QUICK_CHUNK_SIZES_GB if args.quick else CHUNK_SIZES_GB
    worker_counts = QUICK_WORKER_COUNTS if args.quick else WORKER_COUNTS

    if args.model:
        model_path = args.model
        model_name = Path(model_path).name
        run_benchmark(model_path, model_name, chunk_sizes, worker_counts,
                      drop_cache, args.arena_size)
    else:
        # Try default models
        tested = 0
        for name, base_path in DEFAULT_MODELS.items():
            model_path = find_snapshot_dir(base_path)
            sf_files = get_safetensor_files(model_path)
            if sf_files:
                run_benchmark(model_path, name, chunk_sizes, worker_counts,
                              drop_cache, args.arena_size)
                tested += 1
            else:
                print(f"\nSKIP: {name} not found at {base_path}")

        if tested == 0:
            print("\nNo models found. Use --model /path/to/model")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
