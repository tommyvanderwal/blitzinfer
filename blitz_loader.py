#!/usr/bin/env python3
"""
BlitzLoader - Fast model switching via RAM-cached weights

Strategy:
1. Preload all model weights into system RAM using mmap
2. When switching models, weights are already in page cache
3. Transfer from RAM to GPU is fast (~10 GB/s+)
"""

import os
import time
import glob
import mmap
from typing import Dict, List
import torch

class WeightCache:
    """Cache model weights in RAM for fast GPU loading"""

    def __init__(self):
        self.cached_models: Dict[str, List[bytes]] = {}
        self.model_paths: Dict[str, str] = {}

    def get_model_path(self, model_name: str) -> str:
        """Get local path for HuggingFace model"""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        model_dir = f"models--{model_name.replace('/', '--')}"
        model_path = os.path.join(cache_dir, model_dir, "snapshots")
        if os.path.exists(model_path):
            snapshots = os.listdir(model_path)
            if snapshots:
                return os.path.join(model_path, snapshots[0])
        return None

    def preload_model(self, model_name: str) -> float:
        """Preload model weights into RAM page cache"""
        print(f"Preloading {model_name} into RAM...")

        model_path = self.get_model_path(model_name)
        if not model_path:
            raise ValueError(f"Model not found: {model_name}")

        self.model_paths[model_name] = model_path
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

        total_size = sum(os.path.getsize(f) for f in safetensor_files)
        print(f"  Total size: {total_size/1e9:.2f} GB ({len(safetensor_files)} files)")

        start = time.time()

        # Read files to force them into page cache
        # Using mmap for efficient memory-mapped access
        for f in safetensor_files:
            with open(f, "rb") as fp:
                # Memory-map the file
                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                # Touch all pages to ensure they're in RAM
                _ = mm[:]
                mm.close()

        elapsed = time.time() - start
        speed = total_size / elapsed / 1e9
        print(f"  Preloaded in {elapsed:.1f}s ({speed:.2f} GB/s)")

        return elapsed

    def is_preloaded(self, model_name: str) -> bool:
        """Check if model is preloaded"""
        return model_name in self.model_paths


def test_preload_speeds():
    """Test preloading and measure speeds"""
    cache = WeightCache()

    models = [
        "openai/gpt-oss-120b",
        "Qwen/Qwen3-VL-32B-Instruct",
    ]

    print("="*60)
    print("Testing weight preloading")
    print("="*60)

    for model in models:
        try:
            # First load (cold)
            print(f"\n{model} - Cold load:")
            t1 = cache.preload_model(model)

            # Second load (warm)
            print(f"\n{model} - Warm load (should be faster):")
            t2 = cache.preload_model(model)

            print(f"\n  Speedup: {t1/t2:.2f}x")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_preload_speeds()
