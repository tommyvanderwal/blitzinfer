#!/usr/bin/env python3
"""Model weight caching for faster loading."""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "blitzinfer"


def get_cache_key(model_name: str) -> str:
    """Generate a cache key for a model."""
    # Use model name hash as key
    return hashlib.md5(model_name.encode()).hexdigest()[:16]


def get_cache_path(model_name: str) -> Path:
    """Get the cache path for a model."""
    key = get_cache_key(model_name)
    return CACHE_DIR / f"weights_{key}.pt"


def get_metadata_path(model_name: str) -> Path:
    """Get the metadata path for a model."""
    key = get_cache_key(model_name)
    return CACHE_DIR / f"meta_{key}.json"


def save_model_weights(
    model_name: str,
    state_dict: dict,
    metadata: Optional[dict] = None,
) -> Path:
    """Save model weights to cache for faster loading.

    Args:
        model_name: The model name/path
        state_dict: Model state dict
        metadata: Optional metadata (config, etc.)

    Returns:
        Path to the cached weights file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = get_cache_path(model_name)
    meta_path = get_metadata_path(model_name)

    logger.info(f"Saving model weights to cache: {cache_path}")
    start = time.time()

    # Save weights using torch.save with pickle protocol 4 for speed
    torch.save(state_dict, cache_path, pickle_protocol=4)

    # Save metadata
    meta = metadata or {}
    meta['model_name'] = model_name
    meta['cached_at'] = time.time()
    meta['cache_path'] = str(cache_path)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start
    size_gb = cache_path.stat().st_size / (1024**3)
    logger.info(f"Saved {size_gb:.2f} GB in {elapsed:.2f}s ({size_gb/elapsed:.2f} GB/s)")

    return cache_path


def load_cached_weights(
    model_name: str,
    map_location: str = "cpu",
    mmap: bool = True,
) -> Optional[dict]:
    """Load model weights from cache if available.

    Args:
        model_name: The model name/path
        map_location: Where to load tensors ("cpu" or "cuda")
        mmap: Use memory mapping for faster loading

    Returns:
        State dict if cache exists, None otherwise
    """
    cache_path = get_cache_path(model_name)
    meta_path = get_metadata_path(model_name)

    if not cache_path.exists():
        logger.debug(f"No cache found for {model_name}")
        return None

    logger.info(f"Loading cached weights from: {cache_path}")
    start = time.time()

    # Load with mmap for faster access
    if mmap:
        state_dict = torch.load(
            cache_path,
            map_location=map_location,
            mmap=True,  # Use memory mapping
        )
    else:
        state_dict = torch.load(cache_path, map_location=map_location)

    elapsed = time.time() - start
    size_gb = cache_path.stat().st_size / (1024**3)
    logger.info(f"Loaded {size_gb:.2f} GB in {elapsed:.2f}s ({size_gb/elapsed:.2f} GB/s)")

    return state_dict


def is_cached(model_name: str) -> bool:
    """Check if a model is cached."""
    return get_cache_path(model_name).exists()


def get_cache_info(model_name: str) -> Optional[dict]:
    """Get cache metadata for a model."""
    meta_path = get_metadata_path(model_name)
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def clear_cache(model_name: Optional[str] = None):
    """Clear cache for a model or all models."""
    if model_name:
        cache_path = get_cache_path(model_name)
        meta_path = get_metadata_path(model_name)
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    else:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
