"""Page cache warmer for fast model loading.

This module provides a simple approach to speed up model switching:
pre-read safetensor files into the OS page cache so that when vLLM
loads the model, it reads from RAM instead of SSD.

Performance:
- Cold weight loading: ~20s (from NVMe SSD)
- Warm weight loading: ~8s (from page cache)
- Cache warming speed: ~12 GB/s (when not competing with other I/O)
- Net speedup: ~2.5x for weight loading phase
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class WarmStatus(Enum):
    """Status of a model in the cache warmer."""
    COLD = auto()      # Not in page cache
    WARMING = auto()   # Currently being read into page cache
    WARM = auto()      # In page cache, ready for fast loading
    ERROR = auto()     # Failed to warm


def get_safetensor_files(model_path: str) -> List[Path]:
    """Get all safetensor files for a model."""
    path = Path(model_path)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        files = sorted(path.glob("**/*.safetensors"))
    return files


def get_model_size(model_path: str) -> int:
    """Get total size of safetensor files in bytes."""
    files = get_safetensor_files(model_path)
    if not files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")
    return sum(f.stat().st_size for f in files)


class PageCacheWarmer:
    """Background page cache warming for fast model switching.

    This class pre-reads model safetensor files into the OS page cache
    in the background. When vLLM loads the model, the weights are already
    in RAM (page cache) instead of on SSD.

    Usage:
        warmer = PageCacheWarmer()
        warmer.register_model('qwen-7b', '/path/to/qwen')

        # Start warming while current model is serving
        warmer.start_warming('qwen-7b')

        # Check status
        if warmer.is_warm('qwen-7b'):
            # Model weights are in page cache, vLLM load will be fast
            pass

    Performance notes:
        - Warming speed: ~12 GB/s on NVMe SSD (sequential read)
        - This happens in background while current model serves requests
        - vLLM weight loading: ~8s warm vs ~20s cold (for 65GB model)
    """

    def __init__(
        self,
        max_workers: int = 1,
        chunk_size: int = 64 * 1024 * 1024,  # 64MB chunks for optimal throughput
    ):
        """Initialize the cache warmer.

        Args:
            max_workers: Maximum concurrent warming operations.
            chunk_size: Read chunk size in bytes (64MB default for NVMe).
        """
        self._model_paths: Dict[str, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._chunk_size = chunk_size

        self._status: Dict[str, WarmStatus] = {}
        self._futures: Dict[str, Future] = {}
        self._errors: Dict[str, str] = {}
        self._progress: Dict[str, float] = {}  # 0.0 to 1.0
        self._lock = threading.RLock()

        # Callbacks
        self._on_warm_callbacks: Dict[str, List[Callable]] = {}

    def register_model(self, model_name: str, model_path: str):
        """Register a model path for warming.

        Args:
            model_name: Unique name for the model.
            model_path: Path to the model directory.
        """
        with self._lock:
            self._model_paths[model_name] = model_path
            if model_name not in self._status:
                self._status[model_name] = WarmStatus.COLD

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the path for a model."""
        if model_name in self._model_paths:
            return self._model_paths[model_name]

        # Try HuggingFace cache
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_name, local_files_only=True)
        except Exception:
            pass

        # Try as direct path
        if Path(model_name).exists():
            return model_name

        return None

    def start_warming(self, model_name: str) -> bool:
        """Start background cache warming for a model.

        Non-blocking. Files will be read in a background thread.

        Args:
            model_name: Name of the model to warm.

        Returns:
            True if warming started or already warm.
            False if unable to start.
        """
        with self._lock:
            current_status = self._status.get(model_name, WarmStatus.COLD)

            if current_status in (WarmStatus.WARM, WarmStatus.WARMING):
                logger.debug(f"Model {model_name} already {current_status.name}")
                return True

            model_path = self.get_model_path(model_name)
            if model_path is None:
                logger.warning(f"Cannot warm {model_name}: path not found")
                return False

            # Get model size for logging
            try:
                model_size = get_model_size(model_path)
            except FileNotFoundError as e:
                logger.warning(f"Cannot warm {model_name}: {e}")
                return False

            self._status[model_name] = WarmStatus.WARMING
            self._progress[model_name] = 0.0

            future = self._executor.submit(self._warming_worker, model_name, model_path)
            self._futures[model_name] = future

            logger.info(f"Started warming {model_name} ({model_size / 1024**3:.1f} GB)")
            return True

    def _warming_worker(self, model_name: str, model_path: str):
        """Worker thread: reads files to warm page cache."""
        try:
            files = get_safetensor_files(model_path)
            if not files:
                raise FileNotFoundError(f"No safetensor files in {model_path}")

            total_size = sum(f.stat().st_size for f in files)
            bytes_read = 0
            start_time = time.time()

            for sf_file in files:
                file_size = sf_file.stat().st_size

                # Read file in chunks to warm page cache
                with open(sf_file, 'rb') as f:
                    while True:
                        chunk = f.read(self._chunk_size)
                        if not chunk:
                            break
                        bytes_read += len(chunk)

                # Update progress
                with self._lock:
                    self._progress[model_name] = bytes_read / total_size

            elapsed = time.time() - start_time
            speed = (total_size / 1024**3) / elapsed

            with self._lock:
                self._status[model_name] = WarmStatus.WARM
                self._progress[model_name] = 1.0

            logger.info(
                f"Warming complete: {model_name} - "
                f"{total_size / 1024**3:.1f} GB in {elapsed:.2f}s ({speed:.1f} GB/s)"
            )

            self._fire_warm_callbacks(model_name)

        except Exception as e:
            logger.error(f"Warming failed for {model_name}: {e}")
            with self._lock:
                self._status[model_name] = WarmStatus.ERROR
                self._errors[model_name] = str(e)

    def _fire_warm_callbacks(self, model_name: str):
        """Fire callbacks when model becomes warm."""
        callbacks = self._on_warm_callbacks.get(model_name, [])
        for callback in callbacks:
            try:
                callback(model_name)
            except Exception as e:
                logger.error(f"Callback error for {model_name}: {e}")

    def on_warm(self, model_name: str, callback: Callable):
        """Register a callback for when a model becomes warm.

        Args:
            model_name: Model to watch.
            callback: Function(model_name) to call when warm.
        """
        with self._lock:
            if model_name not in self._on_warm_callbacks:
                self._on_warm_callbacks[model_name] = []
            self._on_warm_callbacks[model_name].append(callback)

            # Fire immediately if already warm
            if self._status.get(model_name) == WarmStatus.WARM:
                try:
                    callback(model_name)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def get_status(self, model_name: str) -> WarmStatus:
        """Get the warming status of a model."""
        with self._lock:
            return self._status.get(model_name, WarmStatus.COLD)

    def is_warm(self, model_name: str) -> bool:
        """Check if a model is warm (in page cache)."""
        return self.get_status(model_name) == WarmStatus.WARM

    def is_warming(self, model_name: str) -> bool:
        """Check if a model is currently being warmed."""
        return self.get_status(model_name) == WarmStatus.WARMING

    def get_progress(self, model_name: str) -> float:
        """Get warming progress (0.0 to 1.0)."""
        with self._lock:
            return self._progress.get(model_name, 0.0)

    def wait_for_warm(self, model_name: str, timeout: Optional[float] = None) -> bool:
        """Wait for a model to be warm.

        Args:
            model_name: Model to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            True if model is warm, False if timeout or error.
        """
        with self._lock:
            future = self._futures.get(model_name)

        if future is None:
            return self.is_warm(model_name)

        try:
            future.result(timeout=timeout)
            return self.is_warm(model_name)
        except Exception:
            return False

    def mark_cold(self, model_name: str):
        """Mark a model as cold (e.g., after page cache was cleared).

        Call this if you know the page cache was cleared (e.g., by
        running `echo 3 > /proc/sys/vm/drop_caches`).
        """
        with self._lock:
            self._status[model_name] = WarmStatus.COLD
            self._progress[model_name] = 0.0
            self._futures.pop(model_name, None)

    def get_error(self, model_name: str) -> Optional[str]:
        """Get error message for a failed warming."""
        with self._lock:
            return self._errors.get(model_name)

    def cancel(self, model_name: str) -> bool:
        """Cancel a pending warming operation."""
        with self._lock:
            future = self._futures.get(model_name)
            if future is None:
                return False

            cancelled = future.cancel()
            if cancelled:
                self._status[model_name] = WarmStatus.COLD
                self._futures.pop(model_name, None)

            return cancelled

    def get_stats(self) -> Dict:
        """Get warmer statistics."""
        with self._lock:
            return {
                'models': {
                    name: {
                        'status': status.name,
                        'progress': self._progress.get(name, 0.0),
                        'error': self._errors.get(name),
                    }
                    for name, status in self._status.items()
                },
                'warming_count': sum(
                    1 for s in self._status.values()
                    if s == WarmStatus.WARMING
                ),
                'warm_count': sum(
                    1 for s in self._status.values()
                    if s == WarmStatus.WARM
                ),
            }

    def shutdown(self, wait: bool = True):
        """Shutdown the warmer."""
        logger.info("Shutting down cache warmer")
        self._executor.shutdown(wait=wait)

    def __del__(self):
        """Clean up resources."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
