"""Background model prefetching from SSD to pinned memory arena."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Callable

import torch

from .arena import PinnedMemoryArena
from .fast_loader import load_model_to_arena, get_model_size, get_safetensor_files

logger = logging.getLogger(__name__)


class PrefetchStatus(Enum):
    """Status of a model in the prefetch system."""
    COLD = auto()        # On disk only
    LOADING = auto()     # Currently loading to arena
    READY = auto()       # In arena, ready for GPU transfer
    TRANSFERRING = auto()  # Currently transferring to GPU
    ERROR = auto()       # Failed to load


class ModelPrefetcher:
    """Background prefetching from SSD to pinned arena.

    This class manages background loading of models from SSD into pinned
    memory. When a request enters the queue for a model, prefetching can
    be triggered so the model is ready for fast GPU transfer when needed.

    Usage:
        arena = PinnedMemoryArena(80)  # 80GB
        prefetcher = ModelPrefetcher(arena, model_paths={
            'mistral-7b': '/models/mistral',
            'qwen-7b': '/models/qwen',
        })

        # Trigger prefetch (non-blocking)
        prefetcher.start_prefetch('mistral-7b')

        # Check if ready
        if prefetcher.is_ready('mistral-7b'):
            tensors = prefetcher.get_tensors_for_gpu('mistral-7b')
    """

    def __init__(
        self,
        arena: PinnedMemoryArena,
        model_paths: Optional[Dict[str, str]] = None,
        max_workers: int = 1,
    ):
        """Initialize the prefetcher.

        Args:
            arena: PinnedMemoryArena to load models into.
            model_paths: Dict mapping model names to their paths.
            max_workers: Maximum concurrent prefetch operations.
        """
        self._arena = arena
        self._model_paths: Dict[str, str] = model_paths or {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._status: Dict[str, PrefetchStatus] = {}
        self._futures: Dict[str, Future] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.RLock()

        # Callbacks for status changes
        self._on_ready_callbacks: Dict[str, list[Callable]] = {}

    def register_model(self, model_name: str, model_path: str):
        """Register a model path for prefetching.

        Args:
            model_name: Unique name for the model.
            model_path: Path to the model directory (containing .safetensors).
        """
        with self._lock:
            self._model_paths[model_name] = model_path
            if model_name not in self._status:
                self._status[model_name] = PrefetchStatus.COLD

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the path for a model.

        Handles both registered models and HuggingFace cache paths.
        """
        # Check registered paths first
        if model_name in self._model_paths:
            return self._model_paths[model_name]

        # Try HuggingFace cache
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_name, local_files_only=True)
            return path
        except Exception:
            pass

        # Try as direct path
        if Path(model_name).exists():
            return model_name

        return None

    def start_prefetch(self, model_name: str) -> bool:
        """Start background prefetch for a model.

        Non-blocking. The model will be loaded in a background thread.

        Args:
            model_name: Name of the model to prefetch.

        Returns:
            True if prefetch started or already in progress/ready.
            False if unable to start (no path, no space, etc.).
        """
        with self._lock:
            current_status = self._status.get(model_name, PrefetchStatus.COLD)

            # Already prefetched or in progress
            if current_status in (PrefetchStatus.READY, PrefetchStatus.LOADING, PrefetchStatus.TRANSFERRING):
                logger.debug(f"Model {model_name} already {current_status.name}")
                return True

            # Get model path
            model_path = self.get_model_path(model_name)
            if model_path is None:
                logger.warning(f"Cannot prefetch {model_name}: path not found")
                return False

            # Check if we have enough space
            try:
                model_size = get_model_size(model_path)
            except FileNotFoundError as e:
                logger.warning(f"Cannot prefetch {model_name}: {e}")
                return False

            if self._arena.available_bytes < model_size:
                logger.warning(
                    f"Cannot prefetch {model_name}: need {model_size / 1024**3:.1f}GB, "
                    f"available {self._arena.available_gb:.1f}GB"
                )
                # Could try eviction here
                return False

            # Start prefetch
            self._status[model_name] = PrefetchStatus.LOADING
            future = self._executor.submit(self._prefetch_worker, model_name, model_path)
            self._futures[model_name] = future

            logger.info(f"Started prefetch for {model_name} ({model_size / 1024**3:.1f}GB)")
            return True

    def _prefetch_worker(self, model_name: str, model_path: str):
        """Worker thread: loads safetensor files into arena."""
        try:
            logger.debug(f"Prefetch worker started for {model_name}")
            start_time = time.time()

            # Load model into arena
            load_model_to_arena(
                model_path=model_path,
                arena=self._arena,
                model_name=model_name,
            )

            elapsed = time.time() - start_time

            with self._lock:
                self._status[model_name] = PrefetchStatus.READY

            logger.info(f"Prefetch complete for {model_name} in {elapsed:.2f}s")

            # Fire callbacks
            self._fire_ready_callbacks(model_name)

        except Exception as e:
            logger.error(f"Prefetch failed for {model_name}: {e}")
            with self._lock:
                self._status[model_name] = PrefetchStatus.ERROR
                self._errors[model_name] = str(e)

    def _fire_ready_callbacks(self, model_name: str):
        """Fire registered callbacks when a model becomes ready."""
        callbacks = self._on_ready_callbacks.get(model_name, [])
        for callback in callbacks:
            try:
                callback(model_name)
            except Exception as e:
                logger.error(f"Callback error for {model_name}: {e}")

    def on_ready(self, model_name: str, callback: Callable):
        """Register a callback for when a model becomes ready.

        Args:
            model_name: Model to watch.
            callback: Function(model_name) to call when ready.
        """
        with self._lock:
            if model_name not in self._on_ready_callbacks:
                self._on_ready_callbacks[model_name] = []
            self._on_ready_callbacks[model_name].append(callback)

            # Fire immediately if already ready
            if self._status.get(model_name) == PrefetchStatus.READY:
                try:
                    callback(model_name)
                except Exception as e:
                    logger.error(f"Callback error for {model_name}: {e}")

    def get_status(self, model_name: str) -> PrefetchStatus:
        """Get the prefetch status of a model."""
        with self._lock:
            return self._status.get(model_name, PrefetchStatus.COLD)

    def is_ready(self, model_name: str) -> bool:
        """Check if a model is ready in the arena."""
        return self.get_status(model_name) == PrefetchStatus.READY

    def is_loading(self, model_name: str) -> bool:
        """Check if a model is currently being prefetched."""
        return self.get_status(model_name) == PrefetchStatus.LOADING

    def wait_for_ready(self, model_name: str, timeout: Optional[float] = None) -> bool:
        """Wait for a model to be ready.

        Args:
            model_name: Model to wait for.
            timeout: Maximum time to wait in seconds (None = forever).

        Returns:
            True if model is ready, False if timeout or error.
        """
        with self._lock:
            future = self._futures.get(model_name)

        if future is None:
            # Not loading - check if already ready
            return self.is_ready(model_name)

        try:
            future.result(timeout=timeout)
            return self.is_ready(model_name)
        except Exception:
            return False

    def get_tensors_for_gpu(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Get tensor views ready for GPU transfer.

        Args:
            model_name: Name of the prefetched model.

        Returns:
            Dict mapping tensor names to pinned memory tensor views.

        Raises:
            RuntimeError: If model is not ready.
        """
        with self._lock:
            status = self._status.get(model_name, PrefetchStatus.COLD)
            if status != PrefetchStatus.READY:
                raise RuntimeError(
                    f"Model {model_name} not ready for GPU transfer (status: {status.name})"
                )

            # Mark as transferring
            self._status[model_name] = PrefetchStatus.TRANSFERRING

        try:
            return self._arena.get_all_tensors(model_name)
        except Exception:
            # Revert status on error
            with self._lock:
                self._status[model_name] = PrefetchStatus.READY
            raise

    def mark_transfer_complete(self, model_name: str):
        """Mark a model's GPU transfer as complete.

        After transfer, the model can be released from the arena
        if space is needed.
        """
        with self._lock:
            if self._status.get(model_name) == PrefetchStatus.TRANSFERRING:
                # Keep as ready so it can be reused if model is unloaded from GPU
                self._status[model_name] = PrefetchStatus.READY

    def release(self, model_name: str):
        """Release a model from the arena.

        Args:
            model_name: Model to release.
        """
        with self._lock:
            self._arena.release(model_name)
            self._status[model_name] = PrefetchStatus.COLD
            self._futures.pop(model_name, None)
            self._errors.pop(model_name, None)

    def get_error(self, model_name: str) -> Optional[str]:
        """Get the error message for a failed prefetch."""
        with self._lock:
            return self._errors.get(model_name)

    def cancel(self, model_name: str) -> bool:
        """Cancel a pending prefetch.

        Args:
            model_name: Model to cancel.

        Returns:
            True if cancelled, False if not cancellable.
        """
        with self._lock:
            future = self._futures.get(model_name)
            if future is None:
                return False

            cancelled = future.cancel()
            if cancelled:
                self._status[model_name] = PrefetchStatus.COLD
                self._futures.pop(model_name, None)

            return cancelled

    def get_stats(self) -> Dict:
        """Get prefetcher statistics."""
        with self._lock:
            return {
                'arena': self._arena.get_stats(),
                'models': {
                    name: {
                        'status': status.name,
                        'error': self._errors.get(name),
                    }
                    for name, status in self._status.items()
                },
                'loading_count': sum(
                    1 for s in self._status.values()
                    if s == PrefetchStatus.LOADING
                ),
                'ready_count': sum(
                    1 for s in self._status.values()
                    if s == PrefetchStatus.READY
                ),
            }

    def shutdown(self, wait: bool = True):
        """Shutdown the prefetcher.

        Args:
            wait: If True, wait for pending prefetches to complete.
        """
        logger.info("Shutting down prefetcher")
        self._executor.shutdown(wait=wait)

    def __del__(self):
        """Clean up resources."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
