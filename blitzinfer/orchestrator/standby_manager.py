"""Manages single standby slot for fast model switching.

The standby manager maintains one model pre-loaded in pinned CPU RAM,
ready for fast GPU transfer when switching models. This enables ~5s
model switches instead of ~18s cold loads from SSD.

Architecture:
    GPU VRAM: Active model (weights + KV cache)
    Pinned CPU RAM: Standby model (premerged weights, ~70GB slot)
    NVMe SSD: All other models (cold storage)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class StandbyState(Enum):
    """State of the standby slot."""
    EMPTY = auto()      # No model in standby
    LOADING = auto()    # Loading from SSD to pinned RAM
    READY = auto()      # Ready for fast GPU transfer
    ERROR = auto()      # Load failed


@dataclass
class StandbySlot:
    """Represents the current state of the standby slot."""
    model_name: Optional[str] = None
    state: StandbyState = StandbyState.EMPTY
    premerged_tensors: Optional[Dict[str, torch.Tensor]] = None
    load_time: float = 0.0
    error: Optional[str] = None
    model_size_gb: float = 0.0


class StandbyManager:
    """Manages a single standby slot in pinned CPU memory.

    The standby manager provides fast model switching by pre-loading
    model weights into pinned CPU RAM during inference. When a switch
    is requested, the weights are transferred from pinned RAM to GPU
    at PCIe bandwidth (~44 GB/s) instead of loading from SSD.

    Usage:
        manager = StandbyManager(arena_size_gb=70.0)
        manager.register_model("model-a", "/path/to/model-a")

        # Start prefetch when request arrives for different model
        manager.start_prefetch("model-a")

        # Check if ready
        if manager.is_ready("model-a"):
            tensors = manager.consume_standby()
            # Use tensors for fast GPU load

    Thread Safety:
        All public methods are thread-safe. Background loading runs in
        a separate thread and updates state atomically.
    """

    def __init__(
        self,
        arena_size_gb: float = 80.0,
        chunk_size_gb: float = 16.0,
        pin_memory: bool = True,
        lazy_arena: bool = False,
    ):
        """Initialize the standby manager.

        Args:
            arena_size_gb: Size of the memory arena in GB.
                          Should be large enough for your largest model.
                          Default 80GB (5x 16GB) fits GPT-OSS-120B (~65GB) with margin.
            chunk_size_gb: Size of each arena chunk in GB (default 16GB).
                          MUST be a power of 2 (1, 2, 4, 8, 16, 32GB) to avoid
                          PyTorch's power-of-2 rounding overhead.
                          See: https://github.com/pytorch/pytorch/issues/150517
                          16GB chunks give 0% overhead and ~44 GB/s transfer.
            pin_memory: Whether to use CUDA pinned memory (default True).
                       Pinned memory achieves ~44 GB/s transfer vs ~4.6 GB/s non-pinned.
            lazy_arena: If False (default), pre-allocate arena at init.
                       If True, allocate on first prefetch (original behavior).
        """
        from ..memory import PinnedMemoryArena

        self._arena_size_gb = arena_size_gb
        self._chunk_size_gb = chunk_size_gb
        self._pin_memory = pin_memory
        self._slot = StandbySlot()
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls
        self._load_thread: Optional[threading.Thread] = None
        self._model_paths: Dict[str, str] = {}  # model_name -> path
        self._shutdown_event = threading.Event()

        # Pre-allocate arena unless lazy mode requested
        if lazy_arena:
            self._arena: Optional['PinnedMemoryArena'] = None
        else:
            num_chunks = int(arena_size_gb / chunk_size_gb)
            mem_type = "pinned" if pin_memory else "regular"
            logger.info(
                f"Pre-allocating {arena_size_gb}GB {mem_type} arena "
                f"({num_chunks}x{chunk_size_gb}GB chunks)..."
            )
            self._arena = PinnedMemoryArena(
                arena_size_gb,
                pin_memory=pin_memory,
                chunk_size_gb=chunk_size_gb
            )
            logger.info(f"Arena ready: {self._arena.size_gb:.1f}GB")

    def register_model(self, model_name: str, model_path: str):
        """Register a model's path for loading.

        Args:
            model_name: Name/identifier for the model.
            model_path: Path to the model directory containing safetensors.
        """
        self._model_paths[model_name] = model_path
        logger.debug(f"Registered model path: {model_name} -> {model_path}")

    def get_state(self) -> StandbyState:
        """Get current standby state."""
        with self._lock:
            return self._slot.state

    def get_standby_model(self) -> Optional[str]:
        """Get name of model in standby, or None if empty/loading."""
        with self._lock:
            if self._slot.state == StandbyState.READY:
                return self._slot.model_name
            return None

    def is_ready(self, model_name: str) -> bool:
        """Check if specific model is ready in standby.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if the model is loaded and ready for GPU transfer.
        """
        with self._lock:
            return (self._slot.state == StandbyState.READY and
                    self._slot.model_name == model_name)

    def start_prefetch(self, model_name: str) -> bool:
        """Start background prefetch of model to standby slot.

        This is the main entry point for demand-driven prefetch.
        Call this when a request arrives for a different model.

        Args:
            model_name: Name of the model to prefetch.

        Returns:
            True if prefetch started or model is already ready.
            False if slot is busy loading another model.
        """
        import sys
        print(f"[STANDBY] start_prefetch called for {model_name}", flush=True)
        print(f"[STANDBY] Acquiring lock...", flush=True)

        with self._lock:
            print(f"[STANDBY] Lock acquired, checking state...", flush=True)
            print(f"[STANDBY] Current state: {self._slot.state}, current model: {self._slot.model_name}", flush=True)
            # Already have this model ready
            if self.is_ready(model_name):
                print(f"[STANDBY] Model {model_name} already ready in standby", flush=True)
                return True

            # Slot is busy loading
            if self._slot.state == StandbyState.LOADING:
                if self._slot.model_name == model_name:
                    print(f"[STANDBY] Model {model_name} already loading", flush=True)
                    return True
                print(f"[STANDBY] Standby busy loading {self._slot.model_name}", flush=True)
                return False

            # Evict current standby if different model
            if self._slot.model_name and self._slot.model_name != model_name:
                print(f"[STANDBY] Evicting {self._slot.model_name}", flush=True)
                self._evict_standby_locked()

            # Start background load
            self._slot.model_name = model_name
            self._slot.state = StandbyState.LOADING
            self._slot.error = None
            print(f"[STANDBY] Creating loader thread for {model_name}", flush=True)
            self._load_thread = threading.Thread(
                target=self._load_worker,
                args=(model_name,),
                daemon=True,
                name=f"standby-loader-{model_name}"
            )
            self._load_thread.start()
            print(f"[STANDBY] Loader thread started for {model_name}", flush=True)
            logger.info(f"Started prefetch for {model_name}")
            return True

    def _load_worker(self, model_name: str):
        """Background worker to load model into standby.

        This runs in a separate thread and loads the model's safetensors
        into pinned memory, then pre-merges weights for vLLM compatibility.
        """
        import sys
        from huggingface_hub import snapshot_download

        from ..memory import (
            PinnedMemoryArena,
            load_model_to_arena,
            get_model_size,
            get_premerged_tensors_for_vllm,
        )

        print(f"[STANDBY-WORKER] Starting load worker for {model_name}", flush=True)
        t0 = time.perf_counter()
        try:
            # Get model path
            print(f"[STANDBY-WORKER] Looking up model path...", flush=True)
            if model_name in self._model_paths:
                model_path = Path(self._model_paths[model_name])
                print(f"[STANDBY-WORKER] Using registered path: {model_path}", flush=True)
            else:
                # Try HuggingFace cache
                print(f"[STANDBY-WORKER] Looking up {model_name} in HuggingFace cache...", flush=True)
                model_path = Path(snapshot_download(model_name, local_files_only=True))
                print(f"[STANDBY-WORKER] Found in cache: {model_path}", flush=True)

            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            print(f"[STANDBY-WORKER] Getting model size...", flush=True)
            model_size = get_model_size(str(model_path))
            model_gb = model_size / 1e9
            print(f"[STANDBY-WORKER] Model size: {model_gb:.1f}GB", flush=True)

            # Allocate arena if needed (lazy mode only)
            if self._arena is None:
                num_chunks = int(self._arena_size_gb / self._chunk_size_gb)
                mem_type = "pinned" if self._pin_memory else "regular"
                print(f"[STANDBY-WORKER] Allocating {self._arena_size_gb}GB {mem_type} arena "
                      f"({num_chunks}x{self._chunk_size_gb}GB chunks)...", flush=True)
                arena_start = time.perf_counter()
                self._arena = PinnedMemoryArena(
                    self._arena_size_gb,
                    pin_memory=self._pin_memory,
                    chunk_size_gb=self._chunk_size_gb
                )
                arena_time = time.perf_counter() - arena_start
                print(f"[STANDBY-WORKER] Arena allocated in {arena_time:.1f}s", flush=True)
            else:
                print(f"[STANDBY-WORKER] Using pre-allocated arena ({self._arena.size_gb:.1f}GB)", flush=True)

            # Check if model fits
            if model_gb > self._arena_size_gb:
                raise MemoryError(
                    f"Model {model_name} ({model_gb:.1f}GB) exceeds "
                    f"arena size ({self._arena_size_gb}GB)"
                )

            # Check for shutdown
            if self._shutdown_event.is_set():
                print(f"[STANDBY-WORKER] Shutdown requested, aborting", flush=True)
                return

            # Load into arena with parallel I/O for maximum throughput
            print(f"[STANDBY-WORKER] Loading model to arena...", flush=True)
            load_start = time.perf_counter()
            load_model_to_arena(str(model_path), self._arena, model_name, parallel_workers=8)
            load_time = time.perf_counter() - load_start
            load_speed = model_gb / load_time if load_time > 0 else 0
            print(f"[STANDBY-WORKER] Loaded {model_name} in {load_time:.1f}s ({load_speed:.1f} GB/s)", flush=True)

            # Check for shutdown
            if self._shutdown_event.is_set():
                print(f"[STANDBY-WORKER] Shutdown requested, aborting premerge", flush=True)
                self._arena.release(model_name)
                return

            # Get pinned tensors and premerge for vLLM
            print(f"[STANDBY-WORKER] Getting pinned tensors...", flush=True)
            premerge_start = time.perf_counter()
            pinned_tensors = self._arena.get_all_tensors(model_name)
            print(f"[STANDBY-WORKER] Got {len(pinned_tensors)} tensors, premerging...", flush=True)
            premerged = get_premerged_tensors_for_vllm(pinned_tensors)
            premerge_time = time.perf_counter() - premerge_start
            print(f"[STANDBY-WORKER] Pre-merged {len(premerged)} tensors in {premerge_time:.1f}s", flush=True)

            total_time = time.perf_counter() - t0

            with self._lock:
                # Double-check we haven't been evicted
                if self._slot.model_name != model_name:
                    print(f"[STANDBY-WORKER] WARNING: Slot was reassigned during load", flush=True)
                    self._arena.release(model_name)
                    return

                self._slot.premerged_tensors = premerged
                self._slot.load_time = total_time
                self._slot.model_size_gb = model_gb
                self._slot.state = StandbyState.READY
                self._slot.error = None

            print(f"[STANDBY-WORKER] READY: {model_name} ({model_gb:.1f}GB) in {total_time:.1f}s", flush=True)
            logger.info(f"Standby ready: {model_name} ({model_gb:.1f}GB) in {total_time:.1f}s")

        except Exception as e:
            print(f"[STANDBY-WORKER] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            logger.error(f"Standby load failed for {model_name}: {e}")
            with self._lock:
                if self._slot.model_name == model_name:
                    self._slot.state = StandbyState.ERROR
                    self._slot.error = str(e)

    def get_premerged_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get premerged tensors from standby slot without consuming.

        Returns:
            Dict of tensor name -> tensor, or None if not ready.
            The tensors remain in standby after this call.
        """
        with self._lock:
            if self._slot.state != StandbyState.READY:
                return None
            return self._slot.premerged_tensors

    def consume_standby(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get tensors and mark slot as empty (model moving to GPU).

        This is the main method for switching. Call this when you're
        ready to transfer the standby model to GPU.

        Returns:
            Dict of tensor name -> tensor, or None if not ready.
            After this call, the standby slot is empty.
        """
        with self._lock:
            if self._slot.state != StandbyState.READY:
                return None

            tensors = self._slot.premerged_tensors
            model_name = self._slot.model_name

            # Clear slot but keep arena allocation (will be released later)
            self._slot = StandbySlot()

            logger.info(f"Consumed standby: {model_name}")
            return tensors

    def release_consumed(self, model_name: str):
        """Release arena allocation after GPU transfer completes.

        Call this after the model has been fully loaded to GPU to free
        the arena memory for the next prefetch.
        """
        if self._arena is not None:
            try:
                self._arena.release(model_name)
                logger.info(f"Released arena for consumed model: {model_name}")
            except Exception as e:
                logger.debug(f"Arena release for {model_name}: {e}")

    def _evict_standby_locked(self):
        """Evict current standby (must hold lock)."""
        if self._slot.model_name and self._arena:
            try:
                self._arena.release(self._slot.model_name)
                logger.debug(f"Evicted {self._slot.model_name} from standby")
            except Exception as e:
                logger.warning(f"Failed to release arena: {e}")
        self._slot = StandbySlot()

    def evict_standby(self):
        """Evict current standby model (release pinned memory)."""
        with self._lock:
            self._evict_standby_locked()

    def wait_for_load(self, timeout: float = 120.0) -> bool:
        """Wait for current load to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if model is ready, False if timeout or error.
        """
        if self._load_thread is not None:
            self._load_thread.join(timeout=timeout)
        return self._slot.state == StandbyState.READY

    def get_load_progress(self) -> float:
        """Get approximate loading progress (0.0 to 1.0).

        Note: This is a rough estimate based on arena allocation,
        not actual file read progress.
        """
        with self._lock:
            if self._slot.state == StandbyState.READY:
                return 1.0
            if self._slot.state != StandbyState.LOADING:
                return 0.0
            # Rough estimate based on arena status
            if self._arena is None or self._slot.model_name is None:
                return 0.0
            alloc = self._arena.get_allocation(self._slot.model_name)
            if alloc is None:
                return 0.1  # Allocation started
            if alloc.status == 'loading':
                return 0.5  # Loading in progress
            if alloc.status == 'ready':
                return 0.9  # Loaded, premerging
            return 0.3  # Unknown state

    def get_stats(self) -> Dict:
        """Get standby manager statistics."""
        with self._lock:
            stats = {
                'state': self._slot.state.name,
                'model': self._slot.model_name,
                'model_size_gb': self._slot.model_size_gb,
                'load_time': self._slot.load_time,
                'error': self._slot.error,
                'arena_size_gb': self._arena_size_gb,
            }
            if self._arena is not None:
                stats.update({
                    'arena_used_gb': self._arena.used_gb,
                    'arena_available_gb': self._arena.available_gb,
                })
            return stats

    def shutdown(self):
        """Clean shutdown of the standby manager.

        Signals any running load thread to stop, waits for it,
        and releases all resources.
        """
        logger.info("Shutting down standby manager...")
        self._shutdown_event.set()

        # Wait for load thread
        if self._load_thread is not None and self._load_thread.is_alive():
            logger.debug("Waiting for load thread to finish...")
            self._load_thread.join(timeout=5.0)
            if self._load_thread.is_alive():
                logger.warning("Load thread did not terminate in time")

        # Release arena
        with self._lock:
            self._evict_standby_locked()
            if self._arena is not None:
                self._arena.clear()
                self._arena = None

        logger.info("Standby manager shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
        except Exception:
            pass
