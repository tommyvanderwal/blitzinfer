"""Main orchestrator controller for BlitzInfer."""

import asyncio
import logging
import time
from typing import Optional, Any, Union
from dataclasses import dataclass

from .model_state import ModelState, ModelStatus, ModelRegistry
from .standby_manager import StandbyManager, StandbyState
from ..engine.vllm_adapter import VLLMEngine, GenerationResult
from ..config import ModelConfig, BlitzInferConfig
from ..memory import (
    PinnedMemoryArena,
    ModelPrefetcher,
    PrefetchStatus,
    PageCacheWarmer,
    WarmStatus,
    set_preloaded_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class SwitchMetrics:
    """Metrics from a model switch operation."""
    from_model: Optional[str]
    to_model: str
    unload_time: float
    load_time: float
    total_time: float
    queued_requests_at_switch: int


class BlitzInferOrchestrator:
    """Main orchestrator for multi-model serving with fast switching."""

    def __init__(self, config: BlitzInferConfig):
        self.config = config
        self.registry = ModelRegistry()
        self.engine = VLLMEngine()
        self._lock = asyncio.Lock()
        self._switch_history: list[SwitchMetrics] = []

        # Initialize prefetch system if enabled
        # Two approaches: page cache warming (default) or pinned arena
        self.prefetcher: Optional[ModelPrefetcher] = None
        self.cache_warmer: Optional[PageCacheWarmer] = None
        self._arena: Optional[PinnedMemoryArena] = None

        # Initialize standby manager for fast model switching
        # Uses pinned CPU memory for 1 model ready for fast GPU transfer
        self.standby: Optional[StandbyManager] = None
        if config.standby_enabled:
            logger.info(f"Initializing standby manager with {config.standby_arena_gb}GB arena")
            self.standby = StandbyManager(arena_size_gb=config.standby_arena_gb)

        if config.prefetch.enabled:
            if config.prefetch.use_page_cache:
                # Page cache warming: simpler, no memory allocation needed
                # Pre-reads files into OS page cache for faster vLLM loading
                logger.info("Initializing page cache warmer for model prefetch")
                self.cache_warmer = PageCacheWarmer(
                    max_workers=config.prefetch.max_concurrent_loads,
                    chunk_size=config.prefetch.chunk_size_mb * 1024 * 1024,
                )
            else:
                # Pinned arena: allocates large pinned memory block
                # Faster GPU transfer but requires dedicated RAM
                logger.info(f"Initializing prefetch system with {config.prefetch.arena_size_gb}GB arena")
                self._arena = PinnedMemoryArena(config.prefetch.arena_size_gb)
                self.prefetcher = ModelPrefetcher(
                    arena=self._arena,
                    max_workers=config.prefetch.max_concurrent_loads,
                )

        # Register all configured models
        for model_config in config.models:
            self.register_model(model_config)

    def register_model(self, config: ModelConfig):
        """Register a model for serving."""
        self.registry.register(config.name)
        self.engine.register_model(config)

        # Register model path with standby manager if available
        if self.standby is not None and hasattr(config, 'path') and config.path:
            self.standby.register_model(config.name, config.path)

        logger.info(f"Registered model: {config.display_name}")

    def on_request_queued(self, model_name: str):
        """Called when a request enters the queue for a model.

        Triggers standby prefetch if the request is for a different model.
        This enables demand-driven prefetch: we only prefetch when there's
        actual demand for a different model.

        Args:
            model_name: The model that the request is for.
        """
        if self.standby is None:
            return

        current = self.engine.current_model
        if model_name != current:
            # Start prefetching this model if standby is available
            logger.debug(f"Request for {model_name} (current: {current}), triggering standby prefetch")
            self.standby.start_prefetch(model_name)

    async def ensure_model_loaded(self, model_name: str) -> float:
        """Ensure a specific model is loaded, switching if needed.

        Returns: Time spent on loading/switching (0 if already loaded).
        """
        async with self._lock:
            if self.engine.current_model == model_name:
                return 0.0

            # Perform the switch
            return await self._switch_to_model(model_name)

    async def _switch_to_model(self, model_name: str) -> float:
        """Internal: Switch to a model (must hold lock).

        Priority order for loading:
        1. Standby slot (fastest - premerged weights in pinned RAM)
        2. Arena prefetch (pre-loaded tensors)
        3. Page cache warm (files in OS page cache)
        4. Cold load (from SSD)
        """
        from_model = self.engine.current_model
        status = self.registry.get_or_register(model_name)
        queued_at_switch = status.queue_depth

        # Check standby status (highest priority - fastest path)
        use_standby = (
            self.standby is not None
            and self.standby.is_ready(model_name)
        )

        # Check prefetch status - page cache or arena
        is_cache_warm = (
            self.cache_warmer is not None
            and self.cache_warmer.is_warm(model_name)
        )
        use_arena_prefetch = (
            self.prefetcher is not None
            and self.prefetcher.is_ready(model_name)
        )

        if use_standby:
            logger.info(f"Switching from {from_model} to {model_name} (STANDBY - fast path)")
        elif is_cache_warm:
            logger.info(f"Switching from {from_model} to {model_name} (page cache warm)")
        elif use_arena_prefetch:
            logger.info(f"Switching from {from_model} to {model_name} (arena prefetched)")
        else:
            logger.info(f"Switching from {from_model} to {model_name} (cold load)")

        start_time = time.time()

        # Get standby tensors BEFORE unloading (must consume while model is still known)
        standby_tensors = None
        if use_standby:
            standby_tensors = self.standby.consume_standby()
            if standby_tensors is None:
                # Standby was evicted between check and consume
                logger.warning("Standby was evicted, falling back to standard load")
                use_standby = False

        # Unload current model
        unload_start = time.time()
        if from_model:
            self.engine.unload_model()
            old_status = self.registry.get(from_model)
            if old_status:
                old_status.state = ModelState.COLD
        unload_time = time.time() - unload_start

        # Load new model using the appropriate method
        load_start = time.time()
        if use_standby and standby_tensors is not None:
            # FAST PATH: Load from standby (pinned RAM -> GPU)
            load_time = self._load_from_standby(model_name, standby_tensors)
        elif use_arena_prefetch:
            load_time = self.engine.load_from_prefetch(model_name, self.prefetcher)
            self.prefetcher.mark_transfer_complete(model_name)
        else:
            # Standard load - will be fast if page cache is warm
            load_time = self.engine.load_model(model_name)

        # Update state
        status.state = ModelState.SERVING
        status.load_time = load_time
        self.registry.active_model = model_name

        total_time = time.time() - start_time

        # Record metrics
        metrics = SwitchMetrics(
            from_model=from_model,
            to_model=model_name,
            unload_time=unload_time,
            load_time=load_time,
            total_time=total_time,
            queued_requests_at_switch=queued_at_switch,
        )
        self._switch_history.append(metrics)

        if use_standby:
            load_type = "[STANDBY]"
        elif is_cache_warm:
            load_type = "[WARM]"
        elif use_arena_prefetch:
            load_type = "[PREFETCH]"
        else:
            load_type = "[COLD]"

        logger.info(
            f"Switch complete: {from_model} -> {model_name} "
            f"(unload: {unload_time:.2f}s, load: {load_time:.2f}s, total: {total_time:.2f}s) "
            f"{load_type}"
        )
        return total_time

    def _load_from_standby(
        self,
        model_name: str,
        premerged_tensors: dict,
    ) -> float:
        """Load model from standby slot using premerged tensors.

        This is the fast path for model switching. The weights are already
        in pinned CPU RAM and premerged for vLLM compatibility.

        Args:
            model_name: Name of the model to load.
            premerged_tensors: Dict of tensor name -> pinned tensor.

        Returns:
            Load time in seconds.
        """
        import torch

        logger.info(f"Loading {model_name} from standby...")
        start_time = time.time()

        # Calculate tensor size for logging
        total_bytes = sum(
            t.numel() * t.element_size() for t in premerged_tensors.values()
        )
        logger.debug(f"Standby tensors: {len(premerged_tensors)} ({total_bytes / 1e9:.2f}GB)")

        # Set preloaded weights for vLLM's pinned arena loader
        set_preloaded_weights(premerged_tensors)

        # Get model config
        config = self.engine._get_config(model_name)

        # Build LLM kwargs
        llm_kwargs = {
            "model": config.name,
            "load_format": "pinned_arena",  # Use our custom loader
            "dtype": config.dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enforce_eager": config.enforce_eager,
            "trust_remote_code": config.trust_remote_code,
        }

        if config.kv_cache_memory_bytes:
            llm_kwargs["kv_cache_memory_bytes"] = config.kv_cache_memory_bytes
        if config.compilation_config:
            llm_kwargs["compilation_config"] = config.compilation_config

        # Import LLM here to avoid circular imports
        from vllm import LLM

        # Create LLM with pinned arena loader
        self.engine._llm = LLM(**llm_kwargs)
        self.engine._current_model = model_name

        load_time = time.time() - start_time
        speed = (total_bytes / 1e9) / load_time if load_time > 0 else 0
        logger.info(
            f"Loaded {model_name} from standby in {load_time:.2f}s "
            f"({speed:.1f} GB/s effective)"
        )

        return load_time

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using a specific model.

        Will switch models if needed. Triggers standby prefetch on queue entry.
        """
        # Update queue
        status = self.registry.get_or_register(model)
        status.increment_queue()

        # TRIGGER STANDBY PREFETCH when request enters queue for a different model
        # This is the primary prefetch mechanism for fast switching
        self.on_request_queued(model)

        # Also trigger page cache/arena prefetch if enabled (legacy)
        prefetch_enabled = self.cache_warmer is not None or self.prefetcher is not None
        if (prefetch_enabled
            and self.config.prefetch.trigger_on_queue_entry
            and not status.is_loaded):
            self._trigger_prefetch(model)

        try:
            # Ensure model is loaded
            switch_time = await self.ensure_model_loaded(model)
            if switch_time > 0:
                logger.info(f"Model switch took {switch_time:.2f}s")

            # Generate
            status.request_started()
            try:
                result = self.engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                return result
            finally:
                status.request_completed()
        except Exception:
            status.decrement_queue()
            raise

    def _trigger_prefetch(self, model_name: str):
        """Trigger prefetch for a model if not already in progress.

        Uses page cache warming if enabled, otherwise arena prefetch.
        """
        # Try page cache warming first (default, simpler)
        if self.cache_warmer is not None:
            warm_status = self.cache_warmer.get_status(model_name)
            if warm_status == WarmStatus.COLD:
                logger.info(f"Triggering page cache warming for {model_name}")
                self.cache_warmer.start_warming(model_name)
            return

        # Fall back to arena prefetch
        if self.prefetcher is not None:
            prefetch_status = self.prefetcher.get_status(model_name)
            if prefetch_status == PrefetchStatus.COLD:
                logger.info(f"Triggering arena prefetch for {model_name}")
                self.prefetcher.start_prefetch(model_name)

    def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        status = {
            "active_model": self.registry.active_model,
            "models": {
                m.name: {
                    "state": m.state.name,
                    "queue_depth": m.queue_depth,
                    "active_requests": m.active_requests,
                    "total_served": m.total_requests_served,
                    "last_load_time": m.load_time,
                }
                for m in self.registry.all_models
            },
            "switch_count": len(self._switch_history),
            "last_switch": (
                {
                    "from": self._switch_history[-1].from_model,
                    "to": self._switch_history[-1].to_model,
                    "total_time": self._switch_history[-1].total_time,
                }
                if self._switch_history else None
            ),
        }

        # Add standby info if enabled
        if self.standby is not None:
            status["standby"] = self.standby.get_stats()

        # Add prefetch info if enabled
        if self.cache_warmer is not None:
            status["prefetch"] = {
                "type": "page_cache",
                **self.cache_warmer.get_stats(),
            }
        elif self.prefetcher is not None:
            status["prefetch"] = {
                "type": "arena",
                **self.prefetcher.get_stats(),
            }

        return status

    def get_switch_metrics(self) -> list[SwitchMetrics]:
        """Get all recorded switch metrics."""
        return self._switch_history.copy()

    async def preload_model(self, model_name: str) -> float:
        """Preload a model without generating any text.

        Returns: Load time in seconds.
        """
        return await self.ensure_model_loaded(model_name)

    def register_model_path(self, model_name: str, model_path: str):
        """Register a model path for warming and standby prefetch.

        Call this to register HuggingFace model paths.
        This is automatically done for models resolved via HF cache,
        but can be called manually for custom paths.
        """
        if self.standby is not None:
            self.standby.register_model(model_name, model_path)

        if self.cache_warmer is not None:
            self.cache_warmer.register_model(model_name, model_path)

        logger.debug(f"Registered model path for {model_name}: {model_path}")

    def start_warming(self, model_name: str) -> bool:
        """Explicitly start warming a model in the background.

        Returns True if warming started or model is already warm.
        """
        if self.cache_warmer is not None:
            return self.cache_warmer.start_warming(model_name)
        elif self.prefetcher is not None:
            return self.prefetcher.start_prefetch(model_name)
        return False

    def is_model_warm(self, model_name: str) -> bool:
        """Check if a model is warm (page cache) or prefetched (arena)."""
        if self.cache_warmer is not None:
            return self.cache_warmer.is_warm(model_name)
        elif self.prefetcher is not None:
            return self.prefetcher.is_ready(model_name)
        return False

    def get_warm_status(self, model_name: str) -> str:
        """Get the warming status of a model.

        Returns: "COLD", "WARMING", "WARM", or "ERROR"
        """
        if self.cache_warmer is not None:
            return self.cache_warmer.get_status(model_name).name
        elif self.prefetcher is not None:
            return self.prefetcher.get_status(model_name).name
        return "COLD"

    def get_warm_progress(self, model_name: str) -> float:
        """Get warming progress (0.0 to 1.0)."""
        if self.cache_warmer is not None:
            return self.cache_warmer.get_progress(model_name)
        elif self.prefetcher is not None:
            return self.prefetcher.get_progress(model_name)
        return 0.0

    def warm_other_models(self):
        """Start warming all models except the currently loaded one.

        Call this after loading a model to start pre-warming alternatives
        in the background.
        """
        current = self.engine.current_model
        for status in self.registry.all_models:
            if status.name != current:
                self.start_warming(status.name)

    def start_standby_prefetch(self, model_name: str) -> bool:
        """Explicitly start prefetching a model to the standby slot.

        Args:
            model_name: Name of the model to prefetch.

        Returns:
            True if prefetch started or model is already ready.
        """
        if self.standby is not None:
            return self.standby.start_prefetch(model_name)
        return False

    def is_standby_ready(self, model_name: str) -> bool:
        """Check if a model is ready in the standby slot.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if model is ready for fast GPU transfer.
        """
        if self.standby is not None:
            return self.standby.is_ready(model_name)
        return False

    def get_standby_model(self) -> Optional[str]:
        """Get the name of the model currently in standby, if any."""
        if self.standby is not None:
            return self.standby.get_standby_model()
        return None

    def get_standby_state(self) -> str:
        """Get the current standby state.

        Returns:
            "EMPTY", "LOADING", "READY", or "ERROR"
        """
        if self.standby is not None:
            return self.standby.get_state().name
        return "DISABLED"

    def shutdown(self):
        """Shutdown the orchestrator and release resources."""
        logger.info("Shutting down BlitzInfer orchestrator")
        self.engine.unload_model()

        if self.standby is not None:
            self.standby.shutdown()

        if self.cache_warmer is not None:
            self.cache_warmer.shutdown()

        if self.prefetcher is not None:
            self.prefetcher.shutdown()

        if self._arena is not None:
            self._arena.clear()
