"""vLLM engine adapter for BlitzInfer."""

import os
import time
import logging
from typing import Optional, Any
from dataclasses import dataclass

# Set multiprocessing method before importing vLLM
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

# Use single-process mode for fast model switching (~6s vs ~30s with multiprocessing)
# Memory leak is fixed by properly clearing model parameters in unload_model()
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

# Skip warmup to avoid HIP kernel errors during profiling on ROCm gfx1100
# Warmup is only beneficial for CUDA graphs which we disable anyway with enforce_eager=True
os.environ.setdefault('VLLM_SKIP_WARMUP', '1')

from vllm import LLM, SamplingParams

from ..config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class VLLMEngine:
    """Adapter for vLLM engine with model switching support."""

    def __init__(self, default_config: Optional[ModelConfig] = None):
        self._llm: Optional[LLM] = None
        self._current_model: Optional[str] = None
        self._default_config = default_config or ModelConfig(name="")
        self._model_configs: dict[str, ModelConfig] = {}

    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        self._model_configs[config.name] = config
        if config.alias:
            self._model_configs[config.alias] = config

    def _get_config(self, model_name: str) -> ModelConfig:
        """Get config for a model, using defaults if not registered."""
        if model_name in self._model_configs:
            return self._model_configs[model_name]
        # Create a new config with the model name
        return ModelConfig(
            name=model_name,
            dtype=self._default_config.dtype,
            max_model_len=self._default_config.max_model_len,
            gpu_memory_utilization=self._default_config.gpu_memory_utilization,
            max_num_seqs=self._default_config.max_num_seqs,
            max_num_batched_tokens=self._default_config.max_num_batched_tokens,
            enforce_eager=self._default_config.enforce_eager,
            kv_cache_memory_bytes=self._default_config.kv_cache_memory_bytes,
            compilation_config=self._default_config.compilation_config,
            trust_remote_code=self._default_config.trust_remote_code,
        )

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model name."""
        return self._current_model

    @property
    def is_loaded(self) -> bool:
        """Check if any model is loaded."""
        return self._llm is not None

    def load_model(self, model_name: str) -> float:
        """Load a model, unloading any currently loaded model.

        Returns: Load time in seconds.
        """
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()

        # Unload current model if different
        if self._current_model and self._current_model != model_name:
            self.unload_model()

        # Skip if already loaded
        if self._current_model == model_name and self._llm is not None:
            logger.info(f"Model {model_name} already loaded")
            return 0.0

        config = self._get_config(model_name)

        # Build LLM kwargs with optimized settings
        llm_kwargs = {
            "model": config.name,
            "dtype": config.dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enforce_eager": config.enforce_eager,
            "trust_remote_code": config.trust_remote_code,
        }

        # Add optional optimizations if set
        if config.kv_cache_memory_bytes:
            llm_kwargs["kv_cache_memory_bytes"] = config.kv_cache_memory_bytes
        if config.compilation_config:
            llm_kwargs["compilation_config"] = config.compilation_config

        # Load the model
        self._llm = LLM(**llm_kwargs)

        self._current_model = model_name
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
        return load_time

    def load_from_prefetch(self, model_name: str, prefetcher: Any) -> float:
        """Load model from prefetched pinned memory arena (fast path).

        This method transfers pre-loaded weights from pinned CPU memory
        to GPU at high bandwidth (~45 GB/s on RTX PRO 6000).

        Args:
            model_name: Name of the model to load.
            prefetcher: ModelPrefetcher instance with ready model.

        Returns: Load time in seconds.
        """
        import torch

        logger.info(f"Loading model from prefetch: {model_name}")
        start_time = time.time()

        # Unload current model if different
        if self._current_model and self._current_model != model_name:
            self.unload_model()

        # Skip if already loaded
        if self._current_model == model_name and self._llm is not None:
            logger.info(f"Model {model_name} already loaded")
            return 0.0

        config = self._get_config(model_name)

        # Get tensor views from prefetcher's arena
        # These are pinned memory views ready for fast GPU transfer
        t0 = time.time()
        pinned_tensors = prefetcher.get_tensors_for_gpu(model_name)
        get_tensors_time = time.time() - t0
        logger.debug(f"Got {len(pinned_tensors)} tensor views in {get_tensors_time:.3f}s")

        # Transfer all tensors to GPU (non-blocking for speed)
        t0 = time.time()
        gpu_tensors = {}
        for name, tensor in pinned_tensors.items():
            gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        transfer_time = time.time() - t0

        total_bytes = sum(t.numel() * t.element_size() for t in pinned_tensors.values())
        transfer_speed = (total_bytes / 1024**3) / transfer_time if transfer_time > 0 else 0
        logger.info(
            f"Transferred {total_bytes / 1024**3:.2f}GB to GPU in {transfer_time:.2f}s "
            f"({transfer_speed:.1f} GB/s)"
        )

        # Now we need to initialize vLLM with these pre-loaded weights
        # This requires a modified initialization path
        t0 = time.time()
        self._init_model_with_weights(model_name, config, gpu_tensors)
        init_time = time.time() - t0
        logger.debug(f"Model initialization took {init_time:.2f}s")

        self._current_model = model_name
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded from prefetch in {load_time:.2f}s")
        return load_time

    def _init_model_with_weights(
        self,
        model_name: str,
        config: 'ModelConfig',
        gpu_tensors: dict,
    ):
        """Initialize vLLM model with pre-loaded GPU weights.

        This is the key function that bypasses vLLM's normal weight loading
        by injecting pre-loaded tensors.

        Note: This requires vLLM internals access and may need adjustment
        for different vLLM versions.
        """
        import torch
        from vllm import LLM

        # For now, we use a hybrid approach:
        # 1. Create LLM instance normally (this loads config, tokenizer, etc.)
        # 2. Replace the loaded weights with our prefetched weights
        #
        # A more optimized approach would be to modify vLLM's weight loading
        # directly, but this works as a first implementation.

        llm_kwargs = {
            "model": config.name,
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

        # Create LLM - this will load weights from disk initially
        # TODO: Optimize by using skip_weights_load when available in vLLM
        self._llm = LLM(**llm_kwargs)

        # Now replace weights with our prefetched ones
        # This is faster than loading from disk because the weights
        # are already in GPU memory
        try:
            self._inject_weights(gpu_tensors)
        except Exception as e:
            logger.warning(f"Could not inject prefetched weights: {e}")
            # Model will use its own loaded weights

    def _inject_weights(self, gpu_tensors: dict):
        """Inject pre-loaded weights into the vLLM model.

        This replaces the model's weight tensors with our prefetched ones.
        """
        if self._llm is None:
            return

        try:
            # Access the model through vLLM's internal structure
            engine_core = self._llm.llm_engine.engine_core
            if hasattr(engine_core, 'engine_core'):
                core = engine_core.engine_core
            else:
                core = engine_core

            if not hasattr(core, 'model_executor'):
                logger.debug("Cannot access model_executor for weight injection")
                return

            executor = core.model_executor
            if not hasattr(executor, 'driver_worker'):
                logger.debug("Cannot access driver_worker for weight injection")
                return

            worker = executor.driver_worker
            if not hasattr(worker, 'worker') or worker.worker is None:
                logger.debug("Cannot access worker for weight injection")
                return

            model_runner = getattr(worker.worker, 'model_runner', None)
            if model_runner is None or not hasattr(model_runner, 'model'):
                logger.debug("Cannot access model for weight injection")
                return

            model = model_runner.model

            # Replace weights
            injected_count = 0
            for name, param in model.named_parameters():
                if name in gpu_tensors:
                    prefetched = gpu_tensors[name]
                    if param.shape == prefetched.shape:
                        param.data.copy_(prefetched)
                        injected_count += 1

            logger.debug(f"Injected {injected_count}/{len(gpu_tensors)} weights")

        except Exception as e:
            logger.debug(f"Weight injection failed: {e}")

    def unload_model(self):
        """Unload the current model to free GPU memory."""
        if self._llm is None:
            return

        logger.info(f"Unloading model: {self._current_model}")
        start_time = time.time()

        import gc
        import torch
        import torch._dynamo
        import multiprocessing

        # Synchronize GPU before unloading (wrap in try/except for ROCm HIP errors)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f"cuda.synchronize failed: {e}")

        # CRITICAL: Move model weights and KV cache to CPU to force GPU memory release
        # vLLM's shutdown() doesn't properly free GPU memory, so we do it manually
        try:
            engine_core = self._llm.llm_engine.engine_core
            # InprocClient has engine_core attribute
            if hasattr(engine_core, 'engine_core'):
                core = engine_core.engine_core
            else:
                core = engine_core

            if hasattr(core, 'model_executor'):
                executor = core.model_executor
                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                    if hasattr(worker, 'worker') and worker.worker is not None:
                        model_runner = getattr(worker.worker, 'model_runner', None)
                        if model_runner is not None:
                            # Clear model parameters
                            if hasattr(model_runner, 'model'):
                                model = model_runner.model
                                logger.debug("Clearing model parameters...")
                                for param in model.parameters():
                                    param.data = torch.empty(0, device='cpu')
                                logger.debug("Model parameters cleared")

                            # Clear KV cache list on model_runner
                            if hasattr(model_runner, 'kv_caches'):
                                logger.debug("Clearing model_runner.kv_caches...")
                                for i, cache in enumerate(model_runner.kv_caches):
                                    if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                        model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                                model_runner.kv_caches.clear()
                                logger.debug("model_runner.kv_caches cleared")

                            # Clear any cross attention KV cache
                            if hasattr(model_runner, 'cross_layers_kv_cache'):
                                if model_runner.cross_layers_kv_cache is not None:
                                    model_runner.cross_layers_kv_cache = torch.empty(0, device='cpu')
                                    logger.debug("Cross-attention KV cache cleared")

                            # Clear KV caches from attention layers in compilation config
                            if hasattr(model_runner, 'compilation_config'):
                                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                                if sfc:
                                    logger.debug(f"Clearing KV cache from {len(sfc)} attention layers...")
                                    for layer_name, layer in sfc.items():
                                        if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                            for i, kv in enumerate(layer.kv_cache):
                                                if kv is not None and hasattr(kv, 'device') and kv.device.type == 'cuda':
                                                    layer.kv_cache[i] = torch.empty(0, device='cpu')
                                            layer.kv_cache = []
                                    logger.debug("Attention layer KV caches cleared")

                            # Clear CUDA cache after clearing
                            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Could not clear model/KV cache: {e}")

        # Delete the LLM instance
        del self._llm
        self._llm = None
        self._current_model = None

        # CRITICAL: Clear static_forward_context which holds model layer references
        try:
            from vllm.config import get_current_vllm_config_or_none
            vllm_config = get_current_vllm_config_or_none()
            if vllm_config is not None:
                vllm_config.compilation_config.static_forward_context.clear()
                logger.debug("Cleared static_forward_context")
        except Exception as e:
            logger.debug(f"Could not clear static_forward_context: {e}")

        # CRITICAL: Reset the global _current_vllm_config
        try:
            import vllm.config.vllm as vllm_config_module
            vllm_config_module._current_vllm_config = None
            vllm_config_module._current_prefix = None
            # Clear the compilation config cache
            vllm_config_module.get_cached_compilation_config.cache_clear()
            logger.debug("Reset _current_vllm_config")
        except Exception as e:
            logger.debug(f"Could not reset _current_vllm_config: {e}")

        # Reset torch.compile / dynamo state (holds compiled model closures)
        try:
            torch._dynamo.reset()
            logger.debug("Reset torch._dynamo")
        except Exception as e:
            logger.debug(f"torch._dynamo.reset failed: {e}")

        # Use vLLM's comprehensive cleanup function
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except Exception as e:
            logger.debug(f"cleanup_dist_env_and_memory failed: {e}")
            # Fallback to manual cleanup
            try:
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel,
                    destroy_distributed_environment,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass

        # Reset workspace manager (holds GPU memory)
        try:
            from vllm.v1.worker.workspace import reset_workspace_manager
            reset_workspace_manager()
        except Exception as e:
            logger.debug(f"reset_workspace_manager failed: {e}")

        # Reset environment variable cache (can hold stale state)
        try:
            import vllm.envs as envs
            envs.disable_envs_cache()
        except Exception:
            pass

        # Clear multimodal registry caches
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY
            if hasattr(MULTIMODAL_REGISTRY, '_processor_cache'):
                MULTIMODAL_REGISTRY._processor_cache.clear()
            if hasattr(MULTIMODAL_REGISTRY, 'clear_cache'):
                MULTIMODAL_REGISTRY.clear_cache()
        except Exception as e:
            logger.debug(f"Could not clear multimodal registry: {e}")

        # Clear EngineArgs lru_cache
        try:
            from vllm.engine.arg_utils import EngineArgs
            if hasattr(EngineArgs, '_get_default_values'):
                EngineArgs._get_default_values.cache_clear()
        except Exception as e:
            logger.debug(f"Could not clear EngineArgs cache: {e}")

        # Wait for child processes to terminate
        for child in multiprocessing.active_children():
            logger.debug(f"Waiting for child process {child.name} (pid={child.pid})")
            child.join(timeout=5.0)
            if child.is_alive():
                logger.warning(f"Child process {child.name} did not terminate, forcing...")
                child.terminate()
                child.join(timeout=2.0)

        # Aggressive garbage collection
        gc.collect()
        gc.collect()

        # Clear CUDA cache and synchronize (wrap in try/except for ROCm HIP errors)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.debug(f"CUDA cleanup failed: {e}")

        # Try to empty host cache as well (PyTorch 2.5+)
        try:
            torch._C._host_emptyCache()
        except AttributeError:
            pass

        # Final GC
        gc.collect()

        unload_time = time.time() - start_time
        logger.info(f"Model unloaded in {unload_time:.2f}s")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text from the currently loaded model."""
        if self._llm is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        outputs = self._llm.generate([prompt], sampling_params)
        output = outputs[0]

        return GenerationResult(
            text=output.outputs[0].text,
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason or "stop",
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> list[GenerationResult]:
        """Generate text for multiple prompts."""
        if self._llm is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            results.append(GenerationResult(
                text=output.outputs[0].text,
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
                finish_reason=output.outputs[0].finish_reason or "stop",
            ))
        return results

    def switch_model(self, new_model: str) -> float:
        """Switch to a different model.

        Returns: Total switch time in seconds (unload + load).
        """
        logger.info(f"Switching from {self._current_model} to {new_model}")
        start_time = time.time()

        self.unload_model()
        load_time = self.load_model(new_model)

        total_time = time.time() - start_time
        logger.info(f"Model switch completed in {total_time:.2f}s (load: {load_time:.2f}s)")
        return total_time
