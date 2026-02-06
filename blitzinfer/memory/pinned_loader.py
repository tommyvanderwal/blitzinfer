"""Custom vLLM model loader that loads weights from pinned memory arena.

This loader bypasses disk I/O by injecting pre-loaded weights from
pinned CPU memory, achieving ~48 GB/s transfer speeds instead of
the typical ~3 GB/s from safetensors.

Key insight: instead of doing our own name mapping (which breaks for complex
models like vision encoders, AWQ, MoE, FLA), we delegate to vLLM's own
model.load_weights() method - the same interface the DefaultModelLoader uses.
The model class handles ALL name mapping, quantization repacking, MoE fusion,
etc. We just provide the raw tensor iterator.

Usage:
    # 1. Pre-load weights into pinned arena (during inference)
    arena = PinnedMemoryArena(80)
    load_model_to_arena("model_path", arena, "model_name")

    # 2. Get raw pinned tensor views (original safetensor names)
    pinned_tensors = arena.get_all_tensors(model_name)

    # 3. Register loader and create LLM with injected weights
    from blitzinfer.memory.pinned_loader import (
        PinnedArenaModelLoader,
        set_preloaded_weights,
    )
    set_preloaded_weights(pinned_tensors)  # Raw safetensor names!

    llm = LLM(model=model_name, load_format="pinned_arena", ...)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

# Global storage for pre-loaded weights
# This is set before LLM creation and consumed during load_weights
_PRELOADED_WEIGHTS: Optional[Dict[str, torch.Tensor]] = None


def set_preloaded_weights(weights: Dict[str, torch.Tensor]):
    """Set the pre-loaded weights to be used by the next LLM instantiation.

    Args:
        weights: Dict mapping parameter names to GPU tensors.
    """
    global _PRELOADED_WEIGHTS
    _PRELOADED_WEIGHTS = weights
    logger.info(f"Set {len(weights)} pre-loaded weights for pinned_arena loader")


def get_preloaded_weights() -> Optional[Dict[str, torch.Tensor]]:
    """Get the pre-loaded weights (and clear them)."""
    global _PRELOADED_WEIGHTS
    weights = _PRELOADED_WEIGHTS
    _PRELOADED_WEIGHTS = None
    return weights


def clear_preloaded_weights():
    """Clear any stored pre-loaded weights."""
    global _PRELOADED_WEIGHTS
    _PRELOADED_WEIGHTS = None


@register_model_loader("pinned_arena")
class PinnedArenaModelLoader(BaseModelLoader):
    """Model loader that uses pre-loaded weights from pinned memory arena.

    Instead of reading safetensors from disk, this loader feeds pre-loaded
    pinned CPU tensors to vLLM's own model.load_weights() method. This gives
    us all of vLLM's name mapping, quantization repacking, MoE fusion, vision
    encoder handling, etc. for free - while reading from pinned memory at
    ~48 GB/s instead of ~3 GB/s from SSD.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                "Model loader extra config is not supported for "
                "load format pinned_arena"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        """No download needed - weights are pre-loaded in memory."""
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Inject pre-loaded weights into the model via model.load_weights().

        This delegates to vLLM's own model.load_weights() method, passing our
        pinned arena tensors as a weight iterator - the same interface that
        DefaultModelLoader uses with safetensor weights from disk. The model
        class handles ALL name mapping, quantization repacking, MoE expert
        fusion, vision encoder mapping, etc.

        Weights should be raw safetensor tensors in pinned CPU memory.
        """
        import time

        preloaded = get_preloaded_weights()

        if preloaded is None:
            logger.warning(
                "No pre-loaded weights found! Call set_preloaded_weights() before "
                "creating LLM with load_format='pinned_arena'. "
                "Falling back to random initialization."
            )
            from vllm.model_executor.model_loader.weight_utils import (
                initialize_dummy_weights,
            )
            initialize_dummy_weights(model)
            return

        total_bytes = sum(t.numel() * t.element_size() for t in preloaded.values())
        logger.info(
            f"Injecting {len(preloaded)} weights ({total_bytes / 1e9:.2f}GB) "
            f"via model.load_weights()"
        )
        t0 = time.perf_counter()

        # Feed raw safetensor tensors to the model's own load_weights() method.
        # This is the same interface vLLM's DefaultModelLoader uses.
        # The model class handles ALL name mapping, quantization repacking,
        # MoE expert fusion, vision encoder mapping, FLA params, etc.
        loaded_weights = model.load_weights(preloaded.items())

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bandwidth = (total_bytes / 1e9) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Weight injection complete: {len(loaded_weights)} params loaded "
            f"({total_bytes / 1e9:.2f}GB) in {elapsed:.2f}s ({bandwidth:.1f} GB/s)"
        )


# Utility functions for the loading pipeline


def transfer_arena_to_gpu(
    arena: "PinnedMemoryArena",
    model_name: str,
) -> Dict[str, torch.Tensor]:
    """Transfer all tensors for a model from pinned arena to GPU.

    This is the fast path - achieves ~48 GB/s on PCIe 5.0.

    Args:
        arena: PinnedMemoryArena containing the model.
        model_name: Name of the model in the arena.

    Returns:
        Dict mapping tensor names to GPU tensors.
    """
    import time

    logger.info(f"Transferring {model_name} from pinned arena to GPU")
    t0 = time.perf_counter()

    # Get all tensor views from arena
    pinned_tensors = arena.get_all_tensors(model_name)

    # Transfer to GPU
    gpu_tensors = {}
    total_bytes = 0

    for name, tensor in pinned_tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
        total_bytes += tensor.numel() * tensor.element_size()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bandwidth = (total_bytes / 1e9) / elapsed if elapsed > 0 else 0

    logger.info(
        f"Transferred {total_bytes / 1e9:.2f}GB to GPU in {elapsed:.2f}s "
        f"({bandwidth:.1f} GB/s)"
    )

    return gpu_tensors
