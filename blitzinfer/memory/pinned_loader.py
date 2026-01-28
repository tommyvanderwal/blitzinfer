"""Custom vLLM model loader that loads weights from pinned memory arena.

This loader bypasses disk I/O by injecting pre-loaded weights from
pinned CPU memory, achieving ~48 GB/s transfer speeds instead of
the typical ~3 GB/s from safetensors.

IMPORTANT: Weights should be kept in PINNED CPU memory (not GPU).
The loader transfers them directly to model parameters during load_weights.

Usage:
    # 1. Pre-load weights into pinned arena (during inference)
    arena = PinnedMemoryArena(80)
    load_model_to_arena("model_path", arena, "model_name")

    # 2. Get pinned tensor views (NOT transferred to GPU yet)
    pinned_tensors = arena.get_all_tensors(model_name)

    # 3. Register loader and create LLM with injected weights
    from blitzinfer.memory.pinned_loader import (
        PinnedArenaModelLoader,
        set_preloaded_weights,
    )
    set_preloaded_weights(pinned_tensors)  # Pinned CPU tensors!

    llm = LLM(model=model_name, load_format="pinned_arena", ...)
"""

import logging
import re
from typing import Dict, Optional, Tuple

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


def _permute_mistral_weight(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute Mistral wq/wk weights for vLLM's rotary embedding format.

    vLLM expects rotary embeddings in interleaved format, but Mistral
    safetensors store them in a different format.

    Args:
        w: Weight tensor of shape [n_heads * head_dim, hidden_size]
        n_heads: Number of heads

    Returns:
        Permuted weight tensor with same shape.
    """
    attn_in = w.shape[0]
    attn_out = w.shape[1]
    head_dim = attn_in // n_heads

    return (
        w.view(n_heads, head_dim // 2, 2, attn_out)
        .transpose(1, 2)
        .reshape(attn_in, attn_out)
    )


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

    This loader expects weights to be pre-loaded into GPU memory via
    `set_preloaded_weights()` before LLM instantiation. The weights are
    transferred from pinned CPU memory to GPU at ~48 GB/s, bypassing
    the slow safetensors parsing (~3 GB/s).

    Benefits:
        - ~10x faster weight loading (48 GB/s vs 3 GB/s)
        - Pre-loading can happen during inference (hidden latency)
        - Enables ~6s model switches instead of ~20s
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

    def _normalize_weight_name(self, name: str) -> str:
        """Normalize weight name to handle different prefix conventions.

        Handles mappings like:
        - model.language_model.layers.X -> language_model.model.layers.X
        - model.visual.X -> visual.X
        - lm_head.X -> language_model.lm_head.X
        """
        # Common prefix transformations
        transformations = [
            # Qwen-VL style: model.language_model.X <-> language_model.model.X
            ("model.language_model.", "language_model.model."),
            ("language_model.model.", "model.language_model."),
            # lm_head without prefix -> with prefix
            ("language_model.lm_head.", "lm_head."),
            # Simple model. prefix removal/addition
            ("model.", ""),
        ]

        for old_prefix, new_prefix in transformations:
            if name.startswith(old_prefix):
                return new_prefix + name[len(old_prefix):]

        return name

    def _try_merge_weights(
        self,
        param_name: str,
        param: torch.Tensor,
        preloaded: Dict[str, torch.Tensor],
        normalized_preloaded: Dict[str, tuple],
        model_config: Optional[ModelConfig] = None,
    ) -> Optional[torch.Tensor]:
        """Try to merge component weights into a merged weight.

        Handles vLLM's merged weights:
        - qkv_proj = concat(q_proj, k_proj, v_proj)
        - gate_up_proj = concat(gate_proj, up_proj)

        For Mistral format, also applies permutation to q_proj and k_proj.
        """
        # Detect merged weight patterns
        merge_patterns = [
            # (merged_name_part, component_suffixes, concat_dim)
            ("qkv_proj", ["q_proj", "k_proj", "v_proj"], 0),
            ("gate_up_proj", ["gate_proj", "up_proj"], 0),
        ]

        # Check if model is Mistral-based (needs permutation)
        is_mistral = False
        num_attention_heads = None
        num_kv_heads = None
        if model_config is not None:
            model_type = getattr(model_config.hf_config, 'model_type', '')
            # Handle various Mistral model types
            is_mistral = model_type in ['mistral', 'pixtral', 'mistral3']
            if is_mistral:
                # Check both top-level and nested text_config for attention heads
                num_attention_heads = getattr(model_config.hf_config, 'num_attention_heads', None)
                num_kv_heads = getattr(model_config.hf_config, 'num_key_value_heads', None)
                # Try text_config if top-level doesn't have it
                text_config = getattr(model_config.hf_config, 'text_config', None)
                if text_config and num_attention_heads is None:
                    num_attention_heads = getattr(text_config, 'num_attention_heads', None)
                    num_kv_heads = getattr(text_config, 'num_key_value_heads', num_attention_heads)
                if num_kv_heads is None:
                    num_kv_heads = num_attention_heads
                logger.debug(f"Mistral-style model detected: {model_type}, heads={num_attention_heads}, kv_heads={num_kv_heads}")

        for merged_part, components, concat_dim in merge_patterns:
            if merged_part not in param_name:
                continue

            # Extract base path (everything before the merged part)
            # e.g., "language_model.model.layers.0.self_attn.qkv_proj.weight"
            #    -> "language_model.model.layers.0.self_attn."
            merged_idx = param_name.find(merged_part)
            base_path = param_name[:merged_idx]
            suffix = param_name[merged_idx + len(merged_part):]  # e.g., ".weight" or ".weight_scale_inv"

            # Try to find all component weights
            component_tensors = []
            all_found = True

            for comp in components:
                comp_name = base_path + comp + suffix

                # Try normalized lookup
                norm_comp_name = self._normalize_weight_name(comp_name)
                tensor = None

                if norm_comp_name in normalized_preloaded:
                    _, tensor = normalized_preloaded[norm_comp_name]
                elif comp_name in normalized_preloaded:
                    _, tensor = normalized_preloaded[comp_name]
                else:
                    # Try direct lookup in preloaded
                    for pname, ptensor in preloaded.items():
                        if comp in pname and suffix in pname:
                            # Check if same layer
                            if self._same_layer(param_name, pname):
                                tensor = ptensor
                                break

                if tensor is None:
                    all_found = False
                    break

                # Apply Mistral permutation for q_proj and k_proj weights
                if is_mistral and suffix == ".weight" and merged_part == "qkv_proj":
                    if comp == "q_proj" and num_attention_heads:
                        tensor = _permute_mistral_weight(tensor, num_attention_heads)
                        logger.debug(f"Permuted q_proj for {param_name} with {num_attention_heads} heads")
                    elif comp == "k_proj" and num_kv_heads:
                        tensor = _permute_mistral_weight(tensor, num_kv_heads)
                        logger.debug(f"Permuted k_proj for {param_name} with {num_kv_heads} heads")

                component_tensors.append(tensor)

            if all_found and component_tensors:
                # Merge the tensors
                try:
                    merged = torch.cat(component_tensors, dim=concat_dim)
                    if merged.shape == param.shape:
                        logger.debug(f"Merged {components} -> {merged_part} for {param_name}")
                        return merged
                except Exception as e:
                    logger.debug(f"Failed to merge {components}: {e}")

        return None

    def _same_layer(self, name1: str, name2: str) -> bool:
        """Check if two weight names are from the same layer."""
        import re
        # Extract layer number pattern like "layers.X." or "blocks.X."
        pattern = r'(?:layers|blocks)\.(\d+)\.'
        match1 = re.search(pattern, name1)
        match2 = re.search(pattern, name2)
        if match1 and match2:
            return match1.group(1) == match2.group(1)
        return False

    def _can_fit_with_padding(self, param_shape: torch.Size, weight_shape: torch.Size) -> bool:
        """Check if weight can fit in param with padding.

        For MXFP4/GPT-OSS, vLLM pads tensor dimensions for Marlin kernel alignment.
        The weight from safetensors may be smaller than the padded parameter.
        Returns True if weight fits as a subregion of param.
        """
        if len(param_shape) != len(weight_shape):
            return False

        # Check each dimension: weight must be <= param
        for p_dim, w_dim in zip(param_shape, weight_shape):
            if w_dim > p_dim:
                return False

        # At least one dimension must be different (otherwise shapes would match exactly)
        return param_shape != weight_shape

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Inject pre-loaded weights into the model.

        This method expects weights to have been set via `set_preloaded_weights()`
        before LLM creation. Weights should be in PINNED CPU memory for fast
        transfer to GPU (~48 GB/s).

        The method matches weight names between the pre-loaded dict and the
        model's parameters, handling potential name mismatches.
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

        logger.info(f"Injecting {len(preloaded)} pre-loaded weights into model")
        t0 = time.perf_counter()

        # Build mapping from model parameter names to pre-loaded weight names
        # vLLM model names may differ from safetensor names due to prefixes
        injected = 0
        skipped = 0
        mismatched = 0
        merged = 0
        total_bytes = 0

        # Get all model parameters
        model_params = dict(model.named_parameters())

        # Build normalized name mapping for preloaded weights
        # Handle different prefix conventions between safetensor and vLLM
        normalized_preloaded = {}
        for name, tensor in preloaded.items():
            # Normalize: model.language_model.X -> language_model.model.X
            # and vice versa for different model types
            norm_name = self._normalize_weight_name(name)
            normalized_preloaded[norm_name] = (name, tensor)
            # Also store original name
            normalized_preloaded[name] = (name, tensor)

        # Inject weights
        for param_name, param in model_params.items():
            weight = None
            preloaded_name = None

            # Try direct match (normalized)
            norm_param_name = self._normalize_weight_name(param_name)
            if norm_param_name in normalized_preloaded:
                preloaded_name, weight = normalized_preloaded[norm_param_name]
            elif param_name in normalized_preloaded:
                preloaded_name, weight = normalized_preloaded[param_name]

            # Check for merged weights (qkv_proj, gate_up_proj)
            if weight is None:
                merged_weight = self._try_merge_weights(
                    param_name, param, preloaded, normalized_preloaded, model_config
                )
                if merged_weight is not None:
                    weight = merged_weight
                    preloaded_name = f"merged:{param_name}"

            if weight is not None:
                if param.shape == weight.shape:
                    # Transfer directly from pinned CPU to GPU parameter
                    if weight.device.type == 'cuda':
                        param.data.copy_(weight)
                    else:
                        param.data.copy_(weight.to(param.device, non_blocking=True))
                    injected += 1
                    if "merged:" in str(preloaded_name):
                        merged += 1
                    total_bytes += param.numel() * param.element_size()
                elif self._can_fit_with_padding(param.shape, weight.shape):
                    # Handle padded tensors (e.g., MXFP4/GPT-OSS where vLLM pads for Marlin)
                    # Copy weight into subregion of larger padded parameter
                    if weight.dim() == 3:
                        d0, d1, d2 = weight.shape
                        if weight.device.type == 'cuda':
                            param.data[:d0, :d1, :d2].copy_(weight)
                        else:
                            param.data[:d0, :d1, :d2].copy_(weight.to(param.device, non_blocking=True))
                    elif weight.dim() == 2:
                        d0, d1 = weight.shape
                        if weight.device.type == 'cuda':
                            param.data[:d0, :d1].copy_(weight)
                        else:
                            param.data[:d0, :d1].copy_(weight.to(param.device, non_blocking=True))
                    else:
                        # Fallback for other dimensions
                        param.data[tuple(slice(0, s) for s in weight.shape)].copy_(
                            weight if weight.device.type == 'cuda' else weight.to(param.device, non_blocking=True)
                        )
                    injected += 1
                    if "merged:" in str(preloaded_name):
                        merged += 1
                    total_bytes += weight.numel() * weight.element_size()
                    logger.debug(
                        f"Padded copy for {param_name}: "
                        f"weight={weight.shape} -> param={param.shape}"
                    )
                else:
                    logger.debug(
                        f"Shape mismatch for {param_name}: "
                        f"model={param.shape}, preloaded={weight.shape}"
                    )
                    mismatched += 1
            else:
                skipped += 1
                if skipped <= 10:
                    logger.debug(f"Skipped (no match): {param_name}")

        # Sync to ensure all transfers complete
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bandwidth = (total_bytes / 1e9) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Weight injection: {injected} injected ({total_bytes/1e9:.2f}GB), "
            f"{merged} merged, {skipped} skipped, {mismatched} mismatched "
            f"in {elapsed:.2f}s ({bandwidth:.1f} GB/s)"
        )

        if injected == 0:
            logger.error(
                "Failed to inject any weights! Model will have random weights. "
                "This likely indicates a name mismatch between safetensor weights "
                "and vLLM model parameters."
            )
        elif skipped > injected:
            logger.warning(
                f"Many weights skipped ({skipped} vs {injected} injected). "
                "This may indicate partial weight loading."
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
