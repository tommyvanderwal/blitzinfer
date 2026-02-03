"""Pre-merge weights for faster vLLM injection.

vLLM merges certain weights (qkv_proj, gate_up_proj) for efficiency.
This module pre-computes these merged tensors during background loading
so the injection phase only needs to copy, not merge.

Also handles model-specific naming conventions:
- Mistral: wq/wk/wv/wo/w1/w2/w3 -> q_proj/k_proj/v_proj/o_proj/gate_proj/down_proj/up_proj
- Qwen VL: model.language_model.X -> language_model.model.X
"""

import logging
import re
import time
from typing import Dict, Optional, Tuple, List

import torch

logger = logging.getLogger(__name__)

# Merge patterns: (merged_suffix, component_suffixes, concat_dim)
MERGE_PATTERNS = [
    ("qkv_proj", ["q_proj", "k_proj", "v_proj"], 0),
    ("gate_up_proj", ["gate_proj", "up_proj"], 0),
]

# Mistral-specific name mappings (from safetensor names to vLLM names)
# Based on vLLM's MistralForCausalLM.mistral_mapping
# Order matters! More specific patterns should be applied before less specific ones.
MISTRAL_NAME_MAPPING = [
    # Attention projections
    (".attention.wq.", ".self_attn.q_proj."),
    (".attention.wk.", ".self_attn.k_proj."),
    (".attention.wv.", ".self_attn.v_proj."),
    (".attention.wo.", ".self_attn.o_proj."),
    # MLP/FFN projections
    (".feed_forward.w1.", ".mlp.gate_proj."),
    (".feed_forward.w2.", ".mlp.down_proj."),
    (".feed_forward.w3.", ".mlp.up_proj."),
    # Layer norms (must be before generic norm mapping)
    (".attention_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    # Embeddings and output (only at start of name)
]

# These only apply at the START of the name
MISTRAL_PREFIX_MAPPING = [
    ("layers.", "model.layers."),
    ("tok_embeddings.", "model.embed_tokens."),
    ("output.", "lm_head."),
    ("norm.", "model.norm."),  # Final layer norm (only at start!)
]

# GPT-OSS-120B specific name mappings (from safetensor names to vLLM names)
# GPT-OSS uses MoE architecture with different weight naming conventions.
GPTOSS_NAME_MAPPING = [
    # Embedding
    (".embed_tokens.", ".embedding."),
    # Attention: self_attn -> attn
    (".self_attn.", ".attn."),
    # MoE expert projections (order matters - more specific first)
    (".mlp.experts.gate_up_proj_blocks", ".mlp.experts.w13_weight"),
    (".mlp.experts.gate_up_proj_scales", ".mlp.experts.w13_weight_scale"),
    (".mlp.experts.gate_up_proj_bias", ".mlp.experts.w13_bias"),
    (".mlp.experts.down_proj_blocks", ".mlp.experts.w2_weight"),
    (".mlp.experts.down_proj_scales", ".mlp.experts.w2_weight_scale"),
    (".mlp.experts.down_proj_bias", ".mlp.experts.w2_bias"),
]


def _permute_mistral_weight(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute Mistral wq/wk weights for vLLM's rotary embedding format.

    vLLM expects rotary embeddings in interleaved format, but Mistral
    safetensors store them in a different format. This function applies
    the same transformation as vLLM's maybe_remap_mistral.

    Args:
        w: Weight tensor of shape [n_heads * head_dim, hidden_size]
        n_heads: Number of heads (num_attention_heads for wq, num_kv_heads for wk)

    Returns:
        Permuted weight tensor with same shape.
    """
    attn_in = w.shape[0]
    attn_out = w.shape[1]
    head_dim = attn_in // n_heads

    # Permute: (n_heads, head_dim/2, 2, attn_out) -> (n_heads, 2, head_dim/2, attn_out)
    return (
        w.view(n_heads, head_dim // 2, 2, attn_out)
        .transpose(1, 2)
        .reshape(attn_in, attn_out)
    )


def _apply_mistral_mapping(name: str) -> str:
    """Apply Mistral naming convention transformation.

    Converts Mistral safetensor names to vLLM expected names:
    - layers.0.attention.wk.weight -> model.layers.0.self_attn.k_proj.weight
    - layers.0.feed_forward.w1.weight -> model.layers.0.mlp.gate_proj.weight
    - layers.0.attention_norm.weight -> model.layers.0.input_layernorm.weight
    """
    result = name

    # Apply prefix mapping first (only at start of name)
    for old, new in MISTRAL_PREFIX_MAPPING:
        if result.startswith(old):
            result = new + result[len(old):]
            break

    # Apply name mappings (anywhere in string, but ordered for specificity)
    for old, new in MISTRAL_NAME_MAPPING:
        if old in result:
            result = result.replace(old, new)

    return result


def _apply_gptoss_mapping(name: str) -> str:
    """Apply GPT-OSS-120B naming convention transformation.

    Converts GPT-OSS safetensor names to vLLM expected names:
    - model.embed_tokens.weight -> model.embedding.weight
    - model.layers.0.self_attn.q_proj.weight -> model.layers.0.attn.q_proj.weight
    - model.layers.0.mlp.experts.gate_up_proj_blocks -> model.layers.0.mlp.experts.w13_weight
    - model.layers.0.mlp.experts.down_proj_blocks -> model.layers.0.mlp.experts.w2_weight
    """
    result = name

    # Apply name mappings (in order, more specific first)
    for old, new in GPTOSS_NAME_MAPPING:
        if old in result:
            result = result.replace(old, new)

    return result


def _detect_model_format(tensor_names: List[str]) -> str:
    """Detect model format from tensor names.

    Returns:
        'gptoss' if GPT-OSS-120B naming (mlp.experts.gate_up_proj_blocks, etc.)
        'mistral' if Mistral naming (wq, wk, etc.)
        'qwen_vl' if Qwen VL naming (model.language_model.X)
        'standard' otherwise
    """
    sample = tensor_names[:100] if len(tensor_names) > 100 else tensor_names

    # Check for GPT-OSS naming (MoE with specific expert weight names)
    gptoss_patterns = [".mlp.experts.gate_up_proj_blocks", ".mlp.experts.down_proj_blocks"]
    for name in sample:
        for pattern in gptoss_patterns:
            if pattern in name:
                return "gptoss"

    # Check for Mistral naming
    mistral_patterns = [".wq.", ".wk.", ".wv.", ".w1.", ".w2.", ".w3."]
    for name in sample:
        for pattern in mistral_patterns:
            if pattern in name:
                return "mistral"

    # Check for Qwen VL naming
    for name in sample:
        if name.startswith("model.language_model."):
            return "qwen_vl"

    return "standard"


def premerge_vllm_weights(
    pinned_tensors: Dict[str, torch.Tensor],
    model_type: str = "auto",
) -> Dict[str, torch.Tensor]:
    """Pre-merge weights to match vLLM's expected format.

    Args:
        pinned_tensors: Dict of tensor name -> pinned CPU tensor from arena.
        model_type: Model type hint for name transformations.

    Returns:
        Dict with original tensors + merged tensors added.
        Merged tensor names follow vLLM convention.
    """
    logger.info(f"Pre-merging weights for vLLM ({len(pinned_tensors)} input tensors)")
    t0 = time.perf_counter()

    # Build a map for fast lookups
    result = dict(pinned_tensors)  # Start with all original tensors

    # Find layers and merge patterns
    merged_count = 0
    merged_bytes = 0

    # Group tensors by layer path
    layer_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in pinned_tensors.items():
        # Extract layer path (e.g., "model.language_model.layers.0.self_attn.")
        for merged_suffix, components, _ in MERGE_PATTERNS:
            for comp in components:
                if f".{comp}." in name:
                    # Found a component tensor
                    idx = name.find(f".{comp}.")
                    layer_path = name[:idx + 1]  # Include trailing dot
                    suffix = name[idx + len(comp) + 2:]  # e.g., "weight" or "weight_scale_inv"

                    key = (layer_path, suffix, merged_suffix)
                    if key not in layer_tensors:
                        layer_tensors[key] = {}
                    layer_tensors[key][comp] = tensor
                    break

    # Create merged tensors
    for (layer_path, suffix, merged_suffix), component_dict in layer_tensors.items():
        # Find the merge pattern
        for ms, components, concat_dim in MERGE_PATTERNS:
            if ms != merged_suffix:
                continue

            # Check if we have all components
            if all(c in component_dict for c in components):
                component_tensors = [component_dict[c] for c in components]

                # Verify shapes are compatible for concatenation
                ref_shape = component_tensors[0].shape
                compatible = True
                for t in component_tensors[1:]:
                    for i, (d1, d2) in enumerate(zip(ref_shape, t.shape)):
                        if i != concat_dim and d1 != d2:
                            compatible = False
                            break

                if not compatible:
                    continue

                # Create merged tensor
                try:
                    merged = torch.cat(component_tensors, dim=concat_dim)

                    # Generate merged name following vLLM convention
                    # Original: model.language_model.layers.0.self_attn.q_proj.weight
                    # Merged:   language_model.model.layers.0.self_attn.qkv_proj.weight
                    merged_name = f"{layer_path}{merged_suffix}.{suffix}"

                    # Also create normalized versions
                    normalized_name = _normalize_for_vllm(merged_name)

                    result[merged_name] = merged
                    if normalized_name != merged_name:
                        result[normalized_name] = merged

                    merged_count += 1
                    merged_bytes += merged.numel() * merged.element_size()

                    logger.debug(
                        f"Merged {components} -> {merged_suffix}: "
                        f"{[t.shape for t in component_tensors]} -> {merged.shape}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to merge {layer_path}{merged_suffix}: {e}")
                break

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Pre-merged {merged_count} weight pairs ({merged_bytes / 1e9:.2f}GB) "
        f"in {elapsed:.3f}s"
    )

    return result


def _normalize_for_vllm(name: str) -> str:
    """Normalize weight name to vLLM convention.

    Handles mappings like:
    - model.language_model.X -> language_model.model.X
    """
    transformations = [
        ("model.language_model.", "language_model.model."),
        ("lm_head.", "language_model.lm_head."),
    ]

    for old_prefix, new_prefix in transformations:
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]

    return name


def _add_pixtral_prefix(name: str) -> Optional[str]:
    """Add language_model. prefix for Pixtral models.

    Pixtral wraps the LLM in a `language_model` attribute, so parameter names
    look like `language_model.model.layers.0.self_attn.q_proj.weight`.

    Returns None if the name shouldn't have a prefix added (e.g., vision encoder).
    """
    # LLM weights start with model. and should get language_model. prefix
    if name.startswith("model."):
        return "language_model." + name

    # lm_head needs prefix too
    if name.startswith("lm_head."):
        return "language_model." + name

    return None


def get_premerged_tensors_for_vllm(
    pinned_tensors: Dict[str, torch.Tensor],
    remove_merged_components: bool = True,
    pin_merged: bool = False,  # Changed default to False - arena tensors are already pinned
    skip_merge: bool = True,  # CHANGED: Default True to avoid memory copies
) -> Dict[str, torch.Tensor]:
    """Get tensors ready for vLLM injection with pre-merged weights.

    This is the main entry point. Call this after loading into arena,
    before vLLM init.

    Handles model-specific naming conventions:
    - Mistral: wq/wk/wv -> q_proj/k_proj/v_proj, etc.
    - Qwen VL: model.language_model.X -> language_model.model.X

    Args:
        pinned_tensors: Original tensors from arena (already pinned).
        remove_merged_components: If True, remove original components that
            were merged (q_proj, k_proj, v_proj, gate_proj, up_proj).
            This prevents transferring duplicate data.
        pin_merged: If True, pin the merged tensors. Default False since
            arena tensors are already pinned and we want to avoid doubling
            memory usage. torch.cat on pinned tensors creates a contiguous
            copy that's already in pageable memory - GPU transfer is still
            fast from CPU RAM.
        skip_merge: If True (default), skip torch.cat merging to avoid creating
            large memory copies outside the arena. The pinned_loader will handle
            merging on-the-fly during GPU transfer. This is critical for memory
            efficiency - torch.cat creates new tensors that can easily exceed
            available RAM when combined with the 80GB arena.

    Returns:
        Dict with tensors needed for vLLM (merged + non-merged).
    """
    logger.info("Preparing pre-merged tensors for vLLM...")
    t0 = time.perf_counter()

    # Detect model format and apply name transformations
    model_format = _detect_model_format(list(pinned_tensors.keys()))
    logger.info(f"Detected model format: {model_format}")

    # Transform names if needed
    if model_format == "mistral":
        # For Mistral format, we do MINIMAL transformation:
        # - Apply name mapping (wq -> q_proj, etc.)
        # - Skip permutation (will be done during GPU injection)
        # - Skip QKV/gate_up merging (pinned_loader handles on-the-fly)
        # This keeps memory usage to just the original tensor size (~45GB)
        # instead of creating ~100GB of copies.
        logger.info("Applying Mistral name transformations (lightweight mode)...")
        transformed = {}

        for name, tensor in pinned_tensors.items():
            new_name = _apply_mistral_mapping(name)
            transformed[new_name] = tensor
            if new_name != name:
                logger.debug(f"  {name} -> {new_name}")
        pinned_tensors = transformed

        # Add language_model. prefixed versions for Pixtral compatibility
        # This just adds references to the same tensors (no extra memory)
        logger.info("Adding Pixtral-compatible language_model. prefixed versions...")
        for name, tensor in list(pinned_tensors.items()):
            prefixed_name = _add_pixtral_prefix(name)
            if prefixed_name and prefixed_name not in pinned_tensors:
                pinned_tensors[prefixed_name] = tensor

        # Return early - no QKV/gate_up merging for Mistral
        # pinned_loader's _try_merge_weights handles merging on-the-fly
        elapsed = time.perf_counter() - t0
        final_bytes = sum(t.numel() * t.element_size() for t in pinned_tensors.values())
        # Note: final_bytes counts duplicates from prefixed versions, but memory is shared
        unique_bytes = sum(t.numel() * t.element_size() for t in set(pinned_tensors.values()))
        logger.info(
            f"Pre-merge complete: {len(pinned_tensors)} tensors ({unique_bytes / 1e9:.2f}GB unique, "
            f"{final_bytes / 1e9:.2f}GB with prefixes) in {elapsed:.3f}s"
        )
        return dict(pinned_tensors)

    if model_format == "gptoss":
        # For GPT-OSS-120B (MoE model with MXFP4 quantization):
        # - Apply name mapping (embed_tokens -> embedding, self_attn -> attn, etc.)
        # - Transform MoE expert weight names (gate_up_proj_blocks -> w13_weight, etc.)
        # - CRITICAL: Reshape 4D blocked tensors to 3D format that vLLM expects
        #   Safetensor: [experts, size_n, num_blocks, 16] (blocked MXFP4)
        #   vLLM expects: [experts, size_n, num_blocks * 16] (contiguous)
        # - Let pinned_loader handle QKV merging on-the-fly
        logger.info("Applying GPT-OSS name transformations + 4D->3D reshape...")
        transformed = {}
        transform_count = 0
        reshape_count = 0

        for name, tensor in pinned_tensors.items():
            new_name = _apply_gptoss_mapping(name)

            # Reshape 4D blocked MXFP4 tensors to 3D
            # Safetensor format: [experts, size_n, num_blocks, block_size=16]
            # vLLM format: [experts, size_n, num_blocks * block_size]
            if tensor.dim() == 4 and tensor.shape[-1] == 16:
                # This is a blocked MXFP4 weight tensor
                orig_shape = tensor.shape
                new_shape = (tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])
                tensor = tensor.reshape(new_shape).contiguous()
                reshape_count += 1
                logger.debug(f"  Reshaped {name}: {list(orig_shape)} -> {list(new_shape)}")

            transformed[new_name] = tensor
            if new_name != name:
                transform_count += 1
                logger.debug(f"  {name} -> {new_name}")

        logger.info(f"Transformed {transform_count} tensor names, reshaped {reshape_count} 4D->3D tensors for GPT-OSS")
        pinned_tensors = transformed

        # Return early - let pinned_loader handle QKV merging
        elapsed = time.perf_counter() - t0
        unique_bytes = sum(t.numel() * t.element_size() for t in set(pinned_tensors.values()))
        logger.info(
            f"Pre-merge complete: {len(pinned_tensors)} tensors ({unique_bytes / 1e9:.2f}GB) "
            f"in {elapsed:.3f}s"
        )
        return dict(pinned_tensors)

    # For ALL model formats: skip torch.cat merging by default to avoid memory copies
    # The pinned_loader handles merging on-the-fly during GPU transfer
    # This is CRITICAL for memory efficiency - torch.cat creates new tensors that
    # can easily exceed available RAM (e.g., 16GB+ for Qwen-32B merged weights)
    if skip_merge:
        # Apply name normalization only
        result = dict(pinned_tensors)
        for name, tensor in list(result.items()):
            norm_name = _normalize_for_vllm(name)
            if norm_name not in result:
                result[norm_name] = tensor

        elapsed = time.perf_counter() - t0
        unique_bytes = sum(t.numel() * t.element_size() for t in set(result.values()))
        logger.info(
            f"Pre-merge complete (skip_merge=True): {len(result)} tensors "
            f"({unique_bytes / 1e9:.2f}GB unique) in {elapsed:.3f}s"
        )
        return result

    # Legacy path: Actually merge tensors (creates memory copies!)
    # Only used if skip_merge=False is explicitly passed
    logger.warning("Using legacy torch.cat merge path - this creates large memory copies!")

    # Start with transformed tensors
    result = dict(pinned_tensors)

    # Identify which tensors will be merged
    merged_components = set()  # Original tensor names that get merged
    merged_tensors = {}  # New merged tensor name -> tensor

    # Group tensors by layer path for merging
    layer_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in pinned_tensors.items():
        for merged_suffix, components, _ in MERGE_PATTERNS:
            for comp in components:
                if f".{comp}." in name:
                    idx = name.find(f".{comp}.")
                    layer_path = name[:idx + 1]
                    suffix = name[idx + len(comp) + 2:]
                    key = (layer_path, suffix, merged_suffix)
                    if key not in layer_tensors:
                        layer_tensors[key] = {}
                    layer_tensors[key][comp] = (name, tensor)
                    break

    # Create merged tensors
    for (layer_path, suffix, merged_suffix), component_dict in layer_tensors.items():
        for ms, components, concat_dim in MERGE_PATTERNS:
            if ms != merged_suffix:
                continue

            if all(c in component_dict for c in components):
                component_names = []
                component_tensors = []
                for c in components:
                    orig_name, tensor = component_dict[c]
                    component_names.append(orig_name)
                    component_tensors.append(tensor)

                # Verify shapes are compatible
                ref_shape = component_tensors[0].shape
                compatible = True
                for t in component_tensors[1:]:
                    for i, (d1, d2) in enumerate(zip(ref_shape, t.shape)):
                        if i != concat_dim and d1 != d2:
                            compatible = False
                            break

                if not compatible:
                    continue

                try:
                    merged = torch.cat(component_tensors, dim=concat_dim)

                    # Pin the merged tensor for fast GPU transfer
                    if pin_merged and not merged.is_pinned():
                        merged = merged.pin_memory()

                    # Generate merged name
                    merged_name = f"{layer_path}{merged_suffix}.{suffix}"
                    normalized_name = _normalize_for_vllm(merged_name)

                    merged_tensors[merged_name] = merged
                    if normalized_name != merged_name:
                        merged_tensors[normalized_name] = merged

                    # Track original components for removal
                    merged_components.update(component_names)
                except Exception as e:
                    logger.warning(f"Failed to merge {layer_path}{merged_suffix}: {e}")
                break

    # Add merged tensors to result
    result.update(merged_tensors)

    # Remove merged components if requested
    if remove_merged_components:
        for comp_name in merged_components:
            result.pop(comp_name, None)
            # Also remove normalized version
            norm_name = _normalize_for_vllm(comp_name)
            result.pop(norm_name, None)

    # Add normalized versions for remaining tensors
    for name, tensor in list(result.items()):
        norm_name = _normalize_for_vllm(name)
        if norm_name not in result:
            result[norm_name] = tensor

    # For Mistral format models, add language_model. prefixed versions
    # This handles Pixtral which wraps the LLM in language_model attribute
    if model_format == "mistral":
        logger.info("Adding Pixtral-compatible language_model. prefixed versions...")
        for name, tensor in list(result.items()):
            prefixed_name = _add_pixtral_prefix(name)
            if prefixed_name and prefixed_name not in result:
                result[prefixed_name] = tensor

    elapsed = time.perf_counter() - t0
    final_bytes = sum(t.numel() * t.element_size() for t in result.values())
    logger.info(
        f"Pre-merge complete: {len(result)} tensors ({final_bytes / 1e9:.2f}GB) "
        f"in {elapsed:.3f}s"
    )

    return result
