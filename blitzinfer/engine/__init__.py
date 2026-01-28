"""Engine adapters for LLM inference."""

from .vllm_adapter import VLLMEngine, GenerationResult
from .cleanup import (
    cleanup_vllm_model,
    clear_rope_cache,
    clear_vllm_caches,
    clear_current_vllm_config,  # CRITICAL: Reset vLLM config between model loads
    destroy_parallel_state,
    full_cleanup,
    get_gpu_memory_info,
    log_gpu_memory,
    # Cleanup functions for minimizing residual drift
    clear_triton_caches,
    clear_cudnn_cublas_workspaces,
    clear_dynamo_inductor,
    release_nccl_resources,
    clear_flash_attention_cache,
    clear_marlin_workspace,
    aggressive_allocator_cleanup,
    force_free_phantom_blocks,  # Key function for freeing orphaned CUDA allocations
    nuclear_cleanup,
)

__all__ = [
    "VLLMEngine",
    "GenerationResult",
    "cleanup_vllm_model",
    "clear_rope_cache",
    "clear_vllm_caches",
    "clear_current_vllm_config",
    "destroy_parallel_state",
    "full_cleanup",
    "get_gpu_memory_info",
    "log_gpu_memory",
    # Cleanup functions
    "clear_triton_caches",
    "clear_cudnn_cublas_workspaces",
    "clear_dynamo_inductor",
    "release_nccl_resources",
    "clear_flash_attention_cache",
    "clear_marlin_workspace",
    "aggressive_allocator_cleanup",
    "force_free_phantom_blocks",
    "nuclear_cleanup",
]
