"""Memory management module with pinned memory arena and prefetching."""

from .arena import PinnedMemoryArena, ModelAllocation, TensorMeta
from .prefetcher import ModelPrefetcher, PrefetchStatus
from .cache_warmer import PageCacheWarmer, WarmStatus
from .fast_loader import (
    load_model_to_arena,
    get_model_size,
    get_safetensor_files,
    parse_safetensor_header,
)
from .premerge import (
    premerge_vllm_weights,
    get_premerged_tensors_for_vllm,
)

# Lazy imports for vLLM-dependent modules (vLLM may not be installed)
def __getattr__(name):
    if name in ("PinnedArenaModelLoader", "set_preloaded_weights",
                "get_preloaded_weights", "clear_preloaded_weights",
                "transfer_arena_to_gpu"):
        from .pinned_loader import (
            PinnedArenaModelLoader,
            set_preloaded_weights,
            get_preloaded_weights,
            clear_preloaded_weights,
            transfer_arena_to_gpu,
        )
        globals()["PinnedArenaModelLoader"] = PinnedArenaModelLoader
        globals()["set_preloaded_weights"] = set_preloaded_weights
        globals()["get_preloaded_weights"] = get_preloaded_weights
        globals()["clear_preloaded_weights"] = clear_preloaded_weights
        globals()["transfer_arena_to_gpu"] = transfer_arena_to_gpu
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Arena-based prefetching (for pinned memory approach)
    "PinnedMemoryArena",
    "ModelAllocation",
    "TensorMeta",
    "ModelPrefetcher",
    "PrefetchStatus",
    # Page cache warming (simpler approach)
    "PageCacheWarmer",
    "WarmStatus",
    # Fast pinned loader for vLLM (lazy loaded)
    "PinnedArenaModelLoader",
    "set_preloaded_weights",
    "get_preloaded_weights",
    "clear_preloaded_weights",
    "transfer_arena_to_gpu",
    # Utilities
    "load_model_to_arena",
    "get_model_size",
    "get_safetensor_files",
    "parse_safetensor_header",
    # Pre-merge for vLLM
    "premerge_vllm_weights",
    "get_premerged_tensors_for_vllm",
]
