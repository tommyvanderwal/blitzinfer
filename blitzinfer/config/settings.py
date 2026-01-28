"""Configuration settings for BlitzInfer."""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class PrefetchConfig:
    """Configuration for model prefetching system.

    Two approaches are supported:
    1. Page cache warming (default): Pre-read files into OS page cache.
       Simpler, no pinned memory allocation needed. ~12 GB/s warming speed.
    2. Pinned arena: Allocate large pinned memory block, load tensors.
       Faster GPU transfer (~45 GB/s) but requires dedicated RAM allocation.
    """
    enabled: bool = True
    use_page_cache: bool = True  # True = page cache warming, False = pinned arena
    arena_size_gb: float = 80.0  # Only used if use_page_cache=False
    trigger_on_queue_entry: bool = True  # Start prefetch when request enters queue
    evict_lru_on_full: bool = True  # Evict least recently used when arena is full
    prefetch_ahead: int = 1  # Number of models to prefetch ahead
    max_concurrent_loads: int = 1  # Max concurrent background loads
    chunk_size_mb: int = 64  # Read chunk size for page cache warming


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str  # HuggingFace model name
    alias: Optional[str] = None  # Short alias for API
    path: Optional[str] = None  # Local path to model (optional, uses HF cache if not set)
    dtype: str = "float16"  # float16 more stable on AMD
    max_model_len: int = 512  # Small for fast switching
    gpu_memory_utilization: float = 0.25
    max_num_seqs: int = 20
    max_num_batched_tokens: int = 512
    enforce_eager: bool = True  # Required on gfx1103 (AMD 780M)
    kv_cache_memory_bytes: Optional[int] = 2 * 1024**3  # 2GB fixed - skip memory profiling
    compilation_config: Optional[dict[str, Any]] = None  # Will be set in __post_init__
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.compilation_config is None:
            self.compilation_config = {"custom_ops": ["none"]}

    @property
    def display_name(self) -> str:
        return self.alias or self.name


@dataclass
class BlitzInferConfig:
    """Main configuration for BlitzInfer."""

    # Model configurations
    models: list[ModelConfig] = field(default_factory=list)

    # Memory settings (for future discrete GPU support)
    warm_tier_budget_gb: float = 0.0  # DDR5 budget for WARM tier (0 = disabled)

    # Standby settings (1 active + 1 standby model switching)
    standby_enabled: bool = True  # Enable standby slot for fast switching
    standby_arena_gb: float = 70.0  # Size of standby arena (must fit largest model)

    # Prefetch settings (legacy, for page cache / arena prefetch)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)

    # Engine settings
    device: str = "auto"  # "cuda", "rocm", "auto"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Switch settings
    drain_timeout_seconds: float = 300.0  # Max time to wait for requests to drain


# Default configuration for iGPU (Radeon 780M)
IGPU_DEFAULT_CONFIG = BlitzInferConfig(
    models=[],
    warm_tier_budget_gb=0.0,  # Unified memory - no separate WARM tier needed
    device="rocm",
)
