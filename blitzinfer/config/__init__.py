"""Configuration module."""

from .settings import (
    ModelConfig,
    BlitzInferConfig,
    PrefetchConfig,
    IGPU_DEFAULT_CONFIG,
)

__all__ = ["ModelConfig", "BlitzInferConfig", "PrefetchConfig", "IGPU_DEFAULT_CONFIG"]
