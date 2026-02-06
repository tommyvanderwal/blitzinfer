"""Orchestration layer for model management."""

from .model_state import ModelState, ModelStatus, ModelRegistry
from .standby_manager import StandbyManager, StandbyState, StandbySlot

# Lazy import for controller (depends on vllm which may not be installed)
# Import explicitly with: from blitzinfer.orchestrator.controller import BlitzInferOrchestrator
def __getattr__(name):
    if name in ("BlitzInferOrchestrator", "SwitchMetrics"):
        from .controller import BlitzInferOrchestrator, SwitchMetrics
        globals()["BlitzInferOrchestrator"] = BlitzInferOrchestrator
        globals()["SwitchMetrics"] = SwitchMetrics
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModelState",
    "ModelStatus",
    "ModelRegistry",
    "BlitzInferOrchestrator",
    "SwitchMetrics",
    "StandbyManager",
    "StandbyState",
    "StandbySlot",
]
