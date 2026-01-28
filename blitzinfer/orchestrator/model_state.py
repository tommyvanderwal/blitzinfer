"""Model state machine for BlitzInfer."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import time


class ModelState(Enum):
    """State of a model in the BlitzInfer system."""
    COLD = auto()      # Model on disk, not loaded
    WARM = auto()      # Model in DDR5 RAM (future: for discrete GPU)
    HOT = auto()       # Model loaded in GPU VRAM
    SERVING = auto()   # Model actively serving requests
    SWITCHING = auto() # Model being switched out


@dataclass
class ModelStatus:
    """Complete status of a model including queue info."""
    name: str
    state: ModelState = ModelState.COLD
    queue_depth: int = 0
    active_requests: int = 0
    last_served: Optional[float] = None  # timestamp
    load_time: Optional[float] = None    # seconds to load
    total_requests_served: int = 0

    def mark_serving(self):
        """Mark model as actively serving."""
        self.state = ModelState.SERVING
        self.last_served = time.time()

    def mark_idle(self):
        """Mark model as loaded but idle."""
        if self.state == ModelState.SERVING:
            self.state = ModelState.HOT

    def increment_queue(self):
        self.queue_depth += 1

    def decrement_queue(self):
        self.queue_depth = max(0, self.queue_depth - 1)

    def request_started(self):
        self.active_requests += 1
        self.decrement_queue()

    def request_completed(self):
        self.active_requests = max(0, self.active_requests - 1)
        self.total_requests_served += 1
        self.last_served = time.time()

    @property
    def is_loaded(self) -> bool:
        return self.state in (ModelState.HOT, ModelState.SERVING)

    @property
    def is_busy(self) -> bool:
        return self.active_requests > 0

    @property
    def has_pending(self) -> bool:
        return self.queue_depth > 0


class ModelRegistry:
    """Registry of all models and their states."""

    def __init__(self):
        self._models: dict[str, ModelStatus] = {}
        self._active_model: Optional[str] = None

    def register(self, name: str) -> ModelStatus:
        """Register a new model."""
        if name not in self._models:
            self._models[name] = ModelStatus(name=name)
        return self._models[name]

    def get(self, name: str) -> Optional[ModelStatus]:
        """Get model status by name."""
        return self._models.get(name)

    def get_or_register(self, name: str) -> ModelStatus:
        """Get model status, registering if needed."""
        return self.register(name)

    @property
    def active_model(self) -> Optional[str]:
        """Get the currently active model name."""
        return self._active_model

    @active_model.setter
    def active_model(self, name: Optional[str]):
        """Set the active model."""
        # Mark old model as HOT (not serving)
        if self._active_model and self._active_model in self._models:
            self._models[self._active_model].state = ModelState.HOT
        # Mark new model as SERVING
        if name and name in self._models:
            self._models[name].state = ModelState.SERVING
        self._active_model = name

    @property
    def all_models(self) -> list[ModelStatus]:
        """Get all registered models."""
        return list(self._models.values())

    def get_next_model_to_serve(self) -> Optional[str]:
        """Get the model with pending requests that should be served next.

        Strategy: Simple FCFS - pick model with longest queue that isn't active.
        """
        candidates = [
            m for m in self._models.values()
            if m.name != self._active_model and m.has_pending
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.queue_depth).name
