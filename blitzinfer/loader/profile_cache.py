"""BlitzInfer profile cache.

Persists ``num_gpu_blocks`` (vLLM's KV cache budget, the result of the
slow ~4 s profiling forward pass that runs on every cold load) keyed by
every input that could affect it. Survives gateway restarts and host
reboots.

Cache key includes:
  * model repo + (resolved) revision hash
  * max_model_len, gpu_memory_utilization, max_num_seqs
  * dtype / quantization
  * vLLM version, NVIDIA driver version, GPU model
  * any custom CLI args that change the profile (rope_scaling etc.)

Any change → cache miss → recompute → write new entry. Stale entries
stay on disk forever (small JSON files, a few KB each); we never blow
the cache away unless asked.

Cache file path: ``~/.cache/blitzinfer/profile/<sha16>.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


CACHE_DIR = Path(os.environ.get(
    "BLITZ_PROFILE_CACHE_DIR",
    str(Path.home() / ".cache" / "blitzinfer" / "profile"),
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


def _driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            timeout=2,
        )
        return out.decode().strip().split("\n")[0]
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=2,
        )
        return out.decode().strip().split("\n")[0]
    except Exception:
        return "unknown"


def _key_for(
    model_repo: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
    extra_args: tuple,
) -> tuple[str, dict]:
    """Return (sha16, key_dict) for a given config snapshot."""
    parts = {
        "model_repo": model_repo,
        "max_model_len": int(max_model_len),
        "gpu_util": round(float(gpu_util), 4),
        "max_num_seqs": int(max_num_seqs),
        "extra_args": list(extra_args),
        "vllm_version": _vllm_version(),
        "driver_version": _driver_version(),
        "gpu_name": _gpu_name(),
    }
    s = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16], parts


def get(
    model_repo: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
    extra_args: tuple,
) -> Optional[dict]:
    """Return cached profile dict or None if not present."""
    sha, _ = _key_for(model_repo, max_model_len, gpu_util, max_num_seqs, extra_args)
    path = CACHE_DIR / f"{sha}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "num_gpu_blocks" not in data:
            return None
        return data
    except Exception:
        logger.warning("profile_cache: failed to read %s", path)
        return None


def put(
    model_repo: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
    extra_args: tuple,
    num_gpu_blocks: int,
    extra: Optional[dict] = None,
) -> None:
    """Persist a fresh profile result. Atomic write via .tmp + rename."""
    sha, key_parts = _key_for(model_repo, max_model_len, gpu_util, max_num_seqs, extra_args)
    path = CACHE_DIR / f"{sha}.json"
    data = {
        "num_gpu_blocks": int(num_gpu_blocks),
        "captured_at": time.time(),
        "key": key_parts,
    }
    if extra:
        data.update(extra)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.rename(path)
    logger.info(
        "profile_cache: saved num_gpu_blocks=%d for %s (sha=%s)",
        num_gpu_blocks, model_repo, sha,
    )
