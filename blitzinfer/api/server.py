"""BlitzInfer rotation gateway — Phase H.

Single FastAPI process on :8000. ModelManager owns one AsyncLLM at a time;
on a request for a different model it drains in-flight, tears down the
EngineCore subprocess, and spawns a new one for the requested model.

Memory model: each AsyncLLM owns an EngineCore subprocess. Swap = subprocess
death + respawn → OS reclaims VRAM. Drift across swaps is ~0 MiB.

Concurrency:
  - One asyncio.Condition guards (frozen, active_name, in_flight).
  - acquire(model) is the only entry point; on swap it sets frozen=True,
    drains in_flight to 0, then does the slow IO with the lock released.
  - release() decrements in_flight and notifies; the swapper resumes
    when in_flight reaches 0.

Streaming: chat_completions wraps the SSE async-generator so release()
fires only when the stream completes or the client disconnects.

Run:
  python -m blitzinfer.api.server
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

from blitzinfer.loader.shared_pool import SharedPool, DEFAULT_POOL_PATH
from blitzinfer.loader import profile_cache


logger = logging.getLogger("blitzinfer.server")


# ============================================================================
# vLLM patches applied at module import
# ============================================================================

# MM warmup explicitly clears its cache after running (vllm/renderers/base.py
# `_warmup_mm_processor` → `clear_mm_cache`), so the only retained state is
# Python-module-level lazy imports — which would happen on the first real MM
# request anyway. The 5-12 s of dummy preprocessing per cold load is pure
# throwaway work. Skip it; first MM request pays ~1-2 s for the lazy imports.
try:
    from vllm.renderers.base import BaseRenderer

    def _blitz_skip_mm_warmup(self, processor, *, log_prefix):
        return

    BaseRenderer._warmup_mm_processor = _blitz_skip_mm_warmup
    logger.info("blitz: patched out BaseRenderer._warmup_mm_processor")
except Exception:
    logger.exception("blitz: failed to patch out MM warmup; falling through")


# ============================================================================
# Model registry
# ============================================================================


@dataclass(frozen=True)
class ModelConfig:
    served_name: str
    repo: str
    max_model_len: int
    gpu_util: float = 0.85
    max_num_seqs: int = 32
    tool_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    load_format: str = "auto"
    # Routes through blitzinfer/loader/pinned_loader.py (v3) which mmaps an
    # 80 GB region backed by 1 GB hugepages, registers it once with CUDA,
    # then streams shards into the registered region. Subsequent .to('cuda')
    # is at PCIe wire-speed (~44 GB/s pinned DMA) instead of the ~5.8 GB/s
    # pageable bounce. Requires kernel cmdline: hugepagesz=1G hugepages=80.
    load_threads: int = 8
    env: dict = field(default_factory=dict)
    extra_args: list = field(default_factory=list)


REGISTRY: dict[str, ModelConfig] = {
    "qwen3.5-122b-a10b": ModelConfig(
        # 122B total / 10B active MoE multimodal, 75 GB on disk in NVFP4.
        # Per RedHatAI HF page: --reasoning-parser qwen3 --tool-call-parser
        # qwen3_coder --moe-backend flashinfer_cutlass.
        served_name="qwen3.5-122b-a10b",
        repo="RedHatAI/Qwen3.5-122B-A10B-NVFP4",
        max_model_len=262144,
        gpu_util=0.93,  # 75 GB weights leave ~17 GiB for KV at 256K
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
        extra_args=["--moe-backend", "flashinfer_cutlass"],
    ),
    "qwen3.6-35b-a3b": ModelConfig(
        served_name="qwen3.6-35b-a3b",
        repo="Qwen/Qwen3.6-35B-A3B-FP8",
        max_model_len=262144,
        tool_parser="qwen3_xml",
        reasoning_parser="qwen3",
    ),
    "qwen3.6-27b": ModelConfig(
        served_name="qwen3.6-27b",
        repo="Qwen/Qwen3.6-27B-FP8",
        max_model_len=262144,
        tool_parser="qwen3_xml",
        reasoning_parser="qwen3",
    ),
    "gemma-4-31b": ModelConfig(
        served_name="gemma-4-31b",
        repo="google/gemma-4-31b-it",
        max_model_len=262144,
        gpu_util=0.93,  # need extra room for 27 GiB KV at 256K
        tool_parser="gemma4",
        reasoning_parser="gemma4",
    ),
    "gpt-oss-120b": ModelConfig(
        served_name="gpt-oss-120b",
        repo="openai/gpt-oss-120b",
        max_model_len=131072,
        tool_parser="openai",
        reasoning_parser="openai_gptoss",
        env={"VLLM_MXFP4_USE_MARLIN": "1"},
    ),
    "qwen3-coder-next": ModelConfig(
        served_name="qwen3-coder-next",
        repo="Qwen/Qwen3-Coder-Next-FP8",
        max_model_len=262144,
        gpu_util=0.93,  # 75GB weights at 256K need every spare GiB
        tool_parser="qwen3_coder",
    ),
    "qwen3-32b": ModelConfig(
        served_name="qwen3-32b",
        repo="Qwen/Qwen3-32B-FP8",
        max_model_len=131072,
        tool_parser="qwen3_xml",
        reasoning_parser="qwen3",
        extra_args=[
            "--hf-overrides",
            '{"rope_scaling":{"rope_type":"yarn","factor":3.2,"original_max_position_embeddings":40960}}',
        ],
    ),
    "qwen2.5-7b": ModelConfig(
        served_name="qwen2.5-7b",
        repo="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=131072,
        tool_parser="hermes",
        extra_args=[
            "--hf-overrides",
            '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}',
        ],
    ),
    "kimi-vl": ModelConfig(
        served_name="kimi-vl",
        repo="moonshotai/Kimi-VL-A3B-Instruct",
        max_model_len=131072,
        extra_args=["--trust-remote-code"],
    ),
}


def args_for(cfg: ModelConfig, extra: list[str] | None = None) -> argparse.Namespace:
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    cli = [
        "--model", cfg.repo,
        "--served-model-name", cfg.served_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", str(cfg.gpu_util),
        "--max-model-len", str(cfg.max_model_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--load-format", cfg.load_format,
    ]
    if cfg.load_threads > 0:
        cli += [
            "--model-loader-extra-config",
            f'{{"enable_multithread_load":true,"num_threads":{cfg.load_threads}}}',
        ]
    if cfg.tool_parser:
        cli += ["--enable-auto-tool-choice", "--tool-call-parser", cfg.tool_parser]
    if cfg.reasoning_parser:
        cli += ["--reasoning-parser", cfg.reasoning_parser]
    cli += list(cfg.extra_args)
    if extra:
        cli += list(extra)
    return parser.parse_args(cli)


# ============================================================================
# ModelManager
# ============================================================================


def _resolve_shard_paths(model_repo: str) -> list[str]:
    """Find safetensors shard files for a model in the HF cache.

    Returns paths in canonical (sorted) order. Empty list if not found.
    """
    import glob as _glob

    org, _, name = model_repo.partition("/")
    cache_root = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{org}--{name}/snapshots"
    )
    snapshots = sorted(_glob.glob(os.path.join(cache_root, "*")))
    if not snapshots:
        return []
    snap = snapshots[-1]  # most recent revision
    shards = sorted(_glob.glob(os.path.join(snap, "*.safetensors")))
    # Skip files inside an "original/" subdir (gpt-oss-120b ships dual layouts)
    shards = [s for s in shards if "/original/" not in s]
    return shards


@dataclass
class _PendingRequest:
    """One entry in a per-model FIFO queue. enqueued_at picks the next
    model on swap-decision: lowest enqueued_at wins."""
    model: str
    enqueued_at: float
    fut: asyncio.Future


class ModelManager:
    """Gateway state machine.

    Two key invariants:
      * every request goes through a per-model FIFO queue; nothing fast-paths
        around it. The queue is the source of truth for what's pending.
      * a single dispatcher coroutine owns all queue → engine handoff. It
        (a) drains queues[active_name] onto the live engine, then (b) when
        the active queue is empty, picks the longest-waiting other-model
        request and triggers a pipelined swap.

    Pipelined swap: as soon as the dispatcher commits to the next model,
    it kicks off subprocess spawn (vLLM imports → barrier wait) and parent
    pool load IN PARALLEL with the active engine continuing to serve. The
    dispatcher keeps draining queues[active_name] for arrivals during the
    spawn. Drain → unload → barrier release happens once the active queue
    is empty AND in_flight == 0.
    """

    def __init__(
        self,
        app: FastAPI,
        registry: dict[str, ModelConfig],
        shared_pool: Optional[SharedPool] = None,
    ):
        self.app = app
        self.registry = registry
        self.shared_pool = shared_pool
        self._lock = asyncio.Lock()
        self._cv = asyncio.Condition(self._lock)
        self._queues: dict[str, collections.deque[_PendingRequest]] = {
            name: collections.deque() for name in registry
        }
        self.engine: Optional[AsyncLLM] = None
        self.active_name: Optional[str] = None
        self.in_flight = 0
        self._swapping = False
        self._swap_target: Optional[str] = None
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._stop = False
        # stats
        self.swap_count = 0
        self.last_swap_seconds = 0.0
        self.last_swap_target: Optional[str] = None
        self.last_pool_load_seconds = 0.0
        self.total_requests = 0

    def start(self) -> None:
        """Launch the dispatcher coroutine. Call once at startup."""
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatcher())

    async def stop(self) -> None:
        """Stop the dispatcher and reject any still-queued requests."""
        self._stop = True
        async with self._cv:
            for q in self._queues.values():
                while q:
                    pending = q.popleft()
                    if not pending.fut.done():
                        pending.fut.set_exception(
                            HTTPException(503, "gateway shutting down")
                        )
            self._cv.notify_all()
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass

    async def acquire(self, model_name: str):
        """Append to queues[model_name] and wait for the dispatcher to hand
        back the request handler. Every request goes through the queue —
        there is no fast path. This is what enables the 'drain all of one
        model before switching' policy: same-model requests that arrive
        after a cross-model request still queue on the active model and get
        served before the swap actually happens."""
        if model_name not in self._queues:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not in registry"
            )
        loop = asyncio.get_event_loop()
        pending = _PendingRequest(
            model=model_name,
            enqueued_at=loop.time(),
            fut=loop.create_future(),
        )
        async with self._cv:
            self._queues[model_name].append(pending)
            self.total_requests += 1
            self._cv.notify_all()
        return await pending.fut

    async def release(self):
        async with self._cv:
            self.in_flight -= 1
            if self.in_flight < 0:
                self.in_flight = 0
            self._cv.notify_all()

    def _pick_next_model_locked(self) -> Optional[str]:
        """Choose which model to load next: among all queues OTHER than
        the active one, pick the one whose oldest entry has been waiting
        the longest. Ties broken by name (deterministic). Caller must
        hold the lock."""
        candidates: list[tuple[float, str]] = []
        for name, q in self._queues.items():
            if name == self.active_name:
                continue
            if q:
                candidates.append((q[0].enqueued_at, name))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _dispatcher(self):
        """Single dispatcher coroutine. Wakes whenever state changes
        (request enqueued, request finished, swap finished, engine ready),
        then loops until there's nothing to do.

        Per iteration:
          1. Drain queues[active_name] onto the live engine — every pending
             same-model request gets in_flight++ and the handler future
             resolved. This is concurrent: vLLM batches them internally.
          2. Reap the swap_task if it just finished.
          3. If no swap is in progress AND some other queue has work,
             commit to that model (oldest queued wins) and start a
             pipelined swap task. The swap task drains in parallel — the
             dispatcher stays responsive to new arrivals on the active
             queue.
          4. cv.wait() until something changes.
        """
        swap_task: Optional[asyncio.Task] = None
        while not self._stop:
            async with self._cv:
                # Phase 1: serve every pending request for the active model.
                while (
                    self.active_name is not None
                    and self.engine is not None
                    and self._queues[self.active_name]
                ):
                    pending = self._queues[self.active_name].popleft()
                    self.in_flight += 1
                    if not pending.fut.done():
                        pending.fut.set_result(
                            self.app.state.openai_serving_chat
                        )

                # Phase 2: reap finished swap.
                if swap_task is not None and swap_task.done():
                    try:
                        swap_task.result()
                    except Exception:
                        logger.exception("pipelined swap task failed")
                    swap_task = None

                # Phase 3: pick next model and start swap if needed.
                if swap_task is None:
                    next_model = self._pick_next_model_locked()
                    if next_model is not None:
                        self._swapping = True
                        self._swap_target = next_model
                        swap_task = asyncio.create_task(
                            self._do_pipelined_swap(next_model)
                        )

                # Phase 4: wait for state change.
                await self._cv.wait()

    async def _do_pipelined_swap(self, name: str):
        """Pipelined swap to model ``name``. spawn + pool start IMMEDIATELY;
        drain → unload → barrier release wait for the active queue to
        empty and in_flight to hit 0. Engine install at the end notifies
        the dispatcher to drain queues[name]."""
        try:
            if self.shared_pool is not None:
                # BLITZ_SHARDS_LAYOUT MUST be set BEFORE spawn_task —
                # the subprocess inherits env at fork() time.
                layout_path = (
                    f"/tmp/blitz_shards_{os.getpid()}_{name}.json"
                )
                go_file = (
                    f"/tmp/blitz_gpu_go_{os.getpid()}_"
                    f"{name}_{time.time_ns()}"
                )
                try:
                    if os.path.exists(go_file):
                        os.unlink(go_file)
                except OSError:
                    pass
                os.environ["BLITZ_SHARED_POOL"] = self.shared_pool.path
                os.environ["BLITZ_POOL_GB"] = str(
                    int(self.shared_pool.size / 1024**3)
                )
                os.environ["BLITZ_SHARDS_LAYOUT"] = layout_path
                os.environ["BLITZ_GPU_GO_FILE"] = go_file

                # Stage A: kick off pool + spawn while old engine still
                # serves drained items from queues[active_name].
                pool_task = asyncio.create_task(
                    self._populate_pool(name, layout_path)
                )
                spawn_task = asyncio.create_task(self._spawn_engine(name))

                # Stage B: wait for active queue + in_flight to fully drain.
                # New arrivals on the active queue ARE served first (the
                # dispatcher keeps dispatching them while we're waiting).
                async with self._cv:
                    while (
                        (
                            self.active_name is not None
                            and self._queues[self.active_name]
                        )
                        or self.in_flight > 0
                    ):
                        await self._cv.wait()

                # Stage C: tear down old engine; awaits nvidia-smi VRAM ≤
                # 2 GiB before returning.
                if self.engine is not None:
                    await self._unload()

                # Pool MUST be on disk before subprocess crosses barrier.
                await pool_task

                # Stage D: GPU is idle + pool ready → release subprocess.
                try:
                    open(go_file, "w").close()
                except OSError:
                    logger.exception(
                        "failed to touch GPU go-file %s", go_file
                    )
                # Stage E: subprocess does set_device → weights → KV cache
                # → init_app_state installs the new openai_serving_chat.
                await spawn_task
                try:
                    os.unlink(go_file)
                except OSError:
                    pass
                os.environ.pop("BLITZ_GPU_GO_FILE", None)
            else:
                # No shared pool: drain → unload → sequential load.
                async with self._cv:
                    while (
                        (
                            self.active_name is not None
                            and self._queues[self.active_name]
                        )
                        or self.in_flight > 0
                    ):
                        await self._cv.wait()
                if self.engine is not None:
                    await self._unload()
                await self._load(name)
        except Exception:
            logger.exception("pipelined swap to %s failed", name)
            # Reject any queued futures for this model so callers see 503.
            async with self._cv:
                for pending in list(self._queues.get(name, [])):
                    if not pending.fut.done():
                        pending.fut.set_exception(
                            HTTPException(
                                status_code=503,
                                detail=f"failed to load model {name!r}",
                            )
                        )
                self._queues[name].clear()
        finally:
            async with self._cv:
                self._swapping = False
                self._swap_target = None
                self._cv.notify_all()

    async def _populate_pool(self, name: str, layout_path: str):
        """Parent-side: read shards into the shared hugetlb pool and write
        the layout JSON atomically at ``layout_path``.

        Env vars (BLITZ_SHARED_POOL, BLITZ_SHARDS_LAYOUT, BLITZ_POOL_GB) are
        set by the caller in acquire() BEFORE _spawn_engine creates the
        subprocess — setting them here would race with the spawn fork().

        Skips the disk read if the pool already holds this model (e.g., the
        same model is being re-loaded), but still rewrites the layout file
        so the subprocess sees a fresh, complete file on disk.
        """
        cfg = self.registry[name]
        if self.shared_pool is None:
            return
        shards = _resolve_shard_paths(cfg.repo)
        if not shards:
            logger.warning(
                "shared pool: no shards for %s; subprocess will use private pool",
                name,
            )
            return
        pool_t0 = time.perf_counter()
        layout = await asyncio.get_event_loop().run_in_executor(
            None, self.shared_pool.load_shards, name, shards,
        )
        self.last_pool_load_seconds = time.perf_counter() - pool_t0
        # Atomic write so the subprocess never reads a half-written layout
        # if it polls the file early.
        tmp_path = layout_path + ".tmp"
        self.shared_pool.write_layout_json(layout, tmp_path)
        os.replace(tmp_path, layout_path)
        logger.info(
            "shared pool: %s into pool in %.1fs (%d shards) -> %s",
            name, self.last_pool_load_seconds, len(shards),
            os.path.basename(layout_path),
        )

    async def _spawn_engine(self, name: str):
        """Spawn vLLM EngineCore subprocess. Uses cached num_gpu_blocks if
        available to skip vLLM's profiling forward pass (~4 s saved)."""
        t0 = time.perf_counter()
        cfg = self.registry[name]
        for k, v in cfg.env.items():
            os.environ[k] = v

        # Profile cache lookup: if present, skip vLLM's slow profile pass
        # by passing --num-gpu-blocks-override. Cache key includes vLLM /
        # driver / GPU / model / context-length / mem-util — any change
        # forces a recompute.
        cached_extra: list[str] = []
        cached = profile_cache.get(
            cfg.repo, cfg.max_model_len, cfg.gpu_util,
            cfg.max_num_seqs, tuple(cfg.extra_args),
        )
        cache_hit = cached is not None
        if cache_hit:
            cached_extra = ["--num-gpu-blocks-override", str(cached["num_gpu_blocks"])]
            logger.info(
                "profile_cache HIT for %s: num_gpu_blocks=%d (saves ~4 s of profiling)",
                name, cached["num_gpu_blocks"],
            )

        args = args_for(cfg, extra=cached_extra)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        # AsyncLLM.from_vllm_config is SYNCHRONOUS and blocks for ~25 s
        # (subprocess spawn + handshake). Run in executor so the asyncio
        # event loop can interleave with the parallel _unload / _populate_pool
        # tasks. The subprocess will hit the BLITZ_GPU_GO_FILE barrier
        # patched into vllm/v1/worker/gpu_worker.py and wait until we
        # release it (after unload + pool both complete).
        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(
            None,
            functools.partial(
                AsyncLLM.from_vllm_config,
                vllm_config=vllm_config,
                usage_context=UsageContext.OPENAI_API_SERVER,
                enable_log_requests=engine_args.enable_log_requests,
                disable_log_stats=engine_args.disable_log_stats,
            ),
        )
        try:
            await engine.reset_mm_cache()
            await init_app_state(
                engine, self.app.state, args, supported_tasks=("generate",)
            )
        except Exception:
            logger.exception("post-spawn init failed for %s; tearing down", name)
            try:
                engine.shutdown()
            except Exception:
                logger.exception("engine.shutdown during recovery raised")
            await self._wait_for_gpu_baseline(baseline_mib=2000, timeout=30)
            raise
        self.engine = engine
        self.active_name = name
        self.swap_count += 1
        self.last_swap_target = name
        self.last_swap_seconds = time.perf_counter() - t0
        logger.info(
            "spawned engine for %s in %.1fs %s",
            name, self.last_swap_seconds,
            "(cache hit)" if cache_hit else "(cache miss — saving for next time)",
        )

        # On cache miss, capture num_gpu_blocks for future loads
        if not cache_hit:
            try:
                cc = getattr(engine.vllm_config, "cache_config", None)
                num_gpu_blocks = getattr(cc, "num_gpu_blocks", None) if cc else None
                logger.info(
                    "profile_cache capture probe: cache_config=%s num_gpu_blocks=%s",
                    cc, num_gpu_blocks,
                )
                if num_gpu_blocks:
                    profile_cache.put(
                        cfg.repo, cfg.max_model_len, cfg.gpu_util,
                        cfg.max_num_seqs, tuple(cfg.extra_args),
                        num_gpu_blocks,
                        extra={"block_size": getattr(cc, "block_size", None)},
                    )
                else:
                    # Try alternative access paths
                    for path in [
                        "engine.engine_core.engine_core.num_gpu_blocks",
                        "engine.engine_core.num_gpu_blocks",
                        "engine.num_gpu_blocks",
                    ]:
                        try:
                            obj = engine
                            for attr in path.split(".")[1:]:
                                obj = getattr(obj, attr)
                            logger.info("profile_cache: found at %s = %s", path, obj)
                            if obj:
                                profile_cache.put(
                                    cfg.repo, cfg.max_model_len, cfg.gpu_util,
                                    cfg.max_num_seqs, tuple(cfg.extra_args),
                                    int(obj),
                                )
                                break
                        except Exception as e:
                            logger.info("profile_cache: %s not accessible: %s", path, e)
            except Exception:
                logger.exception("profile_cache: failed to capture num_gpu_blocks")

    async def _load(self, name: str):
        """Sequential path: populate pool, then spawn. Used when there is no
        existing engine to tear down (no overlap opportunity)."""
        if self.shared_pool is not None:
            layout_path = f"/tmp/blitz_shards_{os.getpid()}_{name}.json"
            os.environ["BLITZ_SHARED_POOL"] = self.shared_pool.path
            os.environ["BLITZ_POOL_GB"] = str(
                int(self.shared_pool.size / 1024**3)
            )
            os.environ["BLITZ_SHARDS_LAYOUT"] = layout_path
            await self._populate_pool(name, layout_path)
        await self._spawn_engine(name)

    async def _wait_for_gpu_baseline(
        self, baseline_mib: int = 2000, timeout: float = 30.0,
    ) -> int:
        """Poll nvidia-smi until GPU memory.used drops below baseline_mib.
        Replaces the old fixed 2.5 s sleep — the real time depends on how
        long the EngineCore subprocess takes to die + driver to reclaim
        VRAM, which varies by model size and pinned-host registration size.
        """
        import subprocess
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=2,
                )
                used = int(out.decode().strip())
            except Exception:
                used = 999_999
            if used < baseline_mib:
                return used
            await asyncio.sleep(0.4)
        return used

    async def _unload(self):
        if self.engine is None:
            return
        prev = self.active_name
        t0 = time.perf_counter()
        try:
            self.engine.shutdown()
        except Exception as e:
            logger.warning("engine.shutdown raised: %r", e)
        self.engine = None
        self.active_name = None
        for attr in (
            "openai_serving_chat",
            "openai_serving_models",
            "openai_serving_render",
            "openai_serving_tokenization",
            "openai_serving_completion",
            "openai_serving_responses",
            "openai_serving_transcription",
            "engine_client",
        ):
            if hasattr(self.app.state, attr):
                delattr(self.app.state, attr)
        # Wait for EngineCore subprocess to actually release VRAM. Without
        # this, the next subprocess sees ~12 GiB free and refuses to start
        # with: "Free memory ... less than desired GPU memory utilization".
        # baseline ~2 GiB covers the parent's CUDA context + Xorg.
        used = await self._wait_for_gpu_baseline(baseline_mib=2000, timeout=30)
        logger.info(
            "unloaded %s in %.1fs; GPU now %d MiB used",
            prev, time.perf_counter() - t0, used,
        )


# ============================================================================
# FastAPI routes
# ============================================================================


def attach_routes(app: FastAPI, manager: ModelManager) -> None:
    @app.get("/health")
    async def health():
        return {"status": "ok", "active_model": manager.active_name}

    @app.get("/v1/models")
    async def list_models():
        data = []
        for name, cfg in manager.registry.items():
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "blitzinfer",
                    "root": cfg.repo,
                    "max_model_len": cfg.max_model_len,
                    "loaded": (manager.active_name == name),
                    "permission": [],
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/admin/status")
    async def admin_status():
        pool = manager.shared_pool
        return {
            "active_model": manager.active_name,
            "in_flight": manager.in_flight,
            "swapping": manager._swapping,
            "swap_target": manager._swap_target,
            "queues": {
                name: len(q) for name, q in manager._queues.items() if q
            },
            "swap_count": manager.swap_count,
            "last_swap_seconds": manager.last_swap_seconds,
            "last_swap_target": manager.last_swap_target,
            "last_pool_load_seconds": manager.last_pool_load_seconds,
            "total_requests": manager.total_requests,
            "registered": list(manager.registry.keys()),
            "shared_pool": (
                {
                    "path": pool.path,
                    "size_gb": pool.size / 1024**3,
                    "loaded_model": pool._loaded_model,
                }
                if pool is not None
                else None
            ),
        }

    @app.post("/v1/admin/load")
    async def admin_load(payload: dict):
        name = payload.get("model")
        if not name:
            raise HTTPException(status_code=400, detail="missing 'model'")
        # Goes through the queue → dispatcher → pipelined swap. Wait until
        # the engine is up and we've claimed an in-flight slot, then
        # immediately release.
        await manager.acquire(name)
        await manager.release()
        return {"loaded": name, "swap_seconds": manager.last_swap_seconds}

    @app.post("/v1/admin/unload")
    async def admin_unload():
        # Drain whatever's pending then unload. We grab a "synthetic"
        # queue slot by going through acquire() so the dispatcher serializes
        # with us; then we manually unload.
        async with manager._cv:
            while manager.in_flight > 0:
                await manager._cv.wait()
            if manager.engine is not None:
                # Hold the lock during _unload — short window where new
                # acquires queue but don't dispatch (engine is None
                # mid-shutdown).
                await manager._unload()
            manager._cv.notify_all()
        return {"unloaded": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest, raw_request: Request
    ):
        handler = await manager.acquire(request.model)
        released = False

        async def _release_once():
            nonlocal released
            if not released:
                released = True
                await manager.release()

        try:
            result = await handler.create_chat_completion(request, raw_request)
        except Exception:
            await _release_once()
            raise

        if isinstance(result, ErrorResponse):
            await _release_once()
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code,
            )
        if isinstance(result, ChatCompletionResponse):
            await _release_once()
            return JSONResponse(content=result.model_dump())

        async def wrapped():
            try:
                async for chunk in result:
                    yield chunk
            finally:
                await _release_once()

        return StreamingResponse(content=wrapped(), media_type="text/event-stream")


# ============================================================================
# Entry point
# ============================================================================


async def _shutdown_engine(manager: ModelManager) -> None:
    """Best-effort engine teardown on process exit.

    Without this, a SIGTERM/SIGINT to the gateway leaves the EngineCore
    subprocess orphaned, holding 60+ GiB of VRAM until manually killed.
    """
    if manager.engine is None:
        return
    try:
        manager.engine.shutdown()
    except Exception:
        logger.exception("engine.shutdown during gateway exit raised")
    # Give the EngineCore subprocess time to actually exit.
    await asyncio.sleep(2.5)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    app = FastAPI(title="BlitzInfer Gateway")
    # Open the shared hugetlbfs pool in the parent process. Subprocesses
    # re-mmap + re-register this same file in their own CUDA context,
    # avoiding the ~30 s cudaHostAlloc cost they'd otherwise pay per swap.
    pool: Optional[SharedPool] = None
    try:
        pool = SharedPool()
        pool.open()
    except Exception:
        logger.exception("shared pool unavailable; falling back to private-pool mode")
        pool = None
    manager = ModelManager(app, REGISTRY, shared_pool=pool)
    manager.start()  # launch the dispatcher coroutine
    attach_routes(app, manager)

    default_model = os.environ.get("BLITZ_DEFAULT_MODEL")
    if default_model and default_model in REGISTRY:
        logger.info("preloading default model %s", default_model)
        await manager.acquire(default_model)
        await manager.release()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("BlitzInfer rotation gateway listening on 0.0.0.0:8000")
    try:
        await server.serve()
    finally:
        logger.info("gateway exiting; tearing down active engine")
        await manager.stop()
        await _shutdown_engine(manager)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
