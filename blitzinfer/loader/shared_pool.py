"""BlitzInfer parent-side shared pool manager.

Owns ONE hugetlbfs-backed file (default /mnt/hugetlbfs/blitz_pool, 80 GB)
that is mmap'd + cudaHostRegister'd in the parent gateway process. The
parent reads safetensors shards into the shared file as needed; child
EngineCore subprocesses open the same file and re-register it in their
own CUDA context (~1.1 s per 64 GB based on measurements) — much faster
than both cudaHostAlloc (12 s) and the per-shard pinned alloc dance the
v3 loader paid.

Subprocess wiring: the parent sets two environment variables before
spawning vLLM:
    BLITZ_SHARED_POOL=/mnt/hugetlbfs/blitz_pool
    BLITZ_SHARDS_LAYOUT=/tmp/blitz_layout_<pid>.json
``blitzinfer.loader.pinned_loader`` checks these and switches its read
path to the pool.

The layout JSON describes per-shard byte offsets so the subprocess can
treat each shard's slice as a self-contained safetensors blob:

    {"shards": [{"shard_path": "...", "byte_offset": 0, "byte_size": N}, ...]}
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# --- libc / libcudart ---------------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libcudart = ctypes.CDLL("libcudart.so")

_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int

_cudaHostRegister = _libcudart.cudaHostRegister
_cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudaHostRegister.restype = ctypes.c_int
_cudaHostUnregister = _libcudart.cudaHostUnregister
_cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudaHostUnregister.restype = ctypes.c_int

_PROT_READ = 1
_PROT_WRITE = 2
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value

DEFAULT_POOL_PATH = "/mnt/hugetlbfs/blitz_pool"
DEFAULT_POOL_SIZE = int(os.environ.get("BLITZ_POOL_GB", "80")) * 1024**3


class SharedPool:
    """Parent-side handle to the shared hugetlbfs pool."""

    def __init__(
        self,
        path: str = DEFAULT_POOL_PATH,
        size_bytes: int = DEFAULT_POOL_SIZE,
    ):
        self.path = path
        self.size = size_bytes
        self._addr: Optional[int] = None
        self._fd: Optional[int] = None
        self._loaded_model: Optional[str] = None
        self._loaded_layout: list[dict] = []
        self._lock = threading.Lock()

    def open(self) -> None:
        """Open the hugetlbfs file and mmap it for writing. Does NOT
        cudaHostRegister — that's the subprocess's job in its own CUDA
        context. If the parent registered, the resulting UVA mapping
        would consume ~80 GB of GPU virtual address space (visible to
        nvidia-smi as 'used') and break subprocess `cudaMemGetInfo`."""
        if self._addr is not None:
            return
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        t0 = time.perf_counter()
        addr = _libc.mmap(
            None, self.size, _PROT_READ | _PROT_WRITE, _MAP_SHARED, self._fd, 0
        )
        if addr is None or addr == _MAP_FAILED:
            errno = ctypes.get_errno()
            os.close(self._fd)
            self._fd = None
            raise OSError(
                errno,
                f"mmap hugetlbfs at {self.path} (size={self.size/1e9:.1f}GB): "
                f"{os.strerror(errno)}",
            )
        t_map = time.perf_counter() - t0
        self._addr = addr
        logger.info(
            "shared_pool: opened %s (%.1f GB), mmap %.0fms — NOT registered "
            "(subprocess registers in its own CUDA context)",
            self.path,
            self.size / 1e9,
            t_map * 1000,
        )

    def close(self) -> None:
        with self._lock:
            if self._addr is not None:
                _libc.munmap(self._addr, self.size)
                self._addr = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            try:
                if os.path.exists(self.path):
                    os.unlink(self.path)
            except OSError:
                pass

    def is_loaded(self, model_name: str) -> bool:
        with self._lock:
            return self._loaded_model == model_name

    def load_shards(
        self,
        model_name: str,
        shard_paths: list[str],
        shards_in_parallel: int = 4,
        chunks_per_shard: int = 16,
    ) -> list[dict]:
        """Read shard files into the pool with chunked parallel reads.

        Each shard is split into ``chunks_per_shard`` chunks; each chunk is
        pread() in its own thread, giving the kernel an effective high
        I/O queue depth (single-thread f.readinto tops out at ~4.5 GB/s
        per drive; chunked preadv with 16 chunks reaches ~10-11 GB/s).
        Up to ``shards_in_parallel`` shards run concurrently.

        Reuses pool if model_name matches the currently-loaded model.
        Layout entries: {shard_path, byte_offset, byte_size}.
        """
        import concurrent.futures
        with self._lock:
            if self._addr is None:
                raise RuntimeError("pool not opened")
            if self._loaded_model == model_name:
                logger.info("shared_pool: %s already loaded, reusing", model_name)
                return list(self._loaded_layout)

            t0 = time.perf_counter()
            real_paths = [os.path.realpath(p) for p in shard_paths]
            sizes = [os.path.getsize(p) for p in real_paths]
            total = sum(sizes)
            if total > self.size:
                raise RuntimeError(
                    f"model {model_name} weights {total/1e9:.1f} GB > pool "
                    f"{self.size/1e9:.1f} GB"
                )

            # Pre-compute byte offsets per shard
            offsets = []
            cur = 0
            for sz in sizes:
                offsets.append(cur)
                cur += sz

            pool_addr = self._addr

            def _read_chunk(fd, file_offset, chunk_len, dest_offset):
                """preadv() into pool_addr + dest_offset for chunk_len bytes."""
                cur = file_offset
                end = file_offset + chunk_len
                pool_cur = dest_offset
                while cur < end:
                    remaining = end - cur
                    buf = (ctypes.c_uint8 * remaining).from_address(
                        pool_addr + pool_cur
                    )
                    n = os.preadv(fd, [memoryview(buf)], cur)
                    if n == 0:
                        raise IOError("short read")
                    cur += n
                    pool_cur += n

            def _read_one_shard(idx: int):
                sz = sizes[idx]
                pool_offset = offsets[idx]
                fd = os.open(real_paths[idx], os.O_RDONLY)
                try:
                    chunk_size = (sz + chunks_per_shard - 1) // chunks_per_shard
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=chunks_per_shard
                    ) as inner:
                        futs = []
                        for c in range(chunks_per_shard):
                            file_off = c * chunk_size
                            chunk_end = min(file_off + chunk_size, sz)
                            chunk_len = chunk_end - file_off
                            if chunk_len <= 0:
                                continue
                            futs.append(inner.submit(
                                _read_chunk,
                                fd,
                                file_off,
                                chunk_len,
                                pool_offset + file_off,
                            ))
                        for f in futs:
                            f.result()
                finally:
                    os.close(fd)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=shards_in_parallel
            ) as ex:
                list(ex.map(_read_one_shard, range(len(shard_paths))))

            layout = [
                {
                    "shard_path": shard_paths[i],
                    "byte_offset": offsets[i],
                    "byte_size": sizes[i],
                }
                for i in range(len(shard_paths))
            ]
            self._loaded_model = model_name
            self._loaded_layout = layout
            elapsed = time.perf_counter() - t0
            logger.info(
                "shared_pool: loaded %s (%d shards, %.1f GB) into pool in "
                "%.1fs (%.1f GB/s)  [%d shards in parallel × %d chunks/shard]",
                model_name,
                len(shard_paths),
                total / 1e9,
                elapsed,
                total / 1e9 / elapsed if elapsed > 0 else 0,
                shards_in_parallel,
                chunks_per_shard,
            )
            return list(layout)

    def write_layout_json(self, layout: list[dict], path: str) -> None:
        """Write the per-shard byte-offset layout JSON for the subprocess."""
        with open(path, "w") as f:
            json.dump({"shards": layout}, f)
