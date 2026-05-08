"""BlitzInfer pinned-staged safetensors loader (v4 — shared hugetlbfs pool).

Two operating modes, switched by env vars set by the parent gateway:

(A) Shared-pool mode  — when BLITZ_SHARED_POOL is set:
    Parent has already populated /mnt/hugetlbfs/blitz_pool with this model's
    safetensors bytes, sequentially per shard. Subprocess opens that file,
    re-registers it with CUDA in its own context (~1.1 s for 64 GB), and
    builds tensor views into the registered region. No disk read here.

(B) Private-pool mode — when BLITZ_SHARED_POOL is NOT set:
    Subprocess allocates its own 80 GB region via MAP_HUGETLB | MAP_HUGE_1GB
    and reads shards into it. Used as a fallback if the parent didn't
    pre-load (e.g., admin endpoint, manual test).

Activation: vLLM 0.20.1's default_loader.py imports
``multi_thread_safetensors_weights_iterator`` from this module via the
patched import. Server passes ``enable_multithread_load=True`` via
``--model-loader-extra-config``.

Requires kernel cmdline: ``default_hugepagesz=1G hugepagesz=1G hugepages=80``.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import itertools
import json
import logging
import os
import struct
import threading
from collections.abc import Generator
from typing import Optional

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):
        return it


logger = logging.getLogger(__name__)


# --- libc / libcudart wrappers -----------------------------------------------

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
_cudaGetErrorString = _libcudart.cudaGetErrorString
_cudaGetErrorString.argtypes = [ctypes.c_int]
_cudaGetErrorString.restype = ctypes.c_char_p

_PROT_READ = 1
_PROT_WRITE = 2
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_SHARED = 0x01
_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26
_MAP_HUGE_1GB = 30 << _MAP_HUGE_SHIFT
_MAP_FAILED = ctypes.c_void_p(-1).value


def _cuda_check(rc: int, op: str):
    if rc != 0:
        msg = _cudaGetErrorString(rc).decode()
        raise RuntimeError(f"{op} rc={rc}: {msg}")


# --- Pool state (per-subprocess; one of two modes) ---------------------------

_pool_lock = threading.Lock()
_pool_addr: Optional[int] = None
_pool_size: int = 0
_pool_fd: Optional[int] = None  # set in shared-pool mode
_pool_mode: str = ""  # "shared" or "private"


def _open_shared_pool(path: str) -> tuple[int, int]:
    """Open + mmap + register the parent's shared hugetlbfs pool."""
    global _pool_fd
    fd = os.open(path, os.O_RDWR)
    sz = os.fstat(fd).st_size
    if sz == 0:
        # hugetlbfs files report size only after mmap
        # We don't know size from stat; use BLITZ_POOL_GB if provided
        sz = int(os.environ.get("BLITZ_POOL_GB", "80")) * 1024**3
    addr = _libc.mmap(None, sz, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
    if addr is None or addr == _MAP_FAILED:
        errno = ctypes.get_errno()
        os.close(fd)
        raise OSError(errno, os.strerror(errno))
    rc = _cudaHostRegister(addr, sz, 0)
    if rc != 0:
        _libc.munmap(addr, sz)
        os.close(fd)
        _cuda_check(rc, "cudaHostRegister")
    _pool_fd = fd
    return addr, sz


def _open_private_pool(size: int) -> tuple[int, int]:
    """Allocate a private 1 GB-hugetlb anonymous region."""
    flags = _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB | _MAP_HUGE_1GB
    addr = _libc.mmap(None, size, _PROT_READ | _PROT_WRITE, flags, -1, 0)
    if addr is None or addr == _MAP_FAILED:
        # Fallback to 4 KB pages
        flags = _MAP_PRIVATE | _MAP_ANONYMOUS
        addr = _libc.mmap(None, size, _PROT_READ | _PROT_WRITE, flags, -1, 0)
        if addr is None or addr == _MAP_FAILED:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        logger.warning("private pool: 1GB hugepages unavailable, using 4 KB pages")
    rc = _cudaHostRegister(addr, size, 0)
    if rc != 0:
        _libc.munmap(addr, size)
        _cuda_check(rc, "cudaHostRegister")
    return addr, size


def _ensure_pool() -> tuple[int, int, str]:
    """Return (addr, size, mode). Lazy. mode is 'shared' or 'private'."""
    global _pool_addr, _pool_size, _pool_mode
    with _pool_lock:
        if _pool_addr is not None:
            return _pool_addr, _pool_size, _pool_mode
        shared = os.environ.get("BLITZ_SHARED_POOL")
        if shared:
            import time
            t0 = time.perf_counter()
            _pool_addr, _pool_size = _open_shared_pool(shared)
            _pool_mode = "shared"
            logger.info(
                "loader: opened SHARED pool %s (%.1f GB) in %.2fs",
                shared, _pool_size / 1e9, time.perf_counter() - t0,
            )
        else:
            import time
            size = int(os.environ.get("BLITZ_POOL_GB", "80")) * 1024**3
            t0 = time.perf_counter()
            _pool_addr, _pool_size = _open_private_pool(size)
            _pool_mode = "private"
            logger.info(
                "loader: opened PRIVATE pool (%.1f GB) in %.2fs",
                _pool_size / 1e9, time.perf_counter() - t0,
            )
        return _pool_addr, _pool_size, _pool_mode


# --- Safetensors header parsing ---------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _addr_to_uint8_tensor(addr: int, nbytes: int) -> torch.Tensor:
    arr = (ctypes.c_uint8 * nbytes).from_address(addr)
    return torch.frombuffer(arr, dtype=torch.uint8)


def _parse_shard_at(
    pool_addr: int,
    byte_offset: int,
    byte_size: int,
    local_expert_ids: Optional[set[int]],
) -> dict[str, torch.Tensor]:
    """Treat pool[byte_offset:+byte_size] as a safetensors shard; return tensor views."""
    base = pool_addr + byte_offset
    arr = (ctypes.c_uint8 * byte_size).from_address(base)
    header_len = struct.unpack("<Q", bytes(arr[0:8]))[0]
    if header_len <= 0 or header_len > byte_size - 8:
        raise ValueError(f"bad safetensors header_len {header_len}")
    header_json = bytes(arr[8 : 8 + header_len]).decode("utf-8")
    header = json.loads(header_json)
    data_base = 8 + header_len

    out: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if local_expert_ids is not None and _should_skip(name, local_expert_ids):
            continue
        dtype_str = info["dtype"]
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype {dtype_str!r} for {name}")
        dtype = _DTYPE_MAP[dtype_str]
        shape = list(info["shape"])
        start, end = info["data_offsets"]
        nbytes = end - start
        tensor_addr = base + data_base + start
        u8 = _addr_to_uint8_tensor(tensor_addr, nbytes)
        if dtype is torch.uint8:
            t = u8.reshape(shape) if shape else u8
        else:
            t = u8.view(dtype).reshape(shape) if shape else u8.view(dtype)
        out[name] = t
    return out


def _should_skip(name: str, local_expert_ids: set[int]) -> bool:
    import re

    m = re.search(r"experts?\.?_?(\d+)\.", name)
    return bool(m) and int(m.group(1)) not in local_expert_ids


# --- Two iterator paths: shared-pool (no disk) and private-pool (with disk) --

def _iter_shared(
    hf_weights_files: list[str],
    layout: list[dict],
    local_expert_ids: Optional[set[int]],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    pool_addr, _, _ = _ensure_pool()
    # match hf_weights_files to layout entries by basename
    by_basename = {os.path.basename(e["shard_path"]): e for e in layout}
    progress = tqdm(
        hf_weights_files,
        desc="blitz shared-pool loader",
        disable=not use_tqdm_on_load,
    )
    for shard_path in progress:
        entry = by_basename.get(os.path.basename(shard_path))
        if entry is None:
            raise RuntimeError(
                f"shard {shard_path} not in BLITZ_SHARDS_LAYOUT; "
                f"available: {list(by_basename)}"
            )
        shard = _parse_shard_at(
            pool_addr, entry["byte_offset"], entry["byte_size"], local_expert_ids,
        )
        for name in list(shard):
            yield name, shard.pop(name)


def _iter_private(
    hf_weights_files: list[str],
    max_workers: int,
    local_expert_ids: Optional[set[int]],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    pool_addr, pool_size, _ = _ensure_pool()
    real_paths = [os.path.realpath(f) for f in hf_weights_files]
    sizes = [os.path.getsize(p) for p in real_paths]
    offsets = list(itertools.accumulate([0] + sizes[:-1]))
    total = sum(sizes)
    if total > pool_size:
        raise RuntimeError(
            f"model {total/1e9:.1f} GB > pool {pool_size/1e9:.1f} GB"
        )

    def _read_shard(idx: int) -> tuple[int, int]:
        sz = sizes[idx]
        offset = offsets[idx]
        arr = (ctypes.c_uint8 * sz).from_address(pool_addr + offset)
        with open(real_paths[idx], "rb", buffering=0) as f:
            mv = memoryview(arr)
            n = 0
            while n < sz:
                r = f.readinto(mv[n:])
                if r is None or r == 0:
                    break
                n += r
        if n != sz:
            raise IOError(f"short read on {real_paths[idx]}")
        return idx, sz

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_read_shard, i) for i in range(len(hf_weights_files))]
        progress = tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="blitz private-pool loader",
            disable=not use_tqdm_on_load,
        )
        for fut in progress:
            idx, sz = fut.result()
            shard = _parse_shard_at(
                pool_addr, offsets[idx], sz, local_expert_ids,
            )
            for name in list(shard):
                yield name, shard.pop(name)


def multi_thread_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    max_workers: int = 8,
    local_expert_ids: Optional[set[int]] = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    if not hf_weights_files:
        return
    layout_path = os.environ.get("BLITZ_SHARDS_LAYOUT")
    if layout_path and os.environ.get("BLITZ_SHARED_POOL"):
        with open(layout_path) as f:
            layout = json.load(f)["shards"]
        yield from _iter_shared(
            hf_weights_files, layout, local_expert_ids, use_tqdm_on_load,
        )
    else:
        yield from _iter_private(
            hf_weights_files, max_workers, local_expert_ids, use_tqdm_on_load,
        )
