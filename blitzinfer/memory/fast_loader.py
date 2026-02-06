"""Fast safetensor loader for direct read into pinned memory arena.

Chunk-parallel loading: splits each file into 2GB chunks and reads ALL chunks
across ALL files in parallel via ThreadPoolExecutor. This saturates NVMe queue
depth for 10-13+ GB/s throughput, vs ~3 GB/s with per-file parallelism.

Key insight: f.readinto() releases the GIL during C-level I/O, so threads
run truly parallel. Each thread opens its own fd (no fd lock contention)
and writes to a non-overlapping arena region (no synchronization needed).
"""

import ctypes
import json
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .arena import PinnedMemoryArena, TensorMeta

logger = logging.getLogger(__name__)

# Default chunk size for parallel reads (2GB)
DEFAULT_READ_CHUNK_BYTES = 2 * 1024**3

# Safetensor format constants
SAFETENSOR_HEADER_SIZE_BYTES = 8  # uint64 little-endian

# Dtype mapping from safetensor format to torch
SAFETENSOR_DTYPE_MAP = {
    'F64': torch.float64,
    'F32': torch.float32,
    'F16': torch.float16,
    'BF16': torch.bfloat16,
    'I64': torch.int64,
    'I32': torch.int32,
    'I16': torch.int16,
    'I8': torch.int8,
    'U8': torch.uint8,
    'BOOL': torch.bool,
    # FP8 dtypes (1 byte per element)
    'F8_E4M3': torch.float8_e4m3fn,
    'F8_E5M2': torch.float8_e5m2,
}

# Bytes per element for each dtype
DTYPE_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
    # FP8 dtypes
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
}


def parse_safetensor_header(file_path: str) -> Tuple[int, Dict]:
    """Parse safetensor file header to get tensor metadata.

    Args:
        file_path: Path to the .safetensors file.

    Returns:
        Tuple of (header_size, metadata_dict) where metadata_dict maps
        tensor names to their offset, shape, and dtype info.
    """
    with open(file_path, 'rb') as f:
        # Read header size (first 8 bytes, little-endian uint64)
        header_size_bytes = f.read(SAFETENSOR_HEADER_SIZE_BYTES)
        header_size = struct.unpack('<Q', header_size_bytes)[0]

        # Read header JSON
        header_json = f.read(header_size)
        header = json.loads(header_json)

    return header_size, header


def get_tensor_info(header: Dict) -> Dict[str, Dict]:
    """Extract tensor information from safetensor header.

    Args:
        header: Parsed safetensor header dict.

    Returns:
        Dict mapping tensor names to {dtype, shape, data_offsets}.
    """
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        tensors[name] = {
            'dtype': SAFETENSOR_DTYPE_MAP.get(info['dtype'], torch.float16),
            'shape': tuple(info['shape']),
            'data_offsets': info['data_offsets'],  # [start, end] relative to data section
        }
    return tensors


def get_model_size(model_path: str) -> int:
    """Calculate total size of all safetensor files for a model.

    Args:
        model_path: Path to the model directory.

    Returns:
        Total size in bytes.
    """
    model_path = Path(model_path)
    total = 0

    for sf_file in sorted(model_path.glob('*.safetensors')):
        total += os.path.getsize(sf_file)

    return total


def get_safetensor_files(model_path: str) -> List[Path]:
    """Get sorted list of safetensor files for a model.

    Args:
        model_path: Path to the model directory.

    Returns:
        Sorted list of safetensor file paths.
    """
    model_path = Path(model_path)
    files = list(model_path.glob('*.safetensors'))

    # Sort by name to ensure consistent ordering
    # Handle numbered shards like model-00001-of-00003.safetensors
    def sort_key(p):
        name = p.stem
        # Extract shard number if present
        if '-of-' in name:
            try:
                parts = name.split('-')
                for i, part in enumerate(parts):
                    if part.isdigit():
                        return (0, int(part), name)
            except (ValueError, IndexError):
                pass
        return (1, 0, name)

    return sorted(files, key=sort_key)


def _read_chunk_into_arena(
    file_path: str,
    file_offset: int,
    read_size: int,
    arena: PinnedMemoryArena,
    arena_offset: int,
) -> int:
    """Read a chunk of a file into the arena at the specified offset.

    Each call opens its own file descriptor to avoid fd lock contention
    between threads. f.readinto() releases the GIL during I/O.

    Handles arena chunk boundaries: if the arena region spans multiple
    chunks, reads up to each boundary and continues in the next chunk.

    Args:
        file_path: Path to the file.
        file_offset: Byte offset within the file to start reading.
        read_size: Number of bytes to read.
        arena: Arena to read into.
        arena_offset: Byte offset within the arena to write to.

    Returns:
        Number of bytes actually read.
    """
    bytes_read = 0
    current_arena_offset = arena_offset

    with open(file_path, 'rb') as f:
        f.seek(file_offset)

        while bytes_read < read_size:
            remaining = read_size - bytes_read

            # Get buffer pointer(s) for the current arena position
            segments = arena.get_buffer_ptr(current_arena_offset, remaining)

            for ptr, available in segments:
                to_read = min(remaining - bytes_read, available)
                if to_read <= 0:
                    break

                buffer_view = (ctypes.c_char * to_read).from_address(ptr)
                n = f.readinto(buffer_view)
                if n is None or n == 0:
                    return bytes_read  # EOF
                bytes_read += n
                current_arena_offset += n

                if n < to_read:
                    return bytes_read  # EOF

            if bytes_read >= read_size:
                break

    return bytes_read


def _load_single_file(
    sf_file: Path,
    arena: PinnedMemoryArena,
    file_offset: int,
) -> Tuple[Path, int, int, Dict]:
    """Load a single safetensor file into the arena (legacy, per-file).

    Args:
        sf_file: Path to safetensor file.
        arena: Arena to load into.
        file_offset: Offset within arena for this file.

    Returns:
        Tuple of (file_path, header_size, bytes_read, tensor_info).
    """
    header_size, header = parse_safetensor_header(str(sf_file))
    tensor_info = get_tensor_info(header)
    bytes_read = arena.read_file_into(str(sf_file), file_offset)
    return sf_file, header_size, bytes_read, tensor_info


def load_model_to_arena(
    model_path: str,
    arena: PinnedMemoryArena,
    model_name: Optional[str] = None,
    parallel_workers: int = 16,
    read_chunk_bytes: int = DEFAULT_READ_CHUNK_BYTES,
) -> Dict[str, TensorMeta]:
    """Load all model safetensors into the arena with chunk-parallel I/O.

    Splits each file into read_chunk_bytes chunks and reads ALL chunks
    across ALL files in parallel. With 16 workers and 2GB chunks, a 65GB
    model produces ~32 outstanding I/O ops that saturate NVMe bandwidth.

    Args:
        model_path: Path to the model directory.
        arena: PinnedMemoryArena to load into.
        model_name: Optional name override (defaults to model_path basename).
        parallel_workers: Number of parallel I/O threads (default 16).
        read_chunk_bytes: Size of each read chunk in bytes (default 2GB).

    Returns:
        Dict mapping tensor names to TensorMeta.
    """
    model_path = Path(model_path)
    if model_name is None:
        model_name = model_path.name

    logger.info(f"Loading {model_name} into arena from {model_path}")
    start_time = time.time()

    # Get all safetensor files
    sf_files = get_safetensor_files(str(model_path))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    # Calculate total size and file offsets
    file_sizes = [os.path.getsize(f) for f in sf_files]
    total_size = sum(file_sizes)

    file_offsets = []
    current_offset = 0
    for size in file_sizes:
        file_offsets.append(current_offset)
        current_offset += size

    logger.info(
        f"Found {len(sf_files)} safetensor files, "
        f"total {total_size / 1024**3:.2f}GB, "
        f"chunk_size={read_chunk_bytes / 1024**3:.1f}GB, "
        f"workers={parallel_workers}"
    )

    # Parse ALL headers first (small, sequential, fast)
    header_parse_start = time.time()
    file_headers = []
    for sf_file in sf_files:
        header_size, header = parse_safetensor_header(str(sf_file))
        tensor_info = get_tensor_info(header)
        file_headers.append((header_size, tensor_info))
    header_parse_time = time.time() - header_parse_start
    logger.debug(f"Parsed {len(sf_files)} headers in {header_parse_time:.3f}s")

    # Allocate space in arena
    base_offset = arena.allocate(model_name, total_size)
    arena.set_status(model_name, 'loading')

    # Build flat list of read chunks across ALL files
    # Each chunk: (file_path, file_offset, chunk_size, arena_offset)
    read_chunks = []
    for i, sf_file in enumerate(sf_files):
        file_size = file_sizes[i]
        arena_file_start = base_offset + file_offsets[i]

        pos = 0
        while pos < file_size:
            chunk_size = min(read_chunk_bytes, file_size - pos)
            read_chunks.append((
                str(sf_file),    # file path
                pos,             # offset within file
                chunk_size,      # bytes to read
                arena_file_start + pos,  # offset within arena
            ))
            pos += chunk_size

    logger.info(
        f"Split into {len(read_chunks)} read chunks "
        f"({read_chunk_bytes / 1024**3:.1f}GB each)"
    )

    # Read ALL chunks in parallel
    io_start = time.time()
    total_bytes_read = 0

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {}
        for chunk_spec in read_chunks:
            fp, fo, cs, ao = chunk_spec
            future = executor.submit(_read_chunk_into_arena, fp, fo, cs, arena, ao)
            futures[future] = chunk_spec

        for future in as_completed(futures):
            chunk_spec = futures[future]
            try:
                n = future.result()
                total_bytes_read += n
            except Exception as e:
                fp, fo, cs, ao = chunk_spec
                logger.error(f"Failed to read chunk: file={fp}, offset={fo}, size={cs}: {e}")
                raise

    io_time = time.time() - io_start
    io_speed = (total_bytes_read / 1024**3) / io_time if io_time > 0 else 0

    # Register tensor metadata (fast, just bookkeeping)
    all_tensors: Dict[str, TensorMeta] = {}

    for i, (header_size, tensor_info) in enumerate(file_headers):
        data_offset = SAFETENSOR_HEADER_SIZE_BYTES + header_size
        offset = file_offsets[i]

        for tensor_name, info in tensor_info.items():
            dtype = info['dtype']
            shape = info['shape']
            data_start, data_end = info['data_offsets']
            tensor_size = data_end - data_start
            tensor_offset = offset + data_offset + data_start

            arena.register_tensor(
                model_name=model_name,
                tensor_name=tensor_name,
                offset=tensor_offset,
                size_bytes=tensor_size,
                shape=shape,
                dtype=dtype,
            )

            all_tensors[tensor_name] = TensorMeta(
                offset=tensor_offset,
                size_bytes=tensor_size,
                shape=shape,
                dtype=dtype,
                name=tensor_name,
            )

    arena.set_status(model_name, 'ready')

    elapsed = time.time() - start_time
    total_speed = (total_bytes_read / 1024**3) / elapsed if elapsed > 0 else 0

    logger.info(
        f"Loaded {model_name}: {total_bytes_read / 1024**3:.2f}GB in {elapsed:.2f}s "
        f"(I/O: {io_speed:.1f} GB/s, total: {total_speed:.1f} GB/s), "
        f"{len(all_tensors)} tensors, {len(read_chunks)} chunks"
    )

    return all_tensors


def load_safetensor_to_arena(
    file_path: str,
    arena: PinnedMemoryArena,
    model_name: str,
    arena_offset: int,
) -> Tuple[int, Dict[str, TensorMeta]]:
    """Load a single safetensor file into the arena.

    Lower-level function for loading individual files.

    Args:
        file_path: Path to the .safetensors file.
        arena: PinnedMemoryArena to load into.
        model_name: Name of the model (for registration).
        arena_offset: Offset within the arena to load at.

    Returns:
        Tuple of (bytes_read, tensor_metadata).
    """
    file_path = Path(file_path)
    file_size = os.path.getsize(file_path)

    # Parse header
    header_size, header = parse_safetensor_header(str(file_path))
    tensor_info = get_tensor_info(header)
    data_offset = SAFETENSOR_HEADER_SIZE_BYTES + header_size

    # Read file into arena
    bytes_read = arena.read_file_into(str(file_path), arena_offset)

    # Register tensors
    tensors: Dict[str, TensorMeta] = {}
    for tensor_name, info in tensor_info.items():
        dtype = info['dtype']
        shape = info['shape']
        data_start, data_end = info['data_offsets']
        tensor_size = data_end - data_start

        # Offset within the file's data section
        tensor_offset = data_offset + data_start

        meta = TensorMeta(
            offset=tensor_offset,
            size_bytes=tensor_size,
            shape=shape,
            dtype=dtype,
            name=tensor_name,
        )
        tensors[tensor_name] = meta

    return bytes_read, tensors


def estimate_gpu_transfer_time(
    model_size_bytes: int,
    bandwidth_gbps: float = 45.0,
) -> float:
    """Estimate time to transfer model from pinned memory to GPU.

    Args:
        model_size_bytes: Size of model in bytes.
        bandwidth_gbps: Expected transfer bandwidth in GB/s.

    Returns:
        Estimated time in seconds.
    """
    size_gb = model_size_bytes / 1024**3
    return size_gb / bandwidth_gbps
