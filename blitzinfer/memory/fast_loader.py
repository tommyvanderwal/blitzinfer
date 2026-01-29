"""Fast safetensor loader for direct read into pinned memory arena."""

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


def _load_single_file(
    sf_file: Path,
    arena: PinnedMemoryArena,
    file_offset: int,
) -> Tuple[Path, int, int, Dict]:
    """Load a single safetensor file into the arena.

    Args:
        sf_file: Path to safetensor file.
        arena: Arena to load into.
        file_offset: Offset within arena for this file.

    Returns:
        Tuple of (file_path, header_size, bytes_read, tensor_info).
    """
    # Parse header
    header_size, header = parse_safetensor_header(str(sf_file))
    tensor_info = get_tensor_info(header)

    # Read file into arena
    bytes_read = arena.read_file_into(str(sf_file), file_offset)

    return sf_file, header_size, bytes_read, tensor_info


def load_model_to_arena(
    model_path: str,
    arena: PinnedMemoryArena,
    model_name: Optional[str] = None,
    parallel_workers: int = 16,  # Increased for NVMe queue depth
) -> Dict[str, TensorMeta]:
    """Load all model safetensors into the arena.

    This function:
    1. Calculates total model size
    2. Allocates space in the arena
    3. Reads safetensor files in parallel for maximum throughput
    4. Parses headers and registers tensor metadata

    Args:
        model_path: Path to the model directory.
        arena: PinnedMemoryArena to load into.
        model_name: Optional name override (defaults to model_path basename).
        parallel_workers: Number of parallel file reads (default 4).

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

    logger.debug(f"Found {len(sf_files)} safetensor files")

    # Calculate total size needed and file offsets
    file_sizes = [os.path.getsize(f) for f in sf_files]
    total_size = sum(file_sizes)
    logger.debug(f"Total model size: {total_size / 1024**3:.2f}GB")

    # Calculate offsets for each file
    file_offsets = []
    current_offset = 0
    for size in file_sizes:
        file_offsets.append(current_offset)
        current_offset += size

    # Allocate space in arena
    base_offset = arena.allocate(model_name, total_size)
    arena.set_status(model_name, 'loading')

    # Track tensor metadata across all files
    all_tensors: Dict[str, TensorMeta] = {}
    total_bytes_read = 0

    # Load files in parallel
    num_workers = min(parallel_workers, len(sf_files))
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i, sf_file in enumerate(sf_files):
            file_offset = base_offset + file_offsets[i]
            future = executor.submit(_load_single_file, sf_file, arena, file_offset)
            futures[future] = (i, sf_file, file_offsets[i])

        for future in as_completed(futures):
            idx, sf_file, offset = futures[future]
            try:
                _, header_size, bytes_read, tensor_info = future.result()
                results.append((idx, offset, header_size, bytes_read, tensor_info))
                total_bytes_read += bytes_read
            except Exception as e:
                logger.error(f"Failed to load {sf_file}: {e}")
                raise

    # Sort by file index to maintain consistent ordering
    results.sort(key=lambda x: x[0])

    # Register tensors in order
    for idx, offset, header_size, bytes_read, tensor_info in results:
        data_offset = SAFETENSOR_HEADER_SIZE_BYTES + header_size

        for tensor_name, info in tensor_info.items():
            dtype = info['dtype']
            shape = info['shape']
            data_start, data_end = info['data_offsets']
            tensor_size = data_end - data_start

            # Calculate absolute offset within the model's allocation
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
    speed_gbps = (total_bytes_read / 1024**3) / elapsed

    logger.info(
        f"Loaded {model_name}: {total_bytes_read / 1024**3:.2f}GB in {elapsed:.2f}s "
        f"({speed_gbps:.1f} GB/s), {len(all_tensors)} tensors"
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
