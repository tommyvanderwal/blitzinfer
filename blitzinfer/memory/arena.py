"""Pinned memory arena for fast GPU transfers."""

import ctypes
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorMeta:
    """Metadata for a tensor stored in the arena."""
    offset: int  # Byte offset within the model's allocation
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    name: str


@dataclass
class ModelAllocation:
    """Tracks a model's allocation within the arena."""
    model_name: str
    start_offset: int
    end_offset: int
    status: str  # 'allocated', 'loading', 'ready', 'transferring'
    tensor_metadata: Dict[str, TensorMeta] = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)

    @property
    def size_bytes(self) -> int:
        return self.end_offset - self.start_offset


class PinnedMemoryArena:
    """Chunked pinned memory arena for fast GPU transfers.

    This class manages pinned memory in chunks (default 10GB each) to avoid
    kernel issues with large allocations. Models are loaded from SSD into
    this arena, then transferred to GPU at high bandwidth (~45 GB/s).

    Memory Layout (chunked):
    ┌────────────────────────────────────────────────────────────────────┐
    │  Chunk 0 (10GB)  │  Chunk 1 (10GB)  │ ... │  Chunk N (10GB)       │
    │  [Model A data]  │  [Model A data]  │     │  [Model B data]       │
    └────────────────────────────────────────────────────────────────────┘

    Each chunk is allocated separately, allowing:
    - Graceful degradation if some chunks fail
    - Better kernel memory handling
    - Per-chunk error reporting
    """

    def __init__(
        self,
        size_gb: float,
        pin_memory: bool = True,
        chunk_size_gb: float = 16.0,
    ):
        """Initialize the arena with a fixed size, allocated in chunks.

        Args:
            size_gb: Total size of the arena in gigabytes.
            pin_memory: Whether to use CUDA pinned memory. Default True.
                       Pinned memory achieves ~44 GB/s transfer speed.
                       Non-pinned achieves ~4.6 GB/s (10x slower).
            chunk_size_gb: Size of each chunk in GB (default 16GB).
                          MUST be a power of 2 (1, 2, 4, 8, 16, 32GB) to avoid
                          PyTorch's power-of-2 rounding overhead.
                          See: https://github.com/pytorch/pytorch/issues/150517
                          Non-power-of-2 sizes get rounded up (e.g., 5GB → 8GB = 60% overhead).
                          16GB chunks give 0% overhead and fast allocation.
        """
        self._lock = threading.RLock()
        self._pinned = pin_memory
        self._chunk_size_bytes = int(chunk_size_gb * 1024**3)
        requested_size = int(size_gb * 1024**3)

        mem_type = "pinned" if pin_memory else "regular"
        num_chunks = (requested_size + self._chunk_size_bytes - 1) // self._chunk_size_bytes
        logger.info(
            f"Allocating {size_gb:.1f}GB {mem_type} memory arena "
            f"({num_chunks} chunks of {chunk_size_gb:.1f}GB)..."
        )
        start = time.time()

        # Allocate memory in chunks
        self._chunks: list = []
        self._chunk_offsets: list = []  # Start offset of each chunk
        total_allocated = 0

        for i in range(num_chunks):
            remaining = requested_size - total_allocated
            this_chunk_size = min(self._chunk_size_bytes, remaining)

            try:
                chunk = torch.empty(
                    this_chunk_size,
                    dtype=torch.uint8,
                    pin_memory=pin_memory,
                    device='cpu'
                )
                self._chunk_offsets.append(total_allocated)
                self._chunks.append(chunk)
                total_allocated += this_chunk_size
                logger.info(f"  Chunk {i+1}/{num_chunks}: {this_chunk_size / 1024**3:.1f}GB OK")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if pin_memory and i == 0:
                    # Fall back to non-pinned memory on first failure
                    logger.warning(f"Pinned memory allocation failed at chunk {i+1}: {e}")
                    logger.warning("Falling back to regular memory")
                    chunk = torch.empty(
                        this_chunk_size,
                        dtype=torch.uint8,
                        pin_memory=False,
                        device='cpu'
                    )
                    self._chunk_offsets.append(total_allocated)
                    self._chunks.append(chunk)
                    total_allocated += this_chunk_size
                    self._pinned = False
                else:
                    # For subsequent failures, keep what we have
                    logger.error(f"Failed to allocate chunk {i+1}/{num_chunks}: {e}")
                    logger.error(f"Arena reduced: {total_allocated / 1024**3:.1f}GB of {size_gb:.1f}GB requested")
                    break

        # Update actual size to what we got
        self._size_bytes = total_allocated

        # For backward compatibility, create a single-buffer view
        # We need this for the read_file_into and get_slice methods
        # Use the first chunk as the primary buffer if only one chunk
        if len(self._chunks) == 1:
            self._buffer = self._chunks[0]
            self._base_ptr = self._buffer.data_ptr()
            self._is_chunked = False
        else:
            # Multiple chunks - keep them separate and handle indexing
            self._buffer = None
            self._base_ptr = None
            self._is_chunked = True

        # Track allocations
        self._allocations: Dict[str, ModelAllocation] = {}
        self._free_offset = 0

        elapsed = time.time() - start
        actual_type = "pinned" if self._pinned else "regular"
        logger.info(f"Arena allocated ({actual_type}, {self._size_bytes / 1024**3:.1f}GB) in {elapsed:.2f}s")

    @property
    def is_pinned(self) -> bool:
        """Whether the arena is using pinned memory."""
        return self._pinned

    @property
    def size_bytes(self) -> int:
        """Total size of the arena in bytes."""
        return self._size_bytes

    @property
    def size_gb(self) -> float:
        """Total size of the arena in gigabytes."""
        return self._size_bytes / (1024**3)

    @property
    def available_bytes(self) -> int:
        """Available space in bytes (simple linear allocation)."""
        with self._lock:
            return self._size_bytes - self._free_offset

    @property
    def available_gb(self) -> float:
        """Available space in gigabytes."""
        return self.available_bytes / (1024**3)

    @property
    def used_bytes(self) -> int:
        """Used space in bytes."""
        with self._lock:
            return self._free_offset

    @property
    def used_gb(self) -> float:
        """Used space in gigabytes."""
        return self.used_bytes / (1024**3)

    def allocate(self, model_name: str, size_bytes: int) -> int:
        """Reserve space for a model.

        Args:
            model_name: Unique identifier for the model.
            size_bytes: Number of bytes to allocate.

        Returns:
            Start offset within the arena.

        Raises:
            MemoryError: If not enough space is available.
        """
        with self._lock:
            # Check if already allocated
            if model_name in self._allocations:
                alloc = self._allocations[model_name]
                if alloc.size_bytes >= size_bytes:
                    logger.debug(f"Model {model_name} already allocated at offset {alloc.start_offset}")
                    return alloc.start_offset
                else:
                    # Need more space - release and reallocate
                    self.release(model_name)

            # Check available space
            if self._free_offset + size_bytes > self._size_bytes:
                # Try to make space by evicting LRU models
                needed = (self._free_offset + size_bytes) - self._size_bytes
                freed = self._evict_lru(needed)
                if freed < needed:
                    raise MemoryError(
                        f"Not enough space in arena. Need {size_bytes / 1024**3:.2f}GB, "
                        f"available {self.available_bytes / 1024**3:.2f}GB"
                    )

            # Allocate
            start = self._free_offset
            self._free_offset += size_bytes

            self._allocations[model_name] = ModelAllocation(
                model_name=model_name,
                start_offset=start,
                end_offset=start + size_bytes,
                status='allocated',
                tensor_metadata={},
                last_accessed=time.time(),
            )

            logger.debug(
                f"Allocated {size_bytes / 1024**3:.2f}GB for {model_name} "
                f"at offset {start} (arena: {self.used_gb:.1f}/{self.size_gb:.1f}GB used)"
            )
            return start

    def _evict_lru(self, needed_bytes: int) -> int:
        """Evict least recently used models to free space.

        Args:
            needed_bytes: Minimum bytes to free.

        Returns:
            Number of bytes actually freed.
        """
        if not self._allocations:
            return 0

        # Sort by last accessed time (oldest first)
        models_by_lru = sorted(
            self._allocations.values(),
            key=lambda a: a.last_accessed
        )

        freed = 0
        evicted = []

        for alloc in models_by_lru:
            # Don't evict models currently being transferred
            if alloc.status == 'transferring':
                continue

            evicted.append(alloc.model_name)
            freed += alloc.size_bytes

            if freed >= needed_bytes:
                break

        # Actually release the models
        for name in evicted:
            logger.info(f"Evicting {name} from arena (LRU)")
            self._release_internal(name)

        # Compact the arena after eviction
        if evicted:
            self._compact()

        return freed

    def _compact(self):
        """Compact the arena by moving all allocations to the start.

        This is a simple compaction strategy that moves all data to remove gaps.
        For production use, a more sophisticated approach might be needed.
        """
        if not self._allocations:
            self._free_offset = 0
            return

        # Sort allocations by start offset
        sorted_allocs = sorted(
            self._allocations.values(),
            key=lambda a: a.start_offset
        )

        new_offset = 0
        for alloc in sorted_allocs:
            if alloc.start_offset != new_offset:
                # Need to move this allocation
                size = alloc.size_bytes

                # Move the data
                src_start = alloc.start_offset
                self._buffer[new_offset:new_offset + size] = self._buffer[src_start:src_start + size]

                # Update tensor metadata offsets
                offset_delta = new_offset - alloc.start_offset
                for tensor_meta in alloc.tensor_metadata.values():
                    tensor_meta.offset += offset_delta

                # Update allocation
                alloc.start_offset = new_offset
                alloc.end_offset = new_offset + size

            new_offset = alloc.end_offset

        self._free_offset = new_offset

    def _get_chunk_for_offset(self, offset: int) -> Tuple[int, int]:
        """Find which chunk contains a given offset.

        Returns:
            Tuple of (chunk_index, offset_within_chunk)
        """
        for i, chunk_start in enumerate(self._chunk_offsets):
            chunk_end = chunk_start + len(self._chunks[i])
            if offset < chunk_end:
                return i, offset - chunk_start
        raise IndexError(f"Offset {offset} is beyond arena size {self._size_bytes}")

    def read_file_into(self, file_path: str, offset: int) -> int:
        """Read a file directly into the arena at the specified offset.

        Uses direct memory copy for maximum speed. Handles chunked mode
        by reading across chunk boundaries if needed.

        Args:
            file_path: Path to the file to read.
            offset: Byte offset within the arena.

        Returns:
            Number of bytes read.
        """
        file_size = os.path.getsize(file_path)

        if offset + file_size > self._size_bytes:
            raise MemoryError(
                f"Cannot read {file_size} bytes at offset {offset}: "
                f"would exceed arena size {self._size_bytes}"
            )

        if not self._is_chunked:
            # Single buffer mode - simple direct read
            buffer_ptr = self._base_ptr + offset
            buffer_view = (ctypes.c_char * file_size).from_address(buffer_ptr)

            with open(file_path, 'rb') as f:
                bytes_read = f.readinto(buffer_view)
            return bytes_read

        # Chunked mode - may need to read across chunks
        bytes_read = 0
        current_offset = offset

        with open(file_path, 'rb') as f:
            while bytes_read < file_size:
                chunk_idx, chunk_offset = self._get_chunk_for_offset(current_offset)
                chunk = self._chunks[chunk_idx]
                chunk_remaining = len(chunk) - chunk_offset
                to_read = min(file_size - bytes_read, chunk_remaining)

                # Get pointer to this location in the chunk
                chunk_ptr = chunk.data_ptr() + chunk_offset
                buffer_view = (ctypes.c_char * to_read).from_address(chunk_ptr)

                read_now = f.readinto(buffer_view)
                bytes_read += read_now
                current_offset += read_now

                if read_now < to_read:
                    break  # EOF

        return bytes_read

    def get_slice(self, offset: int, size_bytes: int) -> torch.Tensor:
        """Get a slice from the arena buffer.

        Args:
            offset: Start offset in bytes.
            size_bytes: Size of the slice in bytes.

        Returns:
            Tensor (uint8) containing the data. In chunked mode, this may
            require copying if the slice spans multiple chunks.
        """
        if not self._is_chunked:
            return self._buffer[offset:offset + size_bytes]

        # Chunked mode - check if slice is within a single chunk
        chunk_idx, chunk_offset = self._get_chunk_for_offset(offset)
        chunk = self._chunks[chunk_idx]

        if chunk_offset + size_bytes <= len(chunk):
            # Slice fits entirely within this chunk - return view
            return chunk[chunk_offset:chunk_offset + size_bytes]

        # Slice spans multiple chunks - need to copy
        result = torch.empty(size_bytes, dtype=torch.uint8, device='cpu')
        copied = 0
        current_offset = offset

        while copied < size_bytes:
            c_idx, c_off = self._get_chunk_for_offset(current_offset)
            c = self._chunks[c_idx]
            available = len(c) - c_off
            to_copy = min(size_bytes - copied, available)

            result[copied:copied + to_copy] = c[c_off:c_off + to_copy]
            copied += to_copy
            current_offset += to_copy

        return result

    def get_tensor_view(
        self,
        model_name: str,
        tensor_name: str
    ) -> torch.Tensor:
        """Get a tensor view from the arena for GPU transfer.

        Args:
            model_name: Name of the model.
            tensor_name: Name of the tensor.

        Returns:
            Tensor view with correct shape and dtype.

        Raises:
            KeyError: If model or tensor not found.
        """
        with self._lock:
            if model_name not in self._allocations:
                raise KeyError(f"Model {model_name} not in arena")

            alloc = self._allocations[model_name]
            alloc.last_accessed = time.time()

            if tensor_name not in alloc.tensor_metadata:
                raise KeyError(f"Tensor {tensor_name} not found in {model_name}")

            meta = alloc.tensor_metadata[tensor_name]

            # Get byte view from arena (handles chunked mode)
            abs_offset = alloc.start_offset + meta.offset
            byte_view = self.get_slice(abs_offset, meta.size_bytes)

            # Convert to correct dtype and reshape
            # Note: view() requires contiguous memory which we have
            tensor = byte_view.view(meta.dtype).view(meta.shape)

            return tensor

    def get_all_tensors(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Get all tensor views for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary mapping tensor names to tensor views.
        """
        with self._lock:
            if model_name not in self._allocations:
                raise KeyError(f"Model {model_name} not in arena")

            alloc = self._allocations[model_name]
            alloc.last_accessed = time.time()

            tensors = {}
            for tensor_name in alloc.tensor_metadata:
                tensors[tensor_name] = self.get_tensor_view(model_name, tensor_name)

            return tensors

    def register_tensor(
        self,
        model_name: str,
        tensor_name: str,
        offset: int,
        size_bytes: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ):
        """Register tensor metadata for a model.

        Args:
            model_name: Name of the model.
            tensor_name: Name of the tensor.
            offset: Offset within the model's allocation.
            size_bytes: Size of the tensor data in bytes.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
        """
        with self._lock:
            if model_name not in self._allocations:
                raise KeyError(f"Model {model_name} not allocated")

            alloc = self._allocations[model_name]
            alloc.tensor_metadata[tensor_name] = TensorMeta(
                offset=offset,
                size_bytes=size_bytes,
                shape=shape,
                dtype=dtype,
                name=tensor_name,
            )

    def get_allocation(self, model_name: str) -> Optional[ModelAllocation]:
        """Get allocation info for a model."""
        with self._lock:
            return self._allocations.get(model_name)

    def set_status(self, model_name: str, status: str):
        """Update the status of a model allocation."""
        with self._lock:
            if model_name in self._allocations:
                self._allocations[model_name].status = status
                self._allocations[model_name].last_accessed = time.time()

    def release(self, model_name: str):
        """Release a model's allocation.

        Note: This marks the space as free but doesn't compact.
        Call _compact() if you need to reclaim the space.
        """
        with self._lock:
            self._release_internal(model_name)

    def _release_internal(self, model_name: str):
        """Internal release without lock."""
        if model_name in self._allocations:
            alloc = self._allocations.pop(model_name)
            logger.debug(f"Released {alloc.size_bytes / 1024**3:.2f}GB for {model_name}")

            # If arena is now empty, reset free_offset to reclaim all space
            if not self._allocations:
                self._free_offset = 0
                logger.debug("Arena empty, reset free_offset to 0")

    def clear(self):
        """Clear all allocations from the arena."""
        with self._lock:
            self._allocations.clear()
            self._free_offset = 0
            logger.info("Arena cleared")

    def get_stats(self) -> Dict:
        """Get arena statistics."""
        with self._lock:
            return {
                'total_gb': self.size_gb,
                'used_gb': self.used_gb,
                'available_gb': self.available_gb,
                'num_models': len(self._allocations),
                'models': {
                    name: {
                        'size_gb': alloc.size_bytes / 1024**3,
                        'status': alloc.status,
                        'num_tensors': len(alloc.tensor_metadata),
                    }
                    for name, alloc in self._allocations.items()
                }
            }

    def __del__(self):
        """Clean up arena resources."""
        try:
            if hasattr(self, '_buffer'):
                del self._buffer
        except Exception:
            pass
