"""Tests for the RAM-based model prefetch system."""

import os
import sys
import time
import tempfile
import struct
import json
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from blitzinfer.memory.arena import PinnedMemoryArena, ModelAllocation, TensorMeta
from blitzinfer.memory.prefetcher import ModelPrefetcher, PrefetchStatus
from blitzinfer.memory.fast_loader import (
    parse_safetensor_header,
    get_tensor_info,
    get_model_size,
    load_model_to_arena,
    SAFETENSOR_DTYPE_MAP,
)
from blitzinfer.config import PrefetchConfig, BlitzInferConfig


class TestPinnedMemoryArena:
    """Tests for PinnedMemoryArena class."""

    def test_arena_creation(self):
        """Test basic arena creation."""
        arena = PinnedMemoryArena(size_gb=0.1)  # 100MB for testing
        assert arena.size_bytes == int(0.1 * 1024**3)
        assert arena.available_bytes == arena.size_bytes
        assert arena.used_bytes == 0

    def test_arena_allocation(self):
        """Test allocating space in the arena."""
        arena = PinnedMemoryArena(size_gb=0.1)

        # Allocate 10MB
        offset = arena.allocate("test_model", 10 * 1024**2)
        assert offset == 0
        assert arena.used_bytes == 10 * 1024**2
        assert arena.available_bytes == arena.size_bytes - 10 * 1024**2

        # Allocate another 20MB
        offset2 = arena.allocate("test_model_2", 20 * 1024**2)
        assert offset2 == 10 * 1024**2
        assert arena.used_bytes == 30 * 1024**2

    def test_arena_allocation_reuse(self):
        """Test that allocating the same model reuses space."""
        arena = PinnedMemoryArena(size_gb=0.1)

        offset1 = arena.allocate("test_model", 10 * 1024**2)
        offset2 = arena.allocate("test_model", 10 * 1024**2)  # Same model

        assert offset1 == offset2  # Should return same offset

    def test_arena_allocation_overflow(self):
        """Test that allocation fails when arena is full and no evictable models."""
        arena = PinnedMemoryArena(size_gb=0.01)  # 10MB

        # Allocate model1 and mark it as transferring (non-evictable)
        arena.allocate("model1", 5 * 1024**2)
        arena.set_status("model1", "transferring")

        # This should fail because model1 can't be evicted and there's not enough space
        with pytest.raises(MemoryError):
            arena.allocate("model2", 10 * 1024**2)

    def test_arena_release(self):
        """Test releasing allocations."""
        arena = PinnedMemoryArena(size_gb=0.1)

        arena.allocate("test_model", 10 * 1024**2)
        assert arena.get_allocation("test_model") is not None

        arena.release("test_model")
        assert arena.get_allocation("test_model") is None

    def test_arena_clear(self):
        """Test clearing all allocations."""
        arena = PinnedMemoryArena(size_gb=0.1)

        arena.allocate("model1", 10 * 1024**2)
        arena.allocate("model2", 10 * 1024**2)

        arena.clear()

        assert arena.used_bytes == 0
        assert arena.available_bytes == arena.size_bytes

    def test_arena_tensor_registration(self):
        """Test registering tensor metadata."""
        arena = PinnedMemoryArena(size_gb=0.1)

        arena.allocate("test_model", 10 * 1024**2)
        arena.register_tensor(
            model_name="test_model",
            tensor_name="layer.weight",
            offset=0,
            size_bytes=4096,
            shape=(32, 32),
            dtype=torch.float32,
        )

        alloc = arena.get_allocation("test_model")
        assert "layer.weight" in alloc.tensor_metadata
        meta = alloc.tensor_metadata["layer.weight"]
        assert meta.shape == (32, 32)
        assert meta.dtype == torch.float32

    def test_arena_file_read(self):
        """Test reading a file into the arena."""
        arena = PinnedMemoryArena(size_gb=0.1)

        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_data = b"Hello, Arena!" * 1000
            f.write(test_data)
            temp_path = f.name

        try:
            arena.allocate("test_model", len(test_data))
            bytes_read = arena.read_file_into(temp_path, 0)

            assert bytes_read == len(test_data)

            # Verify content
            slice_data = arena.get_slice(0, len(test_data))
            assert slice_data.numpy().tobytes() == test_data
        finally:
            os.unlink(temp_path)

    def test_arena_stats(self):
        """Test getting arena statistics."""
        arena = PinnedMemoryArena(size_gb=0.1)

        arena.allocate("model1", 10 * 1024**2)
        arena.allocate("model2", 20 * 1024**2)

        stats = arena.get_stats()

        assert stats['num_models'] == 2
        assert 'model1' in stats['models']
        assert 'model2' in stats['models']


class TestSafetensorLoader:
    """Tests for safetensor loading utilities."""

    def create_test_safetensor(self, tensors: dict) -> str:
        """Create a test safetensor file and return its path."""
        # Build header
        header = {}
        current_offset = 0

        for name, tensor in tensors.items():
            tensor_bytes = tensor.numpy().tobytes()
            dtype_str = {
                torch.float32: 'F32',
                torch.float16: 'F16',
                torch.bfloat16: 'BF16',
                torch.int64: 'I64',
                torch.int32: 'I32',
            }.get(tensor.dtype, 'F32')

            header[name] = {
                'dtype': dtype_str,
                'shape': list(tensor.shape),
                'data_offsets': [current_offset, current_offset + len(tensor_bytes)],
            }
            current_offset += len(tensor_bytes)

        # Serialize header
        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        # Write file
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
            # Write header size (8 bytes, little-endian)
            f.write(struct.pack('<Q', header_size))
            # Write header
            f.write(header_json)
            # Write tensor data
            for tensor in tensors.values():
                f.write(tensor.numpy().tobytes())

            return f.name

    def test_parse_safetensor_header(self):
        """Test parsing safetensor header."""
        tensors = {
            'weight': torch.randn(10, 10, dtype=torch.float32),
            'bias': torch.randn(10, dtype=torch.float32),
        }
        temp_path = self.create_test_safetensor(tensors)

        try:
            header_size, header = parse_safetensor_header(temp_path)

            assert 'weight' in header
            assert 'bias' in header
            assert header['weight']['shape'] == [10, 10]
            assert header['bias']['shape'] == [10]
        finally:
            os.unlink(temp_path)

    def test_get_tensor_info(self):
        """Test extracting tensor info from header."""
        header = {
            'layer.weight': {
                'dtype': 'F16',
                'shape': [512, 768],
                'data_offsets': [0, 786432],
            },
            '__metadata__': {'format': 'pt'},  # Should be ignored
        }

        info = get_tensor_info(header)

        assert 'layer.weight' in info
        assert '__metadata__' not in info
        assert info['layer.weight']['dtype'] == torch.float16
        assert info['layer.weight']['shape'] == (512, 768)


class TestModelPrefetcher:
    """Tests for ModelPrefetcher class."""

    def test_prefetcher_creation(self):
        """Test creating a prefetcher."""
        arena = PinnedMemoryArena(size_gb=0.1)
        prefetcher = ModelPrefetcher(arena)

        assert prefetcher is not None
        stats = prefetcher.get_stats()
        assert stats['ready_count'] == 0
        assert stats['loading_count'] == 0

    def test_prefetch_status_tracking(self):
        """Test that prefetch status is tracked correctly."""
        arena = PinnedMemoryArena(size_gb=0.1)
        prefetcher = ModelPrefetcher(arena)

        # Model not registered should be COLD
        assert prefetcher.get_status('unknown_model') == PrefetchStatus.COLD
        assert not prefetcher.is_ready('unknown_model')

    def test_prefetch_callback(self):
        """Test that callbacks are fired when model is ready."""
        arena = PinnedMemoryArena(size_gb=0.1)
        prefetcher = ModelPrefetcher(arena)

        callback_called = []

        def on_ready(model_name):
            callback_called.append(model_name)

        prefetcher.on_ready('test_model', on_ready)

        # Manually set status to ready and fire callbacks
        prefetcher._status['test_model'] = PrefetchStatus.READY
        prefetcher._fire_ready_callbacks('test_model')

        assert 'test_model' in callback_called

    def test_prefetcher_shutdown(self):
        """Test prefetcher shutdown."""
        arena = PinnedMemoryArena(size_gb=0.1)
        prefetcher = ModelPrefetcher(arena)

        # Should not raise
        prefetcher.shutdown(wait=False)


class TestPrefetchConfig:
    """Tests for PrefetchConfig."""

    def test_default_config(self):
        """Test default prefetch configuration."""
        config = PrefetchConfig()

        assert config.enabled is True
        assert config.arena_size_gb == 80.0
        assert config.trigger_on_queue_entry is True
        assert config.evict_lru_on_full is True

    def test_custom_config(self):
        """Test custom prefetch configuration."""
        config = PrefetchConfig(
            enabled=False,
            arena_size_gb=40.0,
            trigger_on_queue_entry=False,
        )

        assert config.enabled is False
        assert config.arena_size_gb == 40.0
        assert config.trigger_on_queue_entry is False

    def test_blitz_infer_config_with_prefetch(self):
        """Test BlitzInferConfig includes prefetch settings."""
        config = BlitzInferConfig()

        assert hasattr(config, 'prefetch')
        assert isinstance(config.prefetch, PrefetchConfig)
        assert config.prefetch.enabled is True


class TestPageCacheWarmer:
    """Tests for PageCacheWarmer class."""

    def test_warmer_creation(self):
        """Test basic warmer creation."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer, WarmStatus

        warmer = PageCacheWarmer()
        stats = warmer.get_stats()
        assert stats['warm_count'] == 0
        assert stats['warming_count'] == 0
        warmer.shutdown()

    def test_warm_status_tracking(self):
        """Test that warm status is tracked correctly."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer, WarmStatus

        warmer = PageCacheWarmer()

        # Unknown model should be COLD
        assert warmer.get_status('unknown_model') == WarmStatus.COLD
        assert not warmer.is_warm('unknown_model')
        warmer.shutdown()

    def test_warm_test_file(self):
        """Test warming a test file."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer, WarmStatus

        # Create a test safetensor-like file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake safetensor file (just needs .safetensors extension)
            test_file = Path(tmpdir) / "model.safetensors"
            test_data = b'\x00' * (1024 * 1024)  # 1MB
            test_file.write_bytes(test_data)

            warmer = PageCacheWarmer()
            warmer.register_model("test_model", tmpdir)

            # Start warming
            success = warmer.start_warming("test_model")
            assert success

            # Wait for completion
            warmer.wait_for_warm("test_model", timeout=10)

            assert warmer.is_warm("test_model")
            assert warmer.get_progress("test_model") == 1.0

            warmer.shutdown()

    def test_warm_callback(self):
        """Test that callbacks are fired when model is warm."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer, WarmStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.safetensors"
            test_file.write_bytes(b'\x00' * 1024)

            warmer = PageCacheWarmer()
            warmer.register_model("test_model", tmpdir)

            callback_called = []

            def on_warm(model_name):
                callback_called.append(model_name)

            warmer.on_warm("test_model", on_warm)
            warmer.start_warming("test_model")
            warmer.wait_for_warm("test_model", timeout=10)

            assert "test_model" in callback_called
            warmer.shutdown()

    def test_mark_cold(self):
        """Test marking a model as cold."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer, WarmStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.safetensors"
            test_file.write_bytes(b'\x00' * 1024)

            warmer = PageCacheWarmer()
            warmer.register_model("test_model", tmpdir)
            warmer.start_warming("test_model")
            warmer.wait_for_warm("test_model", timeout=10)

            assert warmer.is_warm("test_model")

            warmer.mark_cold("test_model")
            assert warmer.get_status("test_model") == WarmStatus.COLD
            assert not warmer.is_warm("test_model")

            warmer.shutdown()


class TestCacheWarmerPerformance:
    """Performance tests for page cache warming."""

    @pytest.mark.slow
    def test_cache_warming_speed(self):
        """Test that cache warming achieves reasonable read speed."""
        from blitzinfer.memory.cache_warmer import PageCacheWarmer

        # Create a 100MB test file
        test_size = 100 * 1024**2
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.safetensors"
            test_file.write_bytes(b'\x00' * test_size)

            warmer = PageCacheWarmer()
            warmer.register_model("test_model", tmpdir)

            # Time the warming
            start = time.time()
            warmer.start_warming("test_model")
            warmer.wait_for_warm("test_model", timeout=30)
            elapsed = time.time() - start

            speed_gbps = (test_size / 1024**3) / elapsed

            print(f"\nCache warming speed: {speed_gbps:.2f} GB/s")

            # Should achieve at least 0.5 GB/s (very conservative for CI)
            assert speed_gbps > 0.5, f"Warming too slow: {speed_gbps:.2f} GB/s"

            warmer.shutdown()


class TestArenaPerformance:
    """Performance tests for the arena."""

    @pytest.mark.slow
    def test_arena_read_speed(self):
        """Test that arena achieves expected read speed."""
        # Create a larger arena for meaningful timing
        arena = PinnedMemoryArena(size_gb=0.5)  # 500MB

        # Create a 100MB test file
        test_size = 100 * 1024**2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'\x00' * test_size)
            temp_path = f.name

        try:
            arena.allocate("test_model", test_size)

            # Time the read
            start = time.time()
            bytes_read = arena.read_file_into(temp_path, 0)
            elapsed = time.time() - start

            speed_gbps = (bytes_read / 1024**3) / elapsed

            print(f"\nArena read speed: {speed_gbps:.2f} GB/s")

            # Should achieve at least 1 GB/s (conservative for CI)
            assert speed_gbps > 1.0, f"Read speed too slow: {speed_gbps:.2f} GB/s"

        finally:
            os.unlink(temp_path)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pinned_to_gpu_transfer_speed(self):
        """Test that pinned memory achieves high GPU transfer speed."""
        arena = PinnedMemoryArena(size_gb=0.5)  # 500MB

        # Allocate and fill with test data
        test_size = 100 * 1024**2  # 100MB
        arena.allocate("test_model", test_size)

        # Get a tensor view
        tensor_view = arena.get_slice(0, test_size).view(torch.float32)

        # Time the transfer
        torch.cuda.synchronize()
        start = time.time()
        gpu_tensor = tensor_view.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        speed_gbps = (test_size / 1024**3) / elapsed

        print(f"\nPinned->GPU transfer speed: {speed_gbps:.2f} GB/s")

        # Should achieve at least 10 GB/s for pinned memory
        assert speed_gbps > 10.0, f"Transfer speed too slow: {speed_gbps:.2f} GB/s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
