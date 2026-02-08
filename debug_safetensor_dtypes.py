#!/usr/bin/env python3
"""Debug safetensor dtype mapping issues."""

import json
import struct
from pathlib import Path
from collections import Counter
from huggingface_hub import snapshot_download

def parse_safetensor_header(file_path: str):
    with open(file_path, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
    return header_size, header


# Check Qwen FP8 model
model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
model_path = Path(snapshot_download(model_name, local_files_only=True))

print(f"Model: {model_name}")
print(f"Path: {model_path}")
print()

# Collect all unique dtypes
dtype_counts = Counter()
dtype_examples = {}

sf_files = sorted(model_path.glob('*.safetensors'))
print(f"Found {len(sf_files)} safetensor files")

for sf_file in sf_files[:2]:  # Just check first 2 files
    print(f"\n=== {sf_file.name} ===")
    header_size, header = parse_safetensor_header(str(sf_file))

    for name, info in header.items():
        if name == '__metadata__':
            continue
        dtype = info['dtype']
        shape = tuple(info['shape'])
        data_start, data_end = info['data_offsets']
        size_bytes = data_end - data_start

        dtype_counts[dtype] += 1
        if dtype not in dtype_examples:
            dtype_examples[dtype] = []
        if len(dtype_examples[dtype]) < 3:
            num_elements = 1
            for s in shape:
                num_elements *= s
            bytes_per_elem = size_bytes / num_elements if num_elements > 0 else 0
            dtype_examples[dtype].append({
                'name': name,
                'shape': shape,
                'size_bytes': size_bytes,
                'num_elements': num_elements,
                'bytes_per_elem': bytes_per_elem,
            })

print("\n" + "=" * 80)
print("DTYPE SUMMARY")
print("=" * 80)

for dtype, count in sorted(dtype_counts.items()):
    print(f"\n{dtype}: {count} tensors")
    for ex in dtype_examples[dtype]:
        print(f"  {ex['name']}: {ex['shape']}")
        print(f"    size: {ex['size_bytes']} bytes, elements: {ex['num_elements']}")
        print(f"    bytes/element: {ex['bytes_per_elem']:.2f}")

print("\n\nKnown torch dtypes and their sizes:")
import torch
known_dtypes = {
    'F64': (torch.float64, 8),
    'F32': (torch.float32, 4),
    'F16': (torch.float16, 2),
    'BF16': (torch.bfloat16, 2),
    'I64': (torch.int64, 8),
    'I32': (torch.int32, 4),
    'I16': (torch.int16, 2),
    'I8': (torch.int8, 1),
    'U8': (torch.uint8, 1),
}
for name, (dtype, size) in known_dtypes.items():
    print(f"  {name}: {dtype} ({size} bytes)")

# Check if torch has FP8 dtypes
print("\n\nChecking torch FP8 dtypes:")
fp8_dtypes = [
    'float8_e4m3fn',
    'float8_e5m2',
    'float8_e4m3fnuz',
    'float8_e5m2fnuz',
]
for name in fp8_dtypes:
    if hasattr(torch, name):
        dtype = getattr(torch, name)
        print(f"  torch.{name}: {dtype}")
        # Get size
        t = torch.tensor([1.0], dtype=dtype)
        print(f"    element size: {t.element_size()} bytes")
