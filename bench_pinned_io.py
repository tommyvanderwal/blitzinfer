"""Benchmark: O_DIRECT into regular vs mlock'd (pinned) memory."""
import os, time, ctypes, ctypes.util, mmap
from concurrent.futures import ThreadPoolExecutor, as_completed

O_DIRECT = 0o40000
BLOCK_ALIGN = 4096

BLOB = "/home/tommy/.cache/huggingface/hub/models--openai--gpt-oss-120b/blobs/68a8dc1f8e2e5996cb702f14332a25ddf3463daeab2df68e21ca09ef181203c3"
SIZE = os.path.getsize(BLOB)
print(f"File: {SIZE/1e9:.2f} GB")

libc = ctypes.CDLL(ctypes.util.find_library('c'))

def read_direct(fp, foff, size, ptr):
    fd = os.open(fp, os.O_RDONLY | O_DIRECT)
    try:
        os.lseek(fd, foff, os.SEEK_SET)
        t = 0
        while t < size:
            to_read = min(64*1024*1024, size - t)
            to_read = (to_read // BLOCK_ALIGN) * BLOCK_ALIGN
            if to_read == 0: break
            v = (ctypes.c_char * to_read).from_address(ptr + t)
            n = os.readv(fd, [v])
            if not n: break
            t += n
        return t
    finally:
        os.close(fd)

aligned_size = (SIZE // BLOCK_ALIGN) * BLOCK_ALIGN
chunk_bytes = 2 * 1024**3

# Build chunk list
chunks = []
pos = 0
while pos < aligned_size:
    cs = min(chunk_bytes, aligned_size - pos)
    chunks.append((pos, cs))
    pos += cs

def run_test(name, ptr, workers=16):
    os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")
    time.sleep(0.5)

    t0 = time.perf_counter()
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(read_direct, BLOB, foff, cs, ptr + foff) for foff, cs in chunks]
        for f in as_completed(futs):
            total += f.result()
    t1 = time.perf_counter()
    speed = total / 1e9 / (t1 - t0)
    print(f"  {name}: {total/1e9:.1f}GB in {t1-t0:.2f}s = {speed:.1f} GB/s (workers={workers})")

# Test 1: Regular mmap
print("\n--- Regular mmap ---")
buf1 = mmap.mmap(-1, SIZE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
ptr1 = ctypes.addressof(ctypes.c_char.from_buffer(buf1))
run_test("regular", ptr1, 8)
run_test("regular", ptr1, 16)

# Test 2: mlock'd mmap (simulates CUDA pinned)
print("\n--- mlock'd mmap (simulating CUDA pinned) ---")
buf2 = mmap.mmap(-1, SIZE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
ptr2 = ctypes.addressof(ctypes.c_char.from_buffer(buf2))
ret = libc.mlock(ptr2, SIZE)
print(f"  mlock result: {ret} (0=success)")
run_test("mlocked", ptr2, 8)
run_test("mlocked", ptr2, 16)

buf1.close()
libc.munlock(ptr2, SIZE)
buf2.close()

# Test 3: If CUDA available, test actual CUDA pinned
try:
    import torch
    if torch.cuda.is_available():
        print("\n--- CUDA pinned memory ---")
        pint = torch.empty(SIZE, dtype=torch.uint8, pin_memory=True)
        ptr3 = pint.data_ptr()
        print(f"  Addr: 0x{ptr3:x}, aligned: {ptr3 % 4096 == 0}")
        run_test("cuda_pinned", ptr3, 8)
        run_test("cuda_pinned", ptr3, 16)
        del pint
except Exception as e:
    print(f"\nCUDA test skipped: {e}")
