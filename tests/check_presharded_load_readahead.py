#!/usr/bin/env python3
"""Check patch 42: presharded reload stages into aligned CPU buffers, one
read ahead of the consumer.

Behavioral: with stubbed staging, file N+1's read is in flight before the
consumer finishes file N, and yields stay in file order. Plus a real
round-trip: staged views read back a freshly written safetensors file
byte-identically for O_DIRECT and buffered modes, use the aligned buffer
(no misaligned O_DIRECT EIO), and survive garbage collection of the
staging internals.
"""
import gc
import os
import tempfile
import threading
import time

import torch
from safetensors.torch import save_file

import sglang.srt.model_loader.loader as L

PML = L.PreshardedModelLoader

SRC = open(L.__file__).read()
assert "os.O_DIRECT" in SRC and "pin_memory=True" not in SRC.split("_SAFETENSORS_DTYPES")[1]
assert "presharded-prefetch" in SRC and "_prefetched_file_reads" in SRC

# --- behavioral: overlap and ordering ------------------------------------
events = []
events_lock = threading.Lock()


def fake_stage(path, keys):
    with events_lock:
        events.append(("read-start", os.path.basename(path)))
    time.sleep(0.25)
    with events_lock:
        events.append(("read-end", os.path.basename(path)))
    return {k: (os.path.basename(path), k) for k in keys}


orig_stage = PML.__dict__["_staged_file_reads"]
PML._staged_file_reads = staticmethod(fake_stage)
try:
    by_file = {
        f"model-{i:05d}-common.safetensor": [
            {"stored_key": f"k{i}a"},
            {"stored_key": f"k{i}b"},
        ]
        for i in range(3)
    }
    gen = PML._prefetched_file_reads(object.__new__(PML), "/dir", by_file)
    order = []
    overlapped = None
    for idx, (filename, cached) in enumerate(gen):
        if idx == 0:
            with events_lock:
                started = [e for e in events if e[0] == "read-start"]
            overlapped = len(started) >= 2
        order.append(filename)
        assert set(cached) == {f"k{idx}a", f"k{idx}b"}
        assert all(v[0] == filename for v in cached.values())
finally:
    PML._staged_file_reads = orig_stage

assert order == list(by_file), f"yield order wrong: {order}"
assert overlapped, "next read did not overlap consumer"

# --- real round-trip: staged views read back correctly -------------------
ref = {
    "a": torch.randn(37, 5).to(torch.bfloat16),
    "b": torch.arange(64, dtype=torch.int32).reshape(4, 16),
    "c": torch.randint(0, 255, (11,), dtype=torch.uint8),
    "d": torch.randn(3, 4, 5),
}
tmp = tempfile.mkdtemp(prefix="presharded-check42-")
path = os.path.join(tmp, "chunk.safetensor")
save_file(ref, path)

self = object.__new__(PML)
for direct in (True, False):
    views = PML._staged_views_from_buffer(path, list(ref), direct=direct)
    assert set(views) == set(ref)
    for k in ref:
        assert views[k].dtype == ref[k].dtype, (k, direct)
        assert views[k].shape == ref[k].shape, (k, direct)
        assert torch.equal(views[k], ref[k].to(views[k].dtype)), (k, direct)
    del views
    gc.collect()

# lifetime: views outliving every staging-internal reference
views = PML._staged_views_from_buffer(path, list(ref), direct=True)
gc.collect()
for k in ref:
    assert torch.equal(views[k], ref[k].to(views[k].dtype)), f"post-gc mismatch {k}"

# --- chunked stage copy: byte-exact vs plain copy_ ------------------------
if torch.cuda.is_available():
    loader = object.__new__(PML)
    stage = torch.empty(PML._COPY_STAGE_BYTES, dtype=torch.uint8, pin_memory=True)
    for shape, dtype in (
        ((37, 5), torch.bfloat16),
        ((4, 16), torch.int32),
        ((11,), torch.uint8),
        ((3, 4, 5), torch.float32),
    ):
        src = torch.randn(shape).to(dtype) if dtype.is_floating_point else torch.randint(0, 255, shape, dtype=dtype)
        gpu_a = torch.empty(shape, dtype=dtype, device="cuda")
        gpu_b = torch.empty(shape, dtype=dtype, device="cuda")
        gpu_a.copy_(src)
        loader._copy_to_gpu(gpu_b, src, stage)
        torch.cuda.synchronize()
        assert torch.equal(gpu_a, gpu_b), f"stage copy mismatch {shape} {dtype}"
    # big tensor spanning several stage chunks
    big = torch.randn(8 * 2**20 // 2).to(torch.bfloat16)  # 8 MiB bf16
    gpu_a = torch.empty(big.shape, dtype=big.dtype, device="cuda")
    gpu_b = torch.empty(big.shape, dtype=big.dtype, device="cuda")
    gpu_a.copy_(big)
    loader._copy_to_gpu(gpu_b, big, stage)
    torch.cuda.synchronize()
    assert torch.equal(gpu_a, gpu_b), "multi-chunk stage copy mismatch"
    # tiny tensors take the plain path; stage=None fallback works
    tiny = torch.randn(4)
    gpu_b2 = torch.empty(4, device="cuda")
    loader._copy_to_gpu(gpu_b2, tiny, None)
    torch.cuda.synchronize()
    assert torch.equal(gpu_b2.cpu(), tiny)

# --- prepark + host-side fill through the parked alias --------------------
if torch.cuda.is_available():
    import torch.nn as nn

    from sglang.srt.model_executor.host_parked_params import (
        is_parked,
        parked_buffer,
        prepark_parameter,
    )

    p = nn.Parameter(torch.zeros(1000, 4, dtype=torch.bfloat16, device="cuda"))
    prepark_parameter(p)
    assert is_parked(p), "preparked param not registered"
    src = torch.randn(1000, 4).to(torch.bfloat16)
    loader = object.__new__(PML)
    loader._copy_to_gpu(p.data, src, None)  # must take the parked branch
    torch.cuda.synchronize()
    assert torch.equal(p.data.cpu(), src), "parked fill mismatch"
    assert parked_buffer(p.data) is not None
    # a non-parked param still goes through the normal paths
    q = nn.Parameter(torch.zeros(3, device="cuda"))
    loader._copy_to_gpu(q.data, torch.tensor([1.0, 2.0, 3.0]), None)
    torch.cuda.synchronize()
    assert torch.equal(q.data.cpu(), torch.tensor([1.0, 2.0, 3.0]))
    assert not is_parked(q)

print("check_presharded_load_readahead: ok")
