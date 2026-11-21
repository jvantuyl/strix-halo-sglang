#!/usr/bin/env python3
"""Check patch 39: QSA spec metadata builds from host-known lengths (no GPU).

Runs in the image without a GPU: exercises _speculative_row_to_request on
CPU tensors and asserts the host-list path produces the same mapping as the
device-tensor fallback path, plus that the hot branch no longer calls
.item() on a device sum.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch  # noqa: E402

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (  # noqa: E402
    QwenSparseAttnBackend,
)

build = QwenSparseAttnBackend._speculative_row_to_request
import inspect  # noqa: E402

src = inspect.getsource(build)

# The hot branch must read the published CPU lengths, not a device sum.
assert "extend_seq_lens_cpu" in src, "host-list fast path missing"
device_branch = src.split("extend_seq_lens = getattr", 1)[1]
assert ".item()" in device_branch, "device fallback lost its shape"
host_branch = src.split("extend_seq_lens = getattr", 1)[0]
assert "sum().item()" not in host_branch, "host path still syncs (.item() present)"


def ref_mapping(repeats, num_rows, batch):
    rows = [r for req in range(batch) for r in [req] * repeats[req]]
    pad = num_rows - len(rows)
    return rows + [0] * pad


cases = [
    ([4], 16),           # decode-like draft-extend, bs=1, 12 padded rows
    ([1024], 1028),      # prefill chunk draft-extend shape
    ([2, 3, 1], 16),     # multi-request fan-out
    ([0, 5], 16),        # a request with no extend rows
]
for extends, num_rows in cases:
    batch = len(extends)
    fb_host = SimpleNamespace(
        req_pool_indices=torch.arange(batch),
        extend_seq_lens=torch.tensor(extends),
        extend_seq_lens_cpu=list(extends),
    )
    got = build(fb_host, num_rows)
    want = torch.tensor(ref_mapping(extends, num_rows, batch), dtype=torch.long)
    assert torch.equal(got, want), f"host path mismatch for {extends}: {got}"

    # Fallback path (no published CPU list) must agree exactly.
    fb_dev = SimpleNamespace(
        req_pool_indices=torch.arange(batch),
        extend_seq_lens=torch.tensor(extends),
        extend_seq_lens_cpu=None,
    )
    got_fb = build(fb_dev, num_rows)
    assert torch.equal(got, got_fb), f"paths disagree for {extends}"

# num_rows smaller than the mapping must still raise on the host path.
fb = SimpleNamespace(
    req_pool_indices=torch.arange(1),
    extend_seq_lens=torch.tensor([4]),
    extend_seq_lens_cpu=[4],
)
try:
    build(fb, 3)
    raise SystemExit("expected ValueError for rows < mapped")
except ValueError:
    pass

print("check_qsa_host_metadata: ok")
