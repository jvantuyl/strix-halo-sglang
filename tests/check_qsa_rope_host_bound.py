#!/usr/bin/env python3
"""Check patch 40: QSA rope cache grows from a host bound, no per-call sync.

Runs in the image without a GPU: source-level asserts on the three touched
files plus a CPU behavioral test of the holder/monotonicity/fallback.
"""
import inspect
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch  # noqa: E402

from sglang.srt.layers.attention.qsa import metadata as qsa_metadata  # noqa: E402
from sglang.srt.layers.attention.qsa import qsa_indexer  # noqa: E402
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend  # noqa: E402

# Holder + helper exist and behave.
holder = qsa_metadata.QSA_HOST_MAX_POSITION
record = qsa_metadata.record_qsa_host_max_position
bound_fn = qsa_metadata._host_position_bound

saved = holder[0]
try:
    holder[0] = 0
    positions = torch.tensor([5, 9, 2])
    assert bound_fn(positions) == 9, "fallback must read the device positions"
    record(torch.tensor([100, 250]), draft_window=4)
    assert holder[0] == 254, "bound must be max + draft window"
    assert bound_fn(positions) == 254, "bound must take precedence over readback"
    record(torch.tensor([50]), draft_window=0)
    assert holder[0] == 254, "bound must be monotone"
    record([7, 3], draft_window=1)
    assert holder[0] == 254
    record(None, draft_window=1)
    assert holder[0] == 254, "None must be a no-op"
finally:
    holder[0] = saved

# Indexer: both ensure sites read the bound; no device readback remains.
idx_src = inspect.getsource(qsa_indexer)
assert "positions.max().item()" not in idx_src, "indexer still syncs per call"
assert idx_src.count("_host_position_bound") >= 3, "helper not used at both sites"

# Backend: the bound is recorded on both serving entry points, and the
# graph-replay path (which has no forward_batch in scope) must not have
# gained a reference to it — the NameError class.
backend_src = inspect.getsource(qsa_backend)
assert backend_src.count("record_qsa_host_max_position") >= 3, (
    "backend must record the bound in both metadata paths"
)
replay_src = inspect.getsource(
    qsa_backend.QwenSparseAttnBackend._replay_cuda_graph_metadata
)
assert "forward_batch." not in replay_src and "getattr(forward_batch" not in replay_src, (
    "replay path uses forward_batch, which is not in scope there"
)

print("check_qsa_rope_host_bound: ok")
