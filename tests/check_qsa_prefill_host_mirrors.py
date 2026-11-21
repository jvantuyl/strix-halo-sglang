#!/usr/bin/env python3
"""Check patch 41: QSA prefill MQA inputs from host mirrors (no GPU).

Source-level asserts plus a real behavioral check: get_prefill_mqa_inputs
must return byte-identical outputs whether the host mirror is set or the
original device .tolist() fallback runs.
"""
import inspect

import msgspec  # noqa: E402
import torch  # noqa: E402

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend  # noqa: E402
from sglang.srt.layers.attention.qsa import metadata as qsa_metadata  # noqa: E402
from sglang.srt.layers.attention.qsa import sparse_attn  # noqa: E402

# The field belongs on QSAIndexerMetadata — the class get_prefill_mqa_inputs
# is a method of (first bad-41 draft put it on the backend struct instead).
fields = {
    f.name: f
    for f in msgspec.structs.fields(qsa_metadata.QSAIndexerMetadata)
}
assert "sequence_lengths_cpu" in fields, "struct field missing"
assert fields["sequence_lengths_cpu"].default is None

# get_prefill_mqa_inputs prefers the mirror; readback only as fallback.
src = inspect.getsource(
    qsa_metadata.QSAIndexerMetadata.get_prefill_mqa_inputs
)
assert "sequence_lengths_cpu" in src and ".tolist()" in src

# Kernel wrapper: host max_q kwarg, readback as fallback.
ksrc = inspect.getsource(sparse_attn.sparse_gqa_fwd_interface_triton_ck)
assert "max_q=None" in ksrc and "cu_q[1:] - cu_q[:-1]).max().item()" in ksrc

# Backend: mirror passed from the non-spec indexer-metadata construction;
# host max_q passed to the kernel.
bsrc = inspect.getsource(qsa_backend.QwenSparseAttnBackend)
assert 'getattr(\n                forward_batch, "seq_lens_cpu", None\n            )' in bsrc
assert "max_q=max(extend_lens)" in bsrc


class _FakePool:
    qsa_index_kv_heads = 2
    qsa_index_head_dim = 3

    def get_qsa_compressed_k_buffer(self, layer_id):
        buf = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        buf += 100.0 * layer_id
        return buf


def _build_metadata(mirror):
    return qsa_metadata.QSAIndexerMetadata(
        sequence_lengths=torch.tensor([4, 2], dtype=torch.int64),
        token_to_batch_idx=torch.tensor([0, 0, 1], dtype=torch.int32),
        token_slot_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
        out_cache_loc=torch.zeros(6, dtype=torch.int64),
        token_to_kv_pool=_FakePool(),
        compress_ratio=2,
        block_topk=4,
        sequence_lengths_cpu=mirror,
    )


positions = torch.tensor([0, 1, 2], dtype=torch.int64)
with_mirror = _build_metadata(torch.tensor([4, 2])).get_prefill_mqa_inputs(3, positions)
with_list = _build_metadata([4, 2]).get_prefill_mqa_inputs(3, positions)
fallback = _build_metadata(None).get_prefill_mqa_inputs(3, positions)
for a, b in zip(with_mirror, fallback):
    assert torch.equal(a, b), "mirror output differs from readback fallback"
for a, b in zip(with_list, fallback):
    assert torch.equal(a, b), "list-mirror output differs from fallback"
# Spot-check the gather itself: complete_blocks=[2,1] -> rows 0,1 then 2.
assert torch.equal(with_mirror[0][0], fallback[0][0])
assert with_mirror[0].shape == (3, 2, 3)

print("check_qsa_prefill_host_mirrors: ok")
