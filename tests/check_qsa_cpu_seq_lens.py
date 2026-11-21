#!/usr/bin/env python3
"""Check patch 37: the QSA backend opts into seq_lens_cpu (no GPU).

Runs in the image without a GPU: imports the patched module and asserts the
class attribute, plus that the sync-free branch exists next to the fallback.
"""
import inspect
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (  # noqa: E402
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)

assert QwenSparseAttnBackend.needs_cpu_seq_lens is True, (
    "QwenSparseAttnBackend.needs_cpu_seq_lens should be True after patch 37"
)
# The draft backend keeps its opt-out (draft decode steps never read it).
assert QwenSparseMultiStepDraftBackend.needs_cpu_seq_lens is False

src = inspect.getsource(QwenSparseAttnBackend._speculative_max_row_length)
assert "seq_lens_cpu.max()" in src, "sync-free branch missing"
assert "sequence_lengths.max()" in src, "device fallback missing"

print("check_qsa_cpu_seq_lens: ok")
