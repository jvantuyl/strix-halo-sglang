#!/usr/bin/env python3
"""gfx1151 patch 20: decode QSA block selection honours the HIP top-k guard.

Patch 11 keeps the JIT `fast_topk` kernel off ROCm inside `qsa_fast_topk`
(it reads out of bounds on RDNA 3.5), but `QSAIndexer.select_decode_tokens`
calls the JIT kernel directly when the block budget is 512, so every decode
step of Qwen3.8-Flash-Next ran it anyway. Besides the safety concern, the
kernel's output *order* is not deterministic once a row has more than 512
compressed blocks (contexts above ~2k tokens): the selected set is right,
but the QSA decode kernel sums the selected blocks in index order, so long
context greedy decode drifted run to run even after patch 18.

The reference `_qsa_fixed_width_topk` cannot replace it there: it is a
Python loop with a host sync per row, which the decode CUDA graphs cannot
capture. Add `qsa_dense_topk`, a vectorised fixed-width top-k over the
already -inf-padded decode logits (mask beyond each row's length,
`torch.topk`, -1 where the value is -inf, pad to the budget), and use it on
HIP for decode. It matches the reference set exactly, is deterministic,
captures into graphs, and costs 65-130 us per QSA layer at bs 1-20 against
6-22 us for the JIT (12 layers: ~1-1.5 ms on a ~66 ms decode step).
SGLANG_QSA_TOPK_JIT=1 (patch 11's switch) restores the JIT kernel. CUDA is
untouched; the guard is shared with patch 11 through `qsa_jit_topk_allowed`.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# ---- kernel.py: shared guard + dense top-k ---------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/kernel.py"
text = open(p).read()

old_guard = """            if not (
                torch.version.hip is not None
                and os.environ.get("SGLANG_QSA_TOPK_JIT", "0") != "1"
            ):
                try:
                    from sglang.kernels.ops.elementwise.fast_topk import fast_topk
"""
assert text.count(old_guard) == 1, "qsa/kernel.py: patch 11 guard anchor not found"
text = text.replace(
    old_guard,
    """            if qsa_jit_topk_allowed():
                try:
                    from sglang.kernels.ops.elementwise.fast_topk import fast_topk
""",
    1,
)

anchor = "\n\ndef qsa_fast_topk(\n"
assert text.count(anchor) == 1, "qsa/kernel.py: qsa_fast_topk anchor not found"
helpers = '''

def qsa_jit_topk_allowed() -> bool:
    """gfx1151 patch 11/20: the JIT fast_topk kernel is unsafe on RDNA 3.5 and
    its output order is not deterministic past 512 blocks; keep it off HIP
    unless SGLANG_QSA_TOPK_JIT=1."""
    return not (
        torch.version.hip is not None
        and os.environ.get("SGLANG_QSA_TOPK_JIT", "0") != "1"
    )


def qsa_dense_topk(
    logits: torch.Tensor, lengths: torch.Tensor, topk: int
) -> torch.Tensor:
    """Fixed-width top-k over dense (row-start 0) logits, graph-capturable.

    Positions at or beyond a row's length are excluded; rows shorter than
    ``topk`` pad with -1. Indices come out in descending-score order, so the
    result is deterministic for distinct scores.
    """
    width = logits.shape[1]
    k = min(topk, width)
    positions = torch.arange(width, device=logits.device).unsqueeze(0)
    masked = logits.masked_fill(
        positions >= lengths.to(device=logits.device, dtype=torch.int64).unsqueeze(1),
        float("-inf"),
    )
    values, indices = torch.topk(masked, k, dim=1)
    out = torch.where(
        values == float("-inf"), torch.full_like(indices, -1), indices
    ).to(torch.int32)
    if k < topk:
        out = torch.cat([out, out.new_full((out.shape[0], topk - k), -1)], dim=1)
    return out
'''
text = text.replace(anchor, helpers + anchor, 1)
open(p, "w").write(text)
print("patched", p)

# ---- qsa_indexer.py: decode path ------------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/qsa_indexer.py"
text = open(p).read()

old_imp = """from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_fast_topk,
)
"""
assert text.count(old_imp) == 1, "qsa_indexer.py: import anchor not found"
text = text.replace(
    old_imp,
    """from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_dense_topk,
    qsa_fast_topk,
    qsa_jit_topk_allowed,
)
""",
    1,
)

old_sel = """        if logits.is_cuda and self.block_topk == 512:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.elementwise.fast_topk import fast_topk

            block_indices = fast_topk(
                logits,
                compressed_lengths.to(torch.int32),
                topk=self.block_topk,
                row_starts=None,
            )
        else:
"""
assert text.count(old_sel) == 1, "qsa_indexer.py: select_decode_tokens anchor not found"
text = text.replace(
    old_sel,
    """        if logits.is_cuda and self.block_topk == 512 and qsa_jit_topk_allowed():
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.elementwise.fast_topk import fast_topk

            block_indices = fast_topk(
                logits,
                compressed_lengths.to(torch.int32),
                topk=self.block_topk,
                row_starts=None,
            )
        elif logits.is_cuda and torch.version.hip is not None:
            # gfx1151 patch 20: graph-capturable, deterministic replacement for
            # the JIT kernel (the reference loop syncs per row).
            block_indices = qsa_dense_topk(logits, compressed_lengths, self.block_topk)
        else:
""",
    1,
)
open(p, "w").write(text)
print("patched", p)
print("patch 20 (qsa decode topk) applied")
