#!/usr/bin/env python3
"""gfx1151 patch 21: tie-stable QSA block selection on ROCm.

The QSA indexer scores each compressed block with `relu(q.k).sum(heads)`, so
many blocks score exactly 0.0 and ties are the norm, not the exception. On
this ROCm torch build `torch.topk` returns tied indices in an order that
changes from launch to launch (and when a tie straddles the k boundary, a
different set). Both QSA selection paths on HIP go through it: the prefill
reference `_qsa_fixed_width_topk` (per-row `torch.topk`) and patch 20's
decode `qsa_dense_topk`. The sparse attention kernels sum the selected blocks
in the order given, so prefill logits above ~1.4k tokens (and decode after
patch 20) still differed bit-wise between identical runs; past 512 scored
blocks the token choice itself could move.

`torch.sort(stable=True)` and `torch.topk` over distinct keys are both
bit-stable here, so:

* `qsa_ordered_topk(masked, k, use_sort)`: top-k indices per row, descending
  score, ties broken toward the lower block index. `use_sort=True` is a
  stable descending sort (fastest for many narrow rows: prefill);
  `use_sort=False` packs an order-preserving int32 image of the score with
  the reversed index into one int64 key and runs `torch.topk` on the unique
  keys (fastest for a few 32k-wide rows: decode). Both give the same
  indices; signed zeros are folded so they tie.
* `qsa_dense_topk` (decode, patch 20) selects with the key form.
* `qsa_stable_rows_topk(logits, lengths, starts, topk)`: vectorised
  `_qsa_fixed_width_topk` (indices relative to each row's start, -1 padding,
  same width rule) using the sort form. `qsa_fast_topk` takes it on HIP in
  place of the Python per-row loop, which also removes a host sync per query
  row from prefill (119 ms per QSA layer at 1475 rows before, ~0.2 ms after).

CUDA and CPU paths are unchanged. Anchors on patch 20's text.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qsa/kernel.py"
text = open(p).read()

# ---- helpers, inserted before patch 20's guard ------------------------------
anchor = "\n\ndef qsa_jit_topk_allowed() -> bool:\n"
assert text.count(anchor) == 1, "qsa/kernel.py: patch 20 guard anchor not found"
helpers = '''

def qsa_ordered_topk(
    masked: torch.Tensor, k: int, *, use_sort: bool
) -> torch.Tensor:
    """gfx1151 patch 21: per-row top-k indices in descending-score order with
    ties broken toward the lower index, bit-stable on ROCm.

    ``torch.topk`` orders tied entries arbitrarily from launch to launch on
    this platform, and QSA scores (relu sums) tie constantly. ``use_sort``
    picks a stable descending sort (cheap for many narrow rows); otherwise the
    score is mapped to an order-preserving int32 image and packed with the
    reversed index into an int64 key so that ``torch.topk`` sees distinct keys
    (cheap for a few wide rows). Both return the same indices.
    """
    if use_sort:
        return torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :k]
    width = masked.shape[1]
    # -0.0 + 0.0 == +0.0: fold signed zeros so they tie like the sort path.
    scores = (masked.float() + 0.0).contiguous()
    bits = scores.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    reversed_index = torch.arange(
        width - 1, -1, -1, device=masked.device, dtype=torch.int64
    )
    keys = (ordered << 32) | reversed_index.unsqueeze(0)
    return torch.topk(keys, k, dim=1).indices


def qsa_stable_rows_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """gfx1151 patch 21: vectorised ``_qsa_fixed_width_topk``.

    Same contract (top-k of ``logits[row, start:start+length]`` as indices
    relative to ``start``, ``min(length, topk)`` valid entries, -1 padding)
    without the per-row host sync, and tie-stable via ``qsa_ordered_topk``.
    """
    rows, width = logits.shape
    k = min(topk, width)
    device = logits.device
    starts = starts.to(device=device, dtype=torch.int64).unsqueeze(1)
    lengths = lengths.to(device=device, dtype=torch.int64).unsqueeze(1)
    columns = torch.arange(width, device=device).unsqueeze(0)
    inside = (columns >= starts) & (columns < starts + lengths)
    order = qsa_ordered_topk(
        logits.masked_fill(~inside, float("-inf")), k, use_sort=True
    )
    keep = torch.arange(k, device=device).unsqueeze(0) < lengths.clamp_max(k)
    out = torch.where(keep, order - starts, torch.full_like(order, -1)).to(
        torch.int32
    )
    if k < topk:
        out = torch.cat([out, out.new_full((rows, topk - k), -1)], dim=1)
    return out
'''
text = text.replace(anchor, helpers + anchor, 1)

# ---- qsa_dense_topk (patch 20): tie-stable selection -------------------------
old_dense = """    Positions at or beyond a row's length are excluded; rows shorter than
    ``topk`` pad with -1. Indices come out in descending-score order, so the
    result is deterministic for distinct scores.
    \"\"\""""
assert text.count(old_dense) == 1, "qsa/kernel.py: qsa_dense_topk docstring anchor not found"
text = text.replace(
    old_dense,
    """    Positions at or beyond a row's length are excluded; rows shorter than
    ``topk`` pad with -1. Indices come out in descending-score order with ties
    broken toward the lower block (patch 21), so the result is bit-stable.
    \"\"\"""",
    1,
)

old_topk = """    values, indices = torch.topk(masked, k, dim=1)
    out = torch.where(
        values == float("-inf"), torch.full_like(indices, -1), indices
    ).to(torch.int32)
"""
assert text.count(old_topk) == 1, "qsa/kernel.py: qsa_dense_topk torch.topk anchor not found"
text = text.replace(
    old_topk,
    """    indices = qsa_ordered_topk(masked, k, use_sort=False)
    values = torch.gather(masked, 1, indices)
    out = torch.where(
        values == float("-inf"), torch.full_like(indices, -1), indices
    ).to(torch.int32)
""",
    1,
)

# ---- qsa_fast_topk: HIP takes the vectorised tie-stable path ------------------
old_ref = """        if topk in supported_topk:
            return top_k_module.fast_topk_v2(
                logits, lengths, topk=topk, row_starts=starts
            )
        return _qsa_fixed_width_topk(logits, lengths, starts, topk)
"""
assert text.count(old_ref) == 1, "qsa/kernel.py: qsa_fast_topk reference fall-through anchor not found"
text = text.replace(
    old_ref,
    """        if topk in supported_topk:
            return top_k_module.fast_topk_v2(
                logits, lengths, topk=topk, row_starts=starts
            )
        if torch.version.hip is not None:
            # gfx1151 patch 21: torch.topk's tie order varies per launch here;
            # the vectorised stable path is also free of the per-row sync.
            return qsa_stable_rows_topk(logits, lengths, starts, topk)
        return _qsa_fixed_width_topk(logits, lengths, starts, topk)
""",
    1,
)

old_all = '''    "qsa_fast_topk",
'''
assert text.count(old_all) == 1, "qsa/kernel.py: __all__ anchor not found"
text = text.replace(
    old_all,
    '''    "qsa_fast_topk",
    "qsa_ordered_topk",
    "qsa_stable_rows_topk",
''',
    1,
)

open(p, "w").write(text)
print("patched", p)
print("patch 21 (qsa topk ties) applied")
