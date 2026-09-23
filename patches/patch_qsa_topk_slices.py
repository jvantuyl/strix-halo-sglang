#!/usr/bin/env python3
"""gfx1151 patch 25: row-sliced prefill block selection.

Patch 21's `qsa_stable_rows_topk` sorts the whole prefill chunk in one
`torch.sort(stable=True)` call: with the 8192-token chunk and a few thousand
compressed blocks per row that is a fp32 copy of the masked logits, the fp32
sorted values and int64 indices of the same shape, plus the boolean mask,
about 1 GiB live at once per QSA layer. An allocator snapshot of a 43k-token
prefill on this box put 1,024 MiB of the 2,530 MiB transient peak in that
one call; the rest is ordinary per-chunk activations. The peak sits on top
of ~90 GiB of resident weights and pools, and with 20 requests decoding it
left 1.2 GiB of the 96 GiB VRAM free.

The selection is per row, so the sort is applied to slices of rows sized to
a fixed element budget (`QSA_TOPK_SLICE_ELEMENTS`) and written into a
preallocated int32 output. Same indices bit for bit, the transient of this
stage bounded at ~70 MiB regardless of chunk or context length, a handful of
extra sort launches per layer (prefill is eager, no graph to break).
Anchors on patch 21's text.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qsa/kernel.py"
text = open(p).read()

old = '''def qsa_stable_rows_topk(
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
assert text.count(old) == 1, "qsa/kernel.py: patch 21 qsa_stable_rows_topk not found"

new = '''# gfx1151 patch 25: logits elements per stable-sort slice in
# qsa_stable_rows_topk. The sort materialises a fp32 copy, fp32 values and
# int64 indices of the slice, so 4 Mi elements is ~70 MiB live per layer
# instead of ~1 GiB for a whole 8192-row chunk.
QSA_TOPK_SLICE_ELEMENTS = 4 << 20


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

    gfx1151 patch 25: rows are processed in slices of about
    ``QSA_TOPK_SLICE_ELEMENTS`` logits each, so the sort's transient buffers
    stay bounded whatever the chunk size and block count. The selection is
    per row, so the result is identical to sorting the chunk in one call.
    """
    rows, width = logits.shape
    k = min(topk, width)
    device = logits.device
    starts = starts.to(device=device, dtype=torch.int64).unsqueeze(1)
    lengths = lengths.to(device=device, dtype=torch.int64).unsqueeze(1)
    columns = torch.arange(width, device=device).unsqueeze(0)
    keep_columns = torch.arange(k, device=device).unsqueeze(0)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    step = max(1, QSA_TOPK_SLICE_ELEMENTS // max(width, 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        s = starts[r0:r1]
        n = lengths[r0:r1]
        inside = (columns >= s) & (columns < s + n)
        order = qsa_ordered_topk(
            logits[r0:r1].masked_fill(~inside, float("-inf")), k, use_sort=True
        )
        keep = keep_columns < n.clamp_max(k)
        out[r0:r1, :k] = torch.where(
            keep, order - s, torch.full_like(order, -1)
        ).to(torch.int32)
    return out
'''
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 25 (qsa topk slices) applied")
