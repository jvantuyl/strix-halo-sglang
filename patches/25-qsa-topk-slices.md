# Patch 25: row-sliced prefill block selection

**Files:** `python/sglang/srt/layers/attention/qsa/kernel.py`
**Script:** [`patch_qsa_topk_slices.py`](patch_qsa_topk_slices.py)

## Symptom

With MTP at the launcher's default 20-request cap the server idles at 90.1
GiB of the 96 GiB VRAM (weights 75.9, MTP layer 0.2, KV 3.05, GDN pools 9.4,
graphs 0.6, all accounted for by the load log). A 43k-token prefill added
2.6 GiB on top, 20 decoding streams another 0.9, and 18 streams plus two
43k prefills arriving together left 1.2 GiB free (`amdgpu_top`, driver
view). The prefill transient grew with prompt length up to ~26k tokens and
then plateaued.

## Cause

An allocator snapshot (`/start_profile` with `activities: ["MEM"]`, a 43k
prefill, `/stop_profile`) put 2,530 MiB of tensors live at the peak, and
1,024 MiB of them in one call: patch 21's `qsa_stable_rows_topk`, which
sorted the whole 8,192-row prefill chunk in one `torch.sort(stable=True)`.
That materialises a fp32 copy of the masked logits, the fp32 sorted values
and int64 indices of the same shape, and the boolean mask: with a few
thousand compressed blocks per row, about 1 GiB per QSA layer. torch's own
peak counter for the stage (which also sees the temporaries the snapshot
does not attribute) read 2,368 MiB at 8,192 × 8,192.

## Fix

The selection is per row, so the sort runs over row slices sized to a fixed
element budget (`QSA_TOPK_SLICE_ELEMENTS = 4 Mi` logits, ~70 MiB of sort
buffers) and writes into a preallocated int32 output. Identical indices bit
for bit; 8–16 sort launches per layer instead of one (prefill is eager, no
graph to break), no measurable time difference.

## Verification

`tools/test_qwen38_rocm.py` item 14: sliced output equals the whole-chunk
sort on an 8,192 × 6,656 chunk with tied relu-style scores (14 slices) and on
shapes whose rows do not divide into slices, selects the fixed-width
reference's scores, is bit-identical over 20 launches, and its peak
allocation is under a third of the whole-chunk sort's (measured 1,924 →
246 MiB). Standalone comparison against the unpatched function: identical
on 8 shapes, 2,368 → 245 MiB at 8,192 × 8,192, 31.0 → 29.2 ms.

End to end (DERISKED, MTP, cap 20): greedy repeats identical with the same
accept histograms as before the patch; 20 streams 99.0 tok/s (101–103
before, noise). Driver-level prefill transient at 99k tokens 2,690 → 2,388
MiB and the mixed worst case 1.2 → 1.5 GiB free: less than the stage's own
saving, because the next snapshot showed the peak had moved to the prefill
MQA reference ([patch 26](26-qsa-mqa-prefill-triton.md)).
