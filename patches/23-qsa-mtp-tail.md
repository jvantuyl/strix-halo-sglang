# Patch 23: MTP draft decode attends to the drafted tokens

**Files:** `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_mtp_tail.py`](patch_qsa_mtp_tail.py)

## Symptom

After patch 22, greedy MTP decode of a 14k-token prompt was bit-identical
across cold runs, but a 19-token prompt still was not: identical tokens for
~80 positions, top-1 logprobs differing from position 7 (or 88) on,
`spec_verify_ct` alternating 40 / 41. Traced with the same per-module
checksums (`--disable-cuda-graph`, 3 runs): runs 1 and 2 identical, run 0
differing first at `D model.layers.0.attn`, the draft MTP layer's sparse
attention, in the first draft decode step (bs 1), with identical inputs.
Inside that layer the block selection, the K/V cache slots written and the K
checksums at those slots were identical across runs; the query was
identical; the *packed* K/V handed to the attention kernel differed.

## Cause

Under MTP the draft model's decode steps do not run the QSA indexer. They
reuse the block selection captured at the draft-extend pass
(`QSAMTPSharedSparseIndices`, one row per request: the indexer's 2051
expanded columns plus a tail of `num_steps + 1` columns) and `lookup`
appends the positions drafted since the capture into the tail. It wrote the
tail into the *last* columns of the row, after the captured selection's
`-1` padding, so a short prompt's row read

```
[0, 1, ..., L-1, -1, -1, ..., -1, L, -1, -1, -1]
```

The KV gather that feeds the attention kernel (`_compact_kv` in
`qsa/sparse_attn.py`, used by both the packed FA2 fallback that runs here
and the paged trtllm path) packs each row's valid entries as a *prefix*:
`_fa2_valid_counts` counts the valid entries (`L + 1` above), and column `c`
is packed only if `c < valid_count`. Its docstring says so ("`valid_count`
is a count, not a mask, so a `-1` in the middle of a row would shift the
packing"). Column `L` is a `-1` and is skipped; column 2051 (position `L`)
is past the count and is never read. Packed slot `L` is therefore never
written: on the FA2 path it holds whatever the scratch buffer had from an
earlier request (the paged path zero-fills it). The draft never sees the
newest token, and on this box its output depended on the previous request.

A captured row that is full (prompts long enough for all 2048 top-k tokens
plus the 3-wide uncompressed tail) has no padding, so the tail columns are
inside the prefix and everything works; that is why the 14k-token prompt was
already clean after patch 22.

## Fix

`lookup` places the tail immediately after the captured row's valid entries
(`(captured >= 0).sum()` per row) with a `scatter_`, after resetting the tail
columns to `-1`, so the row stays a valid prefix as the gather expects. The
captured entries are all below `captured_len`, so the tail never collides
with them; a full row still puts the tail in the last columns. Pure tensor
ops on the row copy `lookup` already makes, graph-capturable; the stored
selection is untouched. Not gfx1151-specific: on CUDA the same layout drops
the drafted tokens silently (zero-filled, so deterministic, but the draft
attends to zeros where the newest tokens should be).

## Verification

`tools/test_qwen38_rocm.py` item 12: a short captured row, a full row and a
never-captured row (zeros, `captured_len` 1) looked up at each of 5 draft
steps come back as captured entries, then the drafted positions up to the
current one, then `-1`; the FA2 valid-count kernel's count equals that
prefix length; the stored selection is unchanged; the lookup captures into a
CUDA graph, replays equal to eager and follows the position buffer. The item
fails on the unpatched image (tail after the padding, count `L + 1` while
the prefix has `L` entries).

End to end on the DERISKED checkpoint with patches 22 + 23 (MTP 3 steps / 4
draft tokens, CUDA graphs on, every run cold, top-1 logprobs compared at
every position): 19-token prompt 96 tokens 4/4 identical and 512 tokens
3/3; 2.5k-token prompt 128 tokens 3/3; 14k-token prompt 64 tokens 3/3. The
accept histograms are identical across runs. Mean accept length rose with
the tail visible: 2.55 at 512 tokens (was 2.3–2.8, varying), 3.37 on the
14k prompt (was 3.1). Throughput is unchanged within noise: bs 1
21.9–23.0 / 19.5 / 16.7 / 18.3 tok/s at 183 / 3.4k / 13.5k / 54k prompt
tokens, 4 / 8 / 10 streams 42.0 / 47.3 / 53.3 and 42.0 / 46.9 / 49.6
(warm) against 37.5 / 46.4 / 50.3 before the two patches.
