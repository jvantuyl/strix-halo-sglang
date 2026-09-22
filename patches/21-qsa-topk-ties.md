# Patch 21: tie-stable QSA block selection on ROCm

**Files:** `python/sglang/srt/layers/attention/qsa/kernel.py`
**Script:** [`patch_qsa_topk_ties.py`](patch_qsa_topk_ties.py)

## Symptom

With patches 18 and 20 in place, greedy decode of short prompts was
bit-identical across cold runs, but a 1,475-token prompt still gave a
different first-token logprob from one run to the next (same token), and an
8.6k-token prompt still diverged on a near-tie token. Prefill-only requests
reproduced it, so the drift was in prefill, not decode; a 12-paragraph
prompt (a few hundred tokens) was clean.

Repeat-and-compare tests over the whole prefill chain at 1,475 and 8,192
tokens (`causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`,
the indexer MQA logits, block top-k, index expansion, sparse GQA attention,
every model GEMM shape, `select_experts` + `fused_moe`, the HC mix GEMMs)
found exactly one stage that differs: the block top-k, 19/19 launches.

## Cause

`torch.topk` on this ROCm torch build (2.13 / HIP 7.13) returns *tied*
entries in an order that changes from launch to launch; when the tie sits on
the k boundary the selected set changes too. Distinct values are ordered
deterministically, which is why random-data tests of patch 20 passed.

The QSA indexer scores each compressed block with `relu(q·k).sum(heads)`, so
every block whose four head scores are all negative scores exactly 0.0, and
ties are the common case rather than a corner. Both HIP selection paths ran
through `torch.topk`:

* prefill: `qsa_fast_topk` → `_qsa_fixed_width_topk`, a Python loop calling
  `torch.topk` per query row (also a host sync per row: 119 ms per QSA layer
  at 1,475 rows);
* decode: patch 20's `qsa_dense_topk`.

The sparse attention kernels accumulate the selected blocks in the order
given, so a different tie order changes the rounding of the attention output
(logprob drift with the same token), and past 512 scored blocks a different
tied set at the boundary changes the token.

## Fix

`torch.sort(..., stable=True)` and `torch.topk` over *distinct* keys are both
bit-stable here, so two equivalent forms of the same rule (descending score,
ties toward the lower block index) cover the two shapes:

* `qsa_ordered_topk(masked, k, use_sort)`: `use_sort=True` is a stable
  descending sort, cheapest for many narrow rows (prefill: 0.19 ms at
  1,475 × 368, 5.5 ms at 8,192 × 2,152); `use_sort=False` maps each fp32
  score to an order-preserving int32 image, packs it with the reversed column
  index into one int64 key, and runs `torch.topk` on the now-unique keys,
  cheapest for a few 32k-wide rows (decode: 0.13 ms at bs 1, 0.26 ms at
  bs 20 under graph replay, against 0.06 / 0.66 ms for the sort). Signed
  zeros are folded (`x + 0.0`) so `-0.0` and `+0.0` tie in both forms; the
  test asserts the two forms return identical indices.
* `qsa_dense_topk` (decode) selects with the key form; the -inf/-1 padding
  logic is unchanged.
* `qsa_stable_rows_topk(logits, lengths, starts, topk)`: a vectorised
  `_qsa_fixed_width_topk` with the same contract (indices relative to each
  row's start, `min(length, topk)` valid entries, -1 padding) built on the
  sort form. `qsa_fast_topk` takes it on HIP after the JIT / `sgl_kernel`
  branches, in place of the per-row loop. The row chunking in
  `select_prefill_tokens` already caps the logits at 128 MB, so the sort's
  int64 index tensor stays under 256 MB.

CUDA keeps the JIT kernel and the reference loop; CPU keeps the reference.

## Cost

Prefill: 119 ms → 3.5 ms per QSA layer at 1,475 rows (the loop's host
syncs dominated), 7.6 ms at 8,192 rows. Decode: within 30 µs of patch 20's
plain `torch.topk` at bs 1, ~0.1 ms more per layer at bs 20.

## Verification

`tools/test_qwen38_rocm.py` item 10: relu-style tied scores (thousands of
exact zeros); sort and key forms agree; both select the same scores as
`torch.topk`; ties ascend by index; 30/30 bit-identical for each form while
plain `torch.topk` differs 30/30 on the same rows; `qsa_dense_topk` matches
the reference scores and padding on tied rows, is bit-identical and captures
into a CUDA graph; `qsa_stable_rows_topk` matches the reference scores with
non-zero row starts, growing lengths and k = 512 / 64. Items 4 and 9 still
pass. The stage-by-stage prefill probe is 0/11 differing at 1,475 and 8,192
tokens with the patch.

End to end on the DERISKED checkpoint, every run cold (`/flush_cache`,
`temperature=0`, top-3 logprobs compared at every position): a ~1.5k-token
prompt 4/4 identical, an 8.6k-token prompt 3/3 at 64 tokens and 4/4 at 200
tokens (it diverged at step 37 before patch 20 and differed in the first
logprob before this patch), 12 behaviour probes text-identical across two
passes. Prefill throughput rose from 535 / 538 / 504 to 629 / 695 / 676
tok/s at ~2k / ~10k / ~40k tokens; decode is unchanged (bs 1 14.5–14.8,
8 / 16 / 20 streams 48.1 / 71.3 / 71.5 tok/s).
