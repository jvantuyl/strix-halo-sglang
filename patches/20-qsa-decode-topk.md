# Patch 20: decode QSA block selection honours the HIP top-k guard

**Files:** `python/sglang/srt/layers/attention/qsa/kernel.py`,
`python/sglang/srt/layers/attention/qsa/qsa_indexer.py`
**Script:** [`patch_qsa_decode_topk.py`](patch_qsa_decode_topk.py)

## Symptom

After patch 18 made short-context greedy decode bit-identical, prompts above
~2k tokens still drifted: the same 8.6k-token prompt gave different top-1
logprobs on later decode steps and diverged on a near-tie token around step
37. Isolated, the JIT `fast_topk` kernel returns the right *set* of blocks but
in a different *order* from one launch to the next as soon as a row has more
than 512 compressed blocks (2048 tokens at compress ratio 4); the QSA decode
kernel sums the selected blocks in index order, so the order changes the
rounding.

## Cause

Patch 11 keeps the JIT kernel off ROCm inside `qsa_fast_topk` (on RDNA 3.5 it
reads out of bounds; measured IndexError, wrong rows and HSA exceptions in
later kernels), but `QSAIndexer.select_decode_tokens` calls
`sglang.kernels.ops.elementwise.fast_topk` directly when the block budget is
512, bypassing the guard. Every decode step of Qwen3.8-Flash-Next ran the JIT
kernel regardless.

The reference `_qsa_fixed_width_topk` is not a drop-in replacement for
decode: it loops over rows with a host sync each (`int(lengths[row])`), which
the decode CUDA graphs cannot capture.

## Fix

* `qsa_jit_topk_allowed()` in `kernel.py` holds the patch 11 condition
  (`torch.version.hip is None or SGLANG_QSA_TOPK_JIT=1`); `qsa_fast_topk`
  uses it instead of the inline test.
* `qsa_dense_topk(logits, lengths, topk)`: fixed-width top-k over the dense,
  row-start-0 logits that `qsa_mqa_decode` returns (already -inf beyond each
  row's length, so the mask is belt and braces): `masked_fill`, `torch.topk`,
  -1 where the value is -inf, pad to the budget. Vectorised, no host sync,
  graph-capturable.
* `select_decode_tokens`: the JIT branch is gated on `qsa_jit_topk_allowed()`;
  on HIP it takes `qsa_dense_topk`; everything else is unchanged (so CUDA
  behaviour is identical).

## Cost

Per QSA layer per decode step, logits `(bs, 32768)`, budget 512: dense
65–130 µs at bs 1–20 (135 µs under graph replay at bs 20) vs 6–22 µs for the
JIT kernel. Twelve QSA layers make that ~1–1.5 ms on a ~66 ms decode step.
`SGLANG_QSA_TOPK_JIT=1` restores the JIT kernel on both paths.

## Verification

`tools/test_qwen38_rocm.py` item 9: guard false on HIP and true with the
override; rows of 30, 511, 600 and 2152 blocks match the reference set with
the right -1 padding; 50/50 bit-identical launches. Item 4 (the
`qsa_fast_topk` chain) still passes with the refactored guard. End-to-end
numbers are in the runbook's long-context determinism note.
