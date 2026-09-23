# Patch 26: Triton prefill MQA for the QSA indexer

**Files:** `python/sglang/srt/layers/attention/qsa/mqa.py`
**Script:** [`patch_qsa_mqa_prefill_triton.py`](patch_qsa_mqa_prefill_triton.py)

## Symptom

After [patch 25](25-qsa-topk-slices.md) the allocator snapshot of a
43k-token prefill still showed 2,370 MiB live at the peak, now with 1,152
MiB in `torch_qsa_mqa_prefill`: 640 + 640 MiB at the `einsum` and its
`relu`, 512 MiB more in the head sum, scale and mask copies.

## Cause

The prefill twin of [patch 24](24-qsa-mqa-triton.md). `qsa_mqa_prefill`
scores every compressed block against every query row of the packed prefill
batch. Upstream slices the rows so the fp32 logits stay under a 128 MiB
budget per call (`_QSA_PREFILL_LOGITS_BUDGET_BYTES`), which is the right
bound for the TileLang kernel that writes logits directly. Without TileLang
the torch reference runs, and `einsum("mhd,nd->mnh")` materialises the
per-head scores, `[rows, keys, 4]` fp32, four times the budget, then `relu`
copies that, then `.sum(-1)`, the scale and `masked_fill` each copy the
result: ~1.15 GiB live per QSA layer for a 128 MiB answer, on every prefill
chunk, twelve layers.

## Fix

`_qsa_mqa_prefill_kernel`, one program per (64-row, 64-key) tile. A tile
with no key inside any of its rows' `[start, end)` stores `-inf` after two
small loads. A live tile loads its keys once as `[head_dim, 64]` and, per
head in `tl.static_range` order, takes `tl.dot` of the bf16 queries and keys
with fp32 accumulation, applies `relu`, sums the heads, scales, and stores
scores where valid and `-inf` elsewhere. bf16 products are exact in fp32, so
only the summation order differs from the reference GEMM: measured max
abs error 2.9e-6 (relative 3e-7). Same contract as the reference (fp32
`[rows, keys]`, `-inf` outside each row's range); the logits are the only
allocation; no atomics. Dispatch is TileLang → Triton → reference, sharing
patch 24's `SGLANG_QSA_MQA_TRITON=0` override.

A first version computed the dot as a `[64, 64, 128]` broadcast product
reduced with `tl.sum`. That is 2,048 fp32 registers per thread; compiling
it OOMed the 30 GB host and the OOM killer took the test container. Kernel
experiments now run under `docker run --memory 12g` and `timeout`.

## Verification

`tools/test_qwen38_rocm.py` item 15: vs the reference on packed rows with
empty, partial and full ranges at 3,072 × 10,715 (the 128 MiB budget shape),
8,192 × 2,048 and tile-edge shapes down to 1 × 1; `-inf` masks equal, finite
scores within 1e-3, bit-identical over 20 launches, dispatch and override,
peak allocation under a quarter of the reference's (measured 1,130 → 126
MiB). Standalone: 25.7 → 5.0 ms at the budget shape.

End to end (DERISKED, MTP, cap 20, patches ≤ 26): the driver-level prefill
transient is flat in prompt length, +2,046 / +2,048 / +2,108 MiB at 13k /
43k / 99k tokens (was 1,824 / 2,668 / 2,690 before patches 25–26); the
remaining ~2 GiB is per-chunk activations (HC combine 2 × 320 MiB, gate
projection 208, MTP fused residual 160, norms). Greedy repeats identical;
the 14k-token prompt's accept histogram moved `[1,2,4,12]` → `[1,3,2,13]`
at the same accept length, one near-tie in the block selection flipping
under the changed rounding, the same class of change as patches 21 and 24.
Needle recall exact at 14k and 50k tokens (code and paragraph). Prefill
655–658 → 681 tok/s on a 54k-token prompt; bs 1 24.7, 20 streams 99.3
tok/s. Mixed worst case (18 streams + two 43k prefills together) 1.58 GiB
free (1.2 before patch 25); the rest of that gap is the caching allocator's
per-stream pools under the overlap scheduler (reserved 93.5 vs allocated
88.9 GiB at the end of the run), which
`garbage_collection_threshold:0.8` did not change (1.56 GiB).
