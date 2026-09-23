# Patch 24: length-bounded Triton decode MQA for the QSA indexer

**Files:** `python/sglang/srt/layers/attention/qsa/mqa.py`
**Script:** [`patch_qsa_mqa_triton.py`](patch_qsa_mqa_triton.py)

## Symptom

Concurrency throughput on the stock checkpoint fell from 99.4 tok/s at 16
streams (Sep 21, patches ≤ 14) to 67–74 at 16–20 streams on every image
since patches 18–21, on stock and DERISKED alike, while single-stream and
prefill did not move. The runbook carried it as "could not be bisected".

## Bisect

The image behind the 99.4 run was rebuilt from the commit that recorded it
(`d1953c8`, Dockerfile pin unchanged, `sgl_kernel` `.so` md5 identical to the
current image, pip freeze differing in two unrelated packages) and both
images were run at both context lengths, stock checkpoint, 200-token
stories, two passes each:

| Image | `--context-length` | 8 streams | 16 streams | 20 streams |
|---|---:|---:|---:|---:|
| `d1953c8` (patches ≤ 14) | 32,768 | 47.9–53.1 | 83.6–85.5 | 87.3–92.7 |
| `d1953c8` (patches ≤ 14) | 131,072 | 42.7–48.0 | 71.4–72.5 | 71.6–77.8 |
| `24884d0` (patches ≤ 23) | 32,768 | 47.5–62.4 | 84.8–96.5 | 89.2–93.5 |
| patches ≤ 21 (earlier session) | 131,072 | 42.8 | 67.7 | 71.7 |
| patches ≤ 24, `SGLANG_QSA_MQA_TRITON=0` | 131,072 | 40.9–53.4 | 66.8–78.9 | 68.4–77.0 |

The launcher's default `SGLANG_CONTEXT` moved from 32768 to 131072 in
`12b5ace`, between the two measurements. That is the whole gap; patches
15–23 cost nothing at any concurrency.

## Cause

`qsa_mqa_decode` scores every compressed block of a request against the
indexer query (`relu(q·k)` summed over the 4 index heads, scaled by
`sqrt(128)`), once per QSA layer per decode step, to pick the top-2048
blocks. Upstream has a TileLang kernel for it and a torch reference,
`torch_qsa_mqa_decode`. TileLang is not installed in this image (its ROCm
build is a from-source TVM stack with no gfx1151 precedent), so the
reference ran. It gathers the *whole* page-table window per row,
`page_table.shape[1] × page_size` = `context_length / 4` blocks × 128,
converts it to fp32, runs an einsum over it, masks past the real length and
copies into a `-inf`-filled `[batch, max_model_len]` buffer: work
proportional to the model's context limit, not the request's length.

Per layer per decode step on this box (256-token sequences, fill + MQA +
top-k, eager, `PYTORCH_TUNABLEOP_ENABLED=0`):

| Window (blocks) | bs 1 | bs 8 | bs 16 | bs 20 |
|---|---:|---:|---:|---:|
| 8,192 (32k context) | 0.26 ms | 0.91 | 1.50 | 1.77 |
| 32,768 (131k context) | 0.31 ms | 2.68 | 4.88 | 6.03 |

About 90% of that is the MQA gather + einsum. Times 12 QSA layers at bs 20
that is 21 vs 72 ms per step, which matches the observed step-time gap.
bs 1 is under a millisecond either way, so single-stream never showed it.

## Fix

A Triton kernel, `_qsa_mqa_decode_kernel`, with one program per (row,
64-block tile). A tile at or past `min(context_len, page_table width)`
stores `-inf` and exits after one scalar load; a live tile gathers its
pages from the compressed K cache (bf16 `[pages, 16, 1, 128]`), computes
the per-head dot products in fp32 in `tl.static_range` order, applies
`relu`, sums the heads, scales, and stores scores where valid and `-inf`
elsewhere. Same contract as the reference: fp32 `[batch, max_model_len]`,
`-inf` at or past `context_len` and past the page table. No atomics, so
launches are bit-identical; static pointers only, so it captures into the
decode graphs. Dispatch order becomes TileLang (if present) → Triton (any
CUDA/HIP device) → torch reference; `SGLANG_QSA_MQA_TRITON=0` forces the
reference for A/B. Any CUDA build without TileLang was on the same path.

Per layer on this box the whole stage goes from 1.77 → 0.022 ms (32k, bs
20) and 6.46 → 0.028 ms (131k, bs 20); a single 32k-token row costs 0.040
ms.

## Verification

`tools/test_qwen38_rocm.py` item 13: page size 16, 4 heads of 128,
shuffled page tables, context lengths of 0, 1, page fractions, tile edges
and full rows, `max_model_len` wider than the page table; finite scores
within 1e-3 of the reference (measured ≤ 2e-6), identical `-inf` masks,
bit-identical over 20 launches, graph replay equal to eager and following
the `context_lens` buffer, dispatcher picks the Triton path unless the env
var says otherwise, and the Triton path is faster than the reference on a
bs 20 / 131k shape (6.5 ms vs 0.03). Fails on the unpatched image (no
`triton_qsa_mqa_decode`).

End to end, stock checkpoint, 131k context, same box and prompts, patches
≤ 24 image:

| Path | bs 1 | 8 streams | 16 streams | 20 streams |
|---|---:|---:|---:|---:|
| torch reference (`SGLANG_QSA_MQA_TRITON=0`) | 13.2–14.2 | 40.9–53.4 | 66.8–78.9 | 68.4–77.0 |
| Triton (default) | 14.0–15.1 | 50.6–62.4 | **89.2–104.7** | **95.2–103.5** |

That is back at, and above, the 99.4 measured at 32k on the old image,
now at the full 131k context. Greedy outputs are identical between the two
paths on 19-, 2.5k-, 14k- and 49k-token prompts (400 tokens each; the 49k
prompt shows a 6e-2 logprob difference at one position, fp32 summation
order, same argmax) and identical across cold repeats on each path. Needle
recall at 14k and 50k tokens is exact.

DERISKED with MTP (3 steps, 4 draft tokens, 10-request cap): greedy decode
stays bit-identical across cold runs with identical accept histograms
(19-token prompt 96 tokens 3/3, 14k-token prompt 64 tokens 3/3); bs 1
21.9–23.0 → 25.3–25.9 tok/s on a 183-token prompt, unchanged within noise
on 3.4k / 13.5k / 54k prompts, and 4 / 8 / 10 streams 42 / 47 / 50–53 →
48–53 / 57–64 / 66–70 tok/s aggregate.
