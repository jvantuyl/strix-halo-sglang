# Patch 38: launch-config tuning for the split HyperConnection mix

**File:** `python/sglang/srt/layers/hc_mix_triton.py`
**Script:** [`patch_hc_mix_tune.py`](patch_hc_mix_tune.py)

## Background

The decode-step profile (`bench/results.md`, 2026-11-19/20) put the
HyperConnection mix at 10.5% of GPU-busy time: two near-empty Triton GEMMs
per layer (`_hc_mix_down_split` at ~97 µs, `_hc_mix_up_split` at ~35 µs in
the profiler trace, ~196 launches per MTP round). Patch 18's deterministic
two-launch variant carried its launch metadata over from upstream's
persistent kernel — `num_warps=8`, default `num_stages` — and the current
model's shapes (hc=4, hidden 2560, lowrank 320) make that a bad fit:

* the down kernel is a 16×10240×320 GEMM tiled as 80 CTAs of
  (16 rows × 32 N × 256 K), each chaining **five** `tl.dot` iterations whose
  accumulators depend on one another, with no load/dot overlap;
* 8 warps per CTA oversubscribes a tile whose useful output is 16×32.

The whole computation moves ~10 MB (w_down 6.55 MB dominates; x and t_part
mostly hit L2) — ~45 µs at DRAM peak, and the pair measured 56 µs
back-to-back (down 42.5, up 13.7) with a per-launch floor near 15 µs.

## What was tried

Standalone sweep at the model's exact shapes
(`~/scratch/rocmfp4-port/hc_mix_tune.py`; correctness against an fp32 torch
reference, timing over 200-iteration loops, rows 1 and 16):

* **More K-splits** (SPLIT_K 8→20, 200 CTAs): down 42.5 → 27.8 µs, but the
  up kernel's `static_range` partial-sum unroll grows with SPLIT_K
  (13.7 → 16.7 µs). Net pair −18%.
* **SPLIT_K=40**: down 14 µs but the up kernel explodes to ~63 µs
  (unrolled 40-iteration sum). Dead end, as is BLOCK_K=512 (bigger K tiles
  pipeline *worse* at these sizes).
* **BLOCK_N=16** (320 exact blocks): more CTAs, similar gains, up unchanged.
* **Launch metadata on the untouched patch-18 grid**: `num_warps` 8→2 and
  `num_stages=2` on the down kernel — down 42.5 → **13.5 µs**, up 13.6 µs,
  pair **27.2 µs, −51%**. This is the shipped change: two launch parameters,
  no grid, layout or math change.

## Change

`_hc_mix_down_split_kernel` launch: `num_warps=8` → `num_warps=2` plus
`num_stages=2`. The up kernel (already near its memory floor) and the
`_SPLIT_MIX_*` constants are untouched.

## Verification

* Max error vs fp32 reference unchanged from patch 18's variant
  (9.75e-04 rows=1, 1.90e-03 rows=16) — the reduction order is identical,
  only scheduling changed.
* Repeat-launch bit-identity 50/50 at rows 1/5/16 (patch 18's determinism
  property preserved; no atomics anywhere on this path).
* `tests/check_hc_mix_launch.py` asserts the launch parameters, the
  untouched up kernel and the unchanged grid constants, in-image without a
  GPU.
* End-to-end (2026-11-20, production A/B, GPU exclusive both sides, same
  request, warm state): decode gap p50 135.2 → 130.8 ms (−3.3%), wall for
  the 220-token request 13.78 → 13.32 s (−3.3%, same 100-step stream on
  both builds, so the acceptance term cancels). Matches the ~2.8 ms/round
  predicted from the standalone pair timings.
