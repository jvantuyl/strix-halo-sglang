# Patch 18: deterministic HyperConnection mix for decode batches on ROCm

**File:** `python/sglang/srt/layers/hc_mix_triton.py`
**Script:** [`patch_hc_mix_rocm.py`](patch_hc_mix_rocm.py)

## Symptom

Greedy decode of the same prompt on Qwen3.8-Flash-Next was not repeatable:
the top-1 logprob differed from the first decode token on, every run,
cycling through a small set of values (e.g. −0.0225 / −0.0256 / −0.0304),
and after a few hundred tokens the text diverged on near-tie tokens.
Prefill (`max_tokens=1`) was bit-identical across cold runs. None of the
usual suspects mattered: `--disable-cuda-graph`, `--disable-overlap-schedule`,
`--disable-radix-cache` and `SGLANG_ENABLE_QWEN4_PLE_FUSION=0` all left the
drift in place, and repeat-and-compare stress tests of the QSA decode kernel,
the GDN recurrent kernels, the fused MoE (int4 g128, tuned tiles),
`select_experts` and every model GEMM shape came back 0 differences.

## Cause

`GatedResidual.mix` (the HyperConnection low-rank input mix, called twice per
layer) has three implementations. The CuTe JIT pair is sm_100-only
(`get_device_capability()[0] == 10`), so on ROCm every batch of ≤ 16 rows,
i.e. every decode step, goes to `hc_mix_triton.fused_hc_mix`: a persistent
Triton kernel that

* accumulates the split-K down projection with device-scope `tl.atomic_add`
  into a shared fp32 buffer, so the summation order, and with it the bf16
  mix weights after `silu`, depend on which CTA lands first (upstream's own
  comment says so; it only steps aside under
  `--enable-deterministic-inference`), and
* synchronises its CTAs with a software grid barrier that spins on a
  device-scope atomic and assumes one resident CTA per SM, which nothing
  guarantees on a shared GPU.

Rows > 16 use the `torch.compile` path, which is why prefill was stable.

## Fix

Sending the mix to the torch path (`fused_hc_mix_supported` → False on HIP)
fixes the drift but cost ~35% decode speed here (bs 1: 15.1 → 9.9 tok/s; 16
streams: 72 → 63), because the two GEMM shapes involved are untuned under
TunableOp and fall to poor hipBLASLt heuristics.

Instead the patch adds a two-launch variant of the same math:

1. `_hc_mix_down_split_kernel`, grid `(⌈lowrank/32⌉, 8)`: each program owns
   one N block and one K split and writes its fp32 partial to
   `t_part[split, row, n]`. No atomics, no zeroing pass.
2. `_hc_mix_up_split_kernel`, grid `(⌈hs/32⌉)`: sums the 8 partials in a
   fixed order, then the same `silu` / up projection / `sigmoid` gate /
   mean tail as the persistent kernel.

No grid barrier, so nothing depends on CTA co-residency. On HIP
`fused_hc_mix` dispatches to it; `SGLANG_HC_MIX_ATOMIC=1` restores upstream's
kernel. Since the variant is deterministic, `--enable-deterministic-inference`
keeps the fused path on HIP instead of falling back to torch. CUDA is
untouched.

## Verification

Standalone (`4 × 2560` hidden, rank 320, the model's shape): the split
variant matches the fp32 reference as closely as the persistent kernel
(max abs error identical to 5 decimals at rows 1–16), is bit-identical
300/300 at rows 1, 5 and 16 and 200/200 under CUDA graph replay, and runs in
58 µs vs 66 µs for the persistent kernel (cache-warm; both are near the
memory bound for the 13 MB of weights). The persistent kernel showed its
non-determinism even in isolation at 16 rows (191/200 identical).

End to end on the DERISKED checkpoint: 96- and 512-token greedy completions
bit-identical (tokens and logprobs) across 4 cold runs; bs 1 decode 15.1
tok/s, 8 streams 49.8, 16 streams 72.0, the same as with upstream's kernel.

`tools/test_qwen38_rocm.py` item 7 checks that the split variant is selected
on HIP, matches the reference within bf16 tolerance, and is bit-identical
over repeated launches at rows 1, 5 and 16.
