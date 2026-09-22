#!/usr/bin/env python3
"""gfx1151 patch 18: deterministic HC low-rank mix for decode-size batches on ROCm.

`GatedResidual.mix` has three implementations of the HyperConnection input mix.
The CuTe JIT pair is sm_100-only, so on ROCm every batch of <= 16 rows (that
is, every decode step) goes to `hc_mix_triton.fused_hc_mix`, a persistent
Triton kernel that

  (a) accumulates the split-K down projection with device-scope `atomic_add`,
      so the fp32 sum order, and with it the bf16 mix weights, change from
      one launch to the next (upstream's own comment says so; it only steps
      aside under --enable-deterministic-inference), and
  (b) synchronises its CTAs with a software grid barrier that spins on a
      device-scope atomic and relies on every CTA staying resident.

Measured on Qwen3.8-Flash-Next: greedy decode of one prompt gave different
logprobs from the first decode token on, cycling through a handful of values,
while prefill (> 16 rows, torch.compile path) was bit-identical. Routing the
mix to the torch path instead fixes the drift but costs ~35% decode speed on
this box (the two GEMM shapes are untuned under TunableOp).

This patch adds a two-launch variant of the same math: the down projection
writes one fp32 partial per K split (no atomics), and the up-projection kernel
sums the partials in a fixed order before silu/gate/mean. No grid barrier, so
nothing depends on CTA co-residency. It is bit-identical run to run (200/200
standalone and under CUDA graph replay) and matches the fp32 reference as
closely as the persistent kernel does, at about the same cost (58 us vs 66 us
cache-warm for the model's 4x2560 / rank-320 shape). On HIP `fused_hc_mix`
dispatches to it; SGLANG_HC_MIX_ATOMIC=1 restores upstream's kernel. Because
the variant is deterministic, --enable-deterministic-inference keeps the fused
path on HIP instead of falling back to torch. CUDA is untouched.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/layers/hc_mix_triton.py"
text = open(p).read()

# 1. imports
old_imp = "import torch\nimport triton\nimport triton.language as tl\n"
assert text.count(old_imp) == 1, "hc_mix_triton: import anchor not found"
text = text.replace(
    old_imp, "import os\n\nimport torch\nimport triton\nimport triton.language as tl\n", 1
)

# 2. the two-launch kernels, inserted ahead of the persistent kernel's helpers
anchor = "\n\n_counters_cache = {}\n"
assert text.count(anchor) == 1, "hc_mix_triton: _counters_cache anchor not found"
split_kernels = '''

# gfx1151 patch 18: deterministic two-launch variant. The down projection is
# split over K into per-split fp32 partials (one program per (N block, split),
# no atomics); the up-projection kernel sums the partials in a fixed order.
# Same tiles and the same silu/sigmoid/mean tail as the persistent kernel.
_SPLIT_MIX_SPLIT_K = 8
_SPLIT_MIX_BLOCK_N = 32
_SPLIT_MIX_BLOCK_K = 256


@triton.jit
def _hc_mix_down_split_kernel(
    x_ptr,
    w_down_ptr,
    t_part_ptr,
    K,
    LOWRANK,
    num_rows,
    k_per_split,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    nb = tl.program_id(0)
    split = tl.program_id(1)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_k = tl.arange(0, BLOCK_K)
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < LOWRANK
    acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    k_begin = split * k_per_split
    for k0 in range(k_begin, k_begin + k_per_split, BLOCK_K):
        k = k0 + offs_k
        mask_k = k < K
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(w), acc)
    tl.store(
        t_part_ptr
        + split * ROWS * LOWRANK
        + offs_m[:, None] * LOWRANK
        + n[None, :],
        acc,
        mask=mask_n[None, :],
    )


@triton.jit
def _hc_mix_up_split_kernel(
    x_ptr,
    w_up_ptr,
    t_part_ptr,
    out_ptr,
    LOWRANK,
    HS,
    num_rows,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    jb = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j = jb * BLOCK_J + offs_j
    mask_j = j < HS
    gj = offs_g[:, None] * HS + j[None, :]
    gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
    mask_gj = tl.reshape(
        tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
    )
    acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
    for r0 in range(0, LOWRANK, BLOCK_R):
        r = r0 + offs_r
        mask_r = r < LOWRANK
        a = tl.zeros((ROWS, BLOCK_R), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            a += tl.load(
                t_part_ptr
                + s * ROWS * LOWRANK
                + offs_m[:, None] * LOWRANK
                + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
        a = a * inv_hc
        t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
        w = tl.load(
            w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
            mask=mask_gj[:, None] & mask_r[None, :],
            other=0.0,
        )
        acc = tl.dot(t, tl.trans(w), acc)
    gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
    xg = tl.load(
        x_ptr
        + offs_m[:, None, None] * (HC * HS)
        + offs_g[None, :, None] * HS
        + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(gate * xg, axis=1) * inv_hc
    tl.store(
        out_ptr + offs_m[:, None] * HS + j[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_j[None, :],
    )


def _use_split_mix() -> bool:
    return torch.version.hip is not None and (
        os.environ.get("SGLANG_HC_MIX_ATOMIC", "0") != "1"
    )


def _fused_hc_mix_split(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    k_blocks = triton.cdiv(k, _SPLIT_MIX_BLOCK_K)
    k_per_split = triton.cdiv(k_blocks, _SPLIT_MIX_SPLIT_K) * _SPLIT_MIX_BLOCK_K
    t_part = torch.empty(
        (_SPLIT_MIX_SPLIT_K, rows_pad, lowrank), dtype=torch.float32, device=device
    )
    _hc_mix_down_split_kernel[
        (triton.cdiv(lowrank, _SPLIT_MIX_BLOCK_N), _SPLIT_MIX_SPLIT_K)
    ](
        hyper_input_normed,
        w_down,
        t_part,
        k,
        lowrank,
        rows,
        k_per_split,
        ROWS=rows_pad,
        BLOCK_N=_SPLIT_MIX_BLOCK_N,
        BLOCK_K=_SPLIT_MIX_BLOCK_K,
        num_warps=8,
    )
    _hc_mix_up_split_kernel[(triton.cdiv(hs, 32),)](
        hyper_input_normed,
        w_up,
        t_part,
        out,
        lowrank,
        hs,
        rows,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        SPLIT_K=_SPLIT_MIX_SPLIT_K,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out
'''
text = text.replace(anchor, split_kernels + anchor, 1)

# 3. deterministic mode keeps the fused path when the split variant is in use
old_det = """    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
"""
assert text.count(old_det) == 1, "hc_mix_triton: deterministic check anchor not found"
new_det = """    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    # (gfx1151 patch 18: the split variant used on HIP has no atomics.)
    if _deterministic_inference() and not _use_split_mix():
        return False
"""
text = text.replace(old_det, new_det, 1)

# 4. dispatch
old_disp = """    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
"""
assert text.count(old_disp) == 1, "hc_mix_triton: fused_hc_mix body anchor not found"
new_disp = """    if _use_split_mix():
        return _fused_hc_mix_split(hyper_input_normed, w_down, w_up, hc, hs)
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
"""
text = text.replace(old_disp, new_disp, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 18 (hc mix rocm) applied")
