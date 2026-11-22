#!/usr/bin/env python3
"""No-launch pre-compilation of the fused-MoE Triton GEMM kernels.

Instead of running the tuner harness (which allocates, launches, captures
CUDA graphs and measures), this module synthesizes the exact argument lists
the runtime builds and hands them to Triton's ``JITFunction.warmup()`` —
the upstream API for "compile into the cache without launching". Tensors
live on the ``meta`` device: specialization only reads dtype, shape,
stride and pointer alignment, and aligned CUDA allocations are divisible
by 16 exactly like meta's data_ptr()==0, so the computed cache key matches
a real launch while the worker needs no GPU and almost no RAM.

PROVENANCE — everything below was read out of the sglang sources INSIDE
image ``strix-halo-sglang:rocpro10`` (upstream pin
``70b5b03e78612c94f86ac98eb4d2d8d19ceda738``, 2026-09-21). If a Triton or
sglang upgrade changes kernel names, signatures or specialization, this is
where to look again:

- ``python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py``
  - ``@triton.jit fused_moe_kernel_gptq_awq`` at line ~92 (int4/AWQ GEMM)
  - ``@triton.jit fused_moe_kernel`` at line ~330 (generic GEMM: bf16/fp8)
  - ``invoke_fused_moe_kernel()`` at line ~793: the single launch site for
    both kernels; the two ``kernel[grid](...)`` call blocks (~line 960 for
    gptq/awq, ~line 1010 for generic) are mirrored one-to-one by
    ``_warm_gptq()`` / ``_warm_generic()`` below, with the launch replaced
    by ``.warmup()``. Keep those two blocks in sync when upgrading.
  - ``support_tensor_descriptor()`` at line ~56.
- ``python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py``
  - ``_prepare_fused_moe_run()`` line ~403: config resolution, TMA flags,
    ``moe_align_block_size`` call. We deliberately do NOT call it — we
    force our own config and compute alignment ourselves (see below).
  - ``_fused_moe_kernel_sequence()`` line ~481: how the two GEMMs are
    invoked: up GEMM at ~line 620 (A=hidden_states, B=w1, C=(total_tokens, N),
    mul_routed_weight=apply_router_weight_on_input, top_k=topk,
    ROUTER_TOPK defaults to 1), down GEMM at ~line 826 (A=intermediate_cache2
    (total_tokens, N//2), B=w2, C=(num_tokens, topk, K),
    mul_routed_weight=True, top_k=1, ROUTER_TOPK=topk). ``total_tokens =
    num_tokens * topk + padded_tokens`` with ``padded_tokens=0`` unless
    down-TMA (off for gfx1151: configs carry no USE_TMA key).
  - ``compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16
    else tl.float16`` (~line 503).
- ``python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py``
  lines ~88-98: ``sorted_token_ids`` length is
  ``topk_ids.numel() * block_size`` when ``numel < num_experts + 1``,
  else ``topk_ids.numel() + (num_experts + 1) * (block_size - 1)``;
  ``expert_ids`` length ``cdiv(sorted_len, block_size)``;
  ``num_tokens_post_padded`` shape ``(1,)``. Both GEMMs share ONE
  alignment computed with the UP config's BLOCK_SIZE_M (the down config's
  BLOCK_M never affects alignment).
- ``python/sglang/srt/layers/moe/moe_runner/triton.py`` line ~124:
  ``filter_expert = num_experts is None or num_experts != num_local_experts``.
  Serving this model (TP=1, no EP) sets num_experts == num_local_experts
  → filter_expert=**False**. NOTE: the upstream tuner's harness builds
  MoeRunnerConfig without num_experts → filter_expert=**True** there, and
  filter_expert is a constexpr — the tuner's own compiled variants do NOT
  match the server's keys. That mismatch is a key reason this module
  exists and synthesizes with the server's value.
- ``benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py``
  lines ~45-115: the tensor shapes/dtypes per quant path that this module
  mirrors for int4 packing (w1 as ``(E, 2N, K//2)`` uint8 etc.).

Constexpr values that must track reality (all verified for this box,
gfx1151 / TP=1 / no EP / no TMA):
- swap_ab: only for fp8_w8a8 (``should_enable_swap_ab``) → False here.
- a_desc/b_desc: None (USE_TMA never set in our tuned config files).
- USE_GDC/launch_pdl: only when ``is_arch_support_pdl()`` (CUDA Hopper+).
- bias: None → both bias strides 0.
- has_zp: False (AWQ g32 has no zero points; GPTQ would set them).
"""
import argparse
import json
import os
import sys
import time
import triton
import triton.language as tl
import torch


def align_shapes(num_tokens, topk, num_experts, block_m):
    """Mirror sglang's moe_align_block_size allocation (see provenance)."""
    numel = num_tokens * topk
    if numel < num_experts + 1:
        sorted_len = numel * block_m
    else:
        sorted_len = numel + (num_experts + 1) * (block_m - 1)
    return sorted_len, triton.cdiv(sorted_len, block_m)


def _meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def build_tensors(kind, E, K, N, topk, group_size):
    """Synthetic tensors with server-realistic shapes/dtypes/strides.

    Shapes mirror the tuner harness's benchmark_config (see provenance).
    N here is the per-shard intermediate size; w1 carries 2N (gated).
    """
    t = {"E": E, "K": K, "N": N, "topk": topk}
    if kind == "bf16":
        t["w1"] = _meta(E, 2 * N, K, dtype=torch.bfloat16)
        t["w2"] = _meta(E, K, N, dtype=torch.bfloat16)
        t["w1_scale"] = None
        t["w2_scale"] = None
    elif kind == "int4":
        assert group_size, "int4 needs --group-size"
        # int4 packing: two int4 values per byte; w1 packs the K dim,
        # w2 packs the (N//2)-wide down-GEMM K dim (see provenance).
        t["w1"] = _meta(E, 2 * N, K // 2, dtype=torch.uint8)
        t["w2"] = _meta(E, K, N // 2, dtype=torch.uint8)
        block_n = 1  # block_shape[0] == 0 for AWQ/GPTQ per invoke's assert
        t["w1_scale"] = _meta(E, 2 * N, triton.cdiv(K, group_size),
                              dtype=torch.bfloat16)
        t["w2_scale"] = _meta(E, K, triton.cdiv(N, group_size),
                              dtype=torch.bfloat16)
        t["group_size"] = group_size
    else:
        sys.exit(f"moe_precompile: unsupported kind {kind!r}")
    return t


def warm_one(kernel, grid, args, kwargs):
    kernel.warmup(*args, grid=grid, **kwargs)


def warm_gptq(cfg, m, t, num_tokens_post_padded_len=1):
    """int4/AWQ up+down GEMMs; mirrors invoke_fused_moe_kernel's gptq branch."""
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        fused_moe_kernel_gptq_awq as kern,
    )

    E, K, N, topk = t["E"], t["K"], t["N"], t["topk"]
    w1, w2 = t["w1"], t["w2"]
    s1, s2 = t["w1_scale"], t["w2_scale"]
    total_tokens = m * topk
    sorted_len, m_blocks = align_shapes(m, topk, E, cfg["BLOCK_SIZE_M"])
    sorted_ids = _meta(sorted_len, dtype=torch.int32)
    expert_ids = _meta(m_blocks, dtype=torch.int32)
    post_pad = _meta(1, dtype=torch.int32)
    topk_weights = _meta(m, topk, dtype=torch.float32)
    # topk_ids is int32 in every real path (topk.py:897 topk_ids.to(torch.int32));
    # a wrong dtype here silently changes the signature and thus the cache key.
    topk_ids = _meta(m, topk, dtype=torch.int32)
    even_Ks = (w1.shape[2]) % cfg["BLOCK_SIZE_K"] == 0

    # up: A (m, K) bf16, B w1 (E, 2N, K//2) uint8, C (total_tokens, 2N) bf16
    A = _meta(m, K, dtype=torch.bfloat16)
    C = _meta(total_tokens, w1.shape[1], dtype=torch.bfloat16)
    grid = (
        triton.cdiv(sorted_len, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(w1.shape[1], cfg["BLOCK_SIZE_N"]),
    )
    warm_one(kern, grid, (
        A, w1, C, s1, None, topk_weights, sorted_ids, expert_ids, post_pad,
        w1.shape[1], A.shape[1], sorted_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        w1.stride(0), w1.stride(2), w1.stride(1),
        C.stride(-2), C.stride(-1),
        s1.stride(0), s1.stride(2), s1.stride(1), 0, 0, 0,
    ), dict(
        group_size=t["group_size"], MUL_ROUTED_WEIGHT=False, top_k=topk,
        compute_type=tl.bfloat16, has_zp=False, use_int4_w4a16=True,
        use_int8_w8a16=False, even_Ks=even_Ks, filter_expert=False,
        **cfg,
    ))

    # down: A (total_tokens, N) bf16, B w2 (E, K, N//2) uint8,
    #       C (m, topk, K) bf16; shares the up alignment (see provenance)
    A2 = _meta(total_tokens, N, dtype=torch.bfloat16)
    C2 = _meta(m, topk, K, dtype=torch.bfloat16)
    even_Ks2 = (w2.shape[2]) % cfg["BLOCK_SIZE_K"] == 0
    grid2 = (
        triton.cdiv(sorted_len, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(w2.shape[1], cfg["BLOCK_SIZE_N"]),
    )
    warm_one(kern, grid2, (
        A2, w2, C2, s2, None, topk_weights, sorted_ids, expert_ids, post_pad,
        w2.shape[1], A2.shape[1], sorted_ids.shape[0], topk_ids.numel(),
        A2.stride(0), A2.stride(1),
        w2.stride(0), w2.stride(2), w2.stride(1),
        C2.stride(-2), C2.stride(-1),
        s2.stride(0), s2.stride(2), s2.stride(1), 0, 0, 0,
    ), dict(
        group_size=t["group_size"], MUL_ROUTED_WEIGHT=True, top_k=1,
        compute_type=tl.bfloat16, has_zp=False, use_int4_w4a16=True,
        use_int8_w8a16=False, even_Ks=even_Ks2, filter_expert=False,
        **cfg,
    ))


def warm_generic(cfg, m, t):
    """bf16 (and fp8) up+down GEMMs; mirrors invoke's generic branch."""
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        fused_moe_kernel as kern,
    )

    E, K, N, topk = t["E"], t["K"], t["N"], t["topk"]
    w1, w2 = t["w1"], t["w2"]
    total_tokens = m * topk
    sorted_len, m_blocks = align_shapes(m, topk, E, cfg["BLOCK_SIZE_M"])
    sorted_ids = _meta(sorted_len, dtype=torch.int32)
    expert_ids = _meta(m_blocks, dtype=torch.int32)
    post_pad = _meta(1, dtype=torch.int32)
    topk_weights = _meta(m, topk, dtype=torch.float32)
    topk_ids = _meta(m, topk, dtype=torch.int32)

    def base_kwargs(even_ks, router_topk, mul_routed, tk):
        return dict(
            MUL_ROUTED_WEIGHT=mul_routed, top_k=tk,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            per_channel_quant=False, even_Ks=even_ks, c_sorted=False,
            filter_expert=False, swap_ab=False,
            FUSE_ADD_TO_OUTPUT=False, MASK_OUTPUT=False,
            LORA_PRESERVE_BASE=False, FUSE_SUM_ALL_REDUCE=False,
            ROUTER_TOPK=router_topk, FUSE_SWIGLU=False,
            **cfg,
        )

    # up: A (m, K), B w1 (E, 2N, K), C (total_tokens, 2N)
    A = _meta(m, K, dtype=torch.bfloat16)
    C = _meta(total_tokens, w1.shape[1], dtype=torch.bfloat16)
    grid = (
        triton.cdiv(sorted_len, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(w1.shape[1], cfg["BLOCK_SIZE_N"]),
    )
    warm_one(kern, grid, (
        A, None, w1, None, None, C, None, None,
        topk_weights, sorted_ids, expert_ids, post_pad, None,
        w1.shape[1], A.shape[1], sorted_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        w1.stride(0), w1.stride(2), w1.stride(1),
        0, 0,
        C.stride(-2), C.stride(-1),
        0, 0, 0, 0, 0,
        0, 0,
    ), base_kwargs(
        A.shape[1] % cfg["BLOCK_SIZE_K"] == 0, 1, False, topk
    ))

    # down: A (total_tokens, N), B w2 (E, K, N), C (m, topk, K)
    A2 = _meta(total_tokens, N, dtype=torch.bfloat16)
    C2 = _meta(m, topk, K, dtype=torch.bfloat16)
    grid2 = (
        triton.cdiv(sorted_len, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(w2.shape[1], cfg["BLOCK_SIZE_N"]),
    )
    warm_one(kern, grid2, (
        A2, None, w2, None, None, C2, None, None,
        topk_weights, sorted_ids, expert_ids, post_pad, None,
        w2.shape[1], A2.shape[1], sorted_ids.shape[0], topk_ids.numel(),
        A2.stride(0), A2.stride(1),
        w2.stride(0), w2.stride(2), w2.stride(1),
        0, 0,
        C2.stride(-2), C2.stride(-1),
        0, 0, 0, 0, 0,
        0, 0,
    ), base_kwargs(
        A2.shape[1] % cfg["BLOCK_SIZE_K"] == 0, topk, True, 1
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-shape", required=True,
                    help='JSON: {"kind": "bf16"|"int4", "E","K","N","topk"'
                         ' [,"group_size"]}')
    ap.add_argument("--space-file", required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--slice", required=True, help="i:K worker slice")
    args = ap.parse_args()

    shape = json.loads(args.model_shape)
    with open(args.space_file) as f:
        space = json.load(f)
    # int4: only BLOCK_K values compatible with the group size make sense
    if shape["kind"] == "int4":
        g = shape["group_size"]
        space = [c for c in space
                 if g % c["BLOCK_SIZE_K"] == 0 or c["BLOCK_SIZE_K"] % g == 0]

    i, k = map(int, args.slice.split(":"))
    mine = space[i::k]
    t = build_tensors(shape["kind"], shape["E"], shape["K"], shape["N"],
                      shape["topk"], shape.get("group_size"))
    warm = warm_gptq if shape["kind"] == "int4" else warm_generic

    # First warmup pays the one-time Triton/LLVM infra cost; report it.
    t0 = time.perf_counter()
    failed = 0
    for mi, m in enumerate(args.batch_sizes):
        slow = 0
        t_m = time.perf_counter()
        for ci, cfg in enumerate(mine):
            try:
                warm(cfg, m, t)
                slow += 1
            except Exception as e:
                failed += 1
                print(f"[w{i}] M={m} cfg={cfg} FAILED: {type(e).__name__}: {e}",
                      flush=True)
            if (ci + 1) % 25 == 0 or ci + 1 == len(mine):
                print(f"[w{i}] M={m}: {ci + 1}/{len(mine)} compiled", flush=True)
        print(f"[w{i}] M={m}: DONE {len(mine)} configs ({slow} ok) "
              f"in {time.perf_counter() - t_m:.0f}s", flush=True)
    if failed:
        print(f"[w{i}] {failed} configs FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
