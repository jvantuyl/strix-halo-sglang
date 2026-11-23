#!/usr/bin/env python3
"""Isolated up/down fused-MoE GEMM spot-checks with real launches.

Where tools/moe_precompile.py compiles without launching (synthesized
args), this module does the opposite: it times a SINGLE GEMM invocation
(the real ``invoke_fused_moe_kernel`` — no patched code) under an explicit
config, so down-projection candidates can be A/B'd against the shared
config, or any constexpr variant (e.g. filter_expert) can be compared,
independent of the tuner harness's combined-sequence measurement.

PROVENANCE (same sources as moe_precompile.py; sglang pin 70b5b03e):
- python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py:
  ``invoke_fused_moe_kernel()`` (~line 793) is called as-is; the tensor
  shapes and the alignment sharing rule come from
  python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
  (``_fused_moe_kernel_sequence`` ~line 481: both GEMMs share ONE
  ``moe_align_block_size`` result computed with the UP config's
  BLOCK_SIZE_M; up C is (total_tokens, w1.shape[1]); down A is
  (total_tokens, N) with N = w1.shape[1]//2 after silu; down C is
  (num_tokens, topk, K); down uses mul_routed_weight=True, top_k=1).
- Timing mirrors the tuner's benchmark_config (capture 10 calls in a CUDA
  graph, replay, cuda events) minus the tuner's bail-out logic.

Usage (inside the image, GPU attached):
  python3 moe_ab_spotcheck.py --kind bf16 --E 512 --K 2560 --N 640 --topk 8 \
      --batch-sizes 1024 2048 4096 --which down \
      --configs configs/moe/.../E=512,N=640,...json \
                configs/moe/.../E=512,N=640,..._down.json
Prints one µs-per-call line per (M, config).
"""
import argparse
import json
import os
import time

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)


def build_tensors(kind, E, K, N, topk, group_size, device):
    """Real (CUDA) tensors mirroring the tuner harness's benchmark_config."""
    def t(*shape, dtype=torch.float32):
        x = torch.empty(*shape, dtype=dtype, device=device)
        if x.is_floating_point():
            x.normal_()
        else:
            x.random_(0, 255)  # int4 weights: any bytes exercise the kernel
        return x

    out = {"E": E, "K": K, "N": N, "topk": topk}
    if kind == "bf16":
        out["w1"] = t(E, 2 * N, K, dtype=torch.bfloat16)
        out["w2"] = t(E, K, N, dtype=torch.bfloat16)
        out["w1_scale"] = out["w2_scale"] = None
    elif kind == "int4":
        assert group_size
        out["w1"] = t(E, 2 * N, K // 2, dtype=torch.uint8)
        out["w2"] = t(E, K, N // 2, dtype=torch.uint8)
        out["w1_scale"] = t(E, 2 * N, triton.cdiv(K, group_size),
                            dtype=torch.bfloat16)
        out["w2_scale"] = t(E, K, triton.cdiv(N, group_size),
                            dtype=torch.bfloat16)
        out["group_size"] = group_size
    else:
        raise SystemExit(f"moe_ab_spotcheck: unsupported kind {kind!r}")
    return out


def time_gemm(m, t, config, which, filter_expert, num_iters=10):
    """Time one GEMM (up or down) under `config`; returns µs per call."""
    E, K, N, topk = t["E"], t["K"], t["N"], t["topk"]
    total_tokens = m * topk  # padded_tokens=0 without down-TMA
    hidden = t["x"].shape[1]
    topk_weights = torch.randn(m, topk, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, E, (m, topk), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, post_pad = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    if which == "up":
        A, B, C = t["x"], t["w1"], torch.empty(
            total_tokens, t["w1"].shape[1], dtype=torch.bfloat16, device="cuda"
        )
        mul_routed, tk = False, topk
        b_scale = t["w1_scale"]
    else:
        A = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")
        B, C = t["w2"], torch.empty(
            m, topk, hidden, dtype=torch.bfloat16, device="cuda"
        )
        mul_routed, tk = True, 1
        b_scale = t["w2_scale"]
    n_tile = B.shape[1]

    def run():
        invoke_fused_moe_kernel(
            A, B, None, C, None, b_scale, None,
            topk_weights, topk_ids, sorted_ids, expert_ids, post_pad,
            mul_routed, tk, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=t["w1_scale"] is not None,
            per_channel_quant=False,
            block_shape=[0, t["group_size"]] if t.get("group_size") else None,
            filter_expert=filter_expert,
        )

    graph = torch.cuda.CUDAGraph()
    run()  # JIT compile & warmup outside capture
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph.reset()
    return start.elapsed_time(end) / num_iters / 10 * 1000  # µs


def load_config(path, m):
    """Accept a bare config dict or the tuner's per-M file form."""
    with open(path) as f:
        stored = json.load(f)
    if "BLOCK_SIZE_M" in stored:
        return stored, ""
    key = min(stored.keys(), key=lambda k: abs(int(k) - m))
    entry = stored[key]
    cfg = entry["config"] if isinstance(entry, dict) and "config" in entry else entry
    return cfg, f"@M={key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["bf16", "int4"])
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--K", type=int, required=True, help="hidden size")
    ap.add_argument("--N", type=int, required=True, help="intermediate size")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--which", required=True, choices=["up", "down"])
    ap.add_argument("--configs", nargs="+", required=True,
                    help="config JSON paths to A/B (labels from basenames)")
    ap.add_argument("--filter-expert", default="False",
                    help="constexpr value to compile under (server-real: False)")
    ap.add_argument("--num-iters", type=int, default=10)
    args = ap.parse_args()

    filter_expert = args.filter_expert == "True"
    for m in args.batch_sizes:
        t = build_tensors(args.kind, args.E, args.K, args.N, args.topk,
                          args.group_size, "cuda")
        t["x"] = torch.randn(m, args.K, dtype=torch.bfloat16, device="cuda")
        results = []
        for path in args.configs:
            cfg, tag = load_config(path, m)
            label = os.path.basename(path) + tag
            us = time_gemm(m, t, cfg, args.which, filter_expert,
                           args.num_iters)
            results.append((label, us))
        line = " | ".join(f"{l}: {u:.1f} µs" for l, u in results)
        print(f"M={m} [{args.which}]: {line}", flush=True)


if __name__ == "__main__":
    main()
