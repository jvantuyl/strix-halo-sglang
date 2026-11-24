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
Reuses the same seeded inputs within each comparison. Use --calibrate to
time the first config again and see same-config timing drift.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)


def build_tensors(kind, E, K, N, topk, group_size, device, generator,
                  asymmetric=False):
    """Real (CUDA) tensors mirroring the tuner harness's benchmark_config."""
    def t(*shape, dtype=torch.float32):
        x = torch.empty(*shape, dtype=dtype, device=device)
        if x.is_floating_point():
            x.normal_(generator=generator)
        else:
            x.random_(0, 255, generator=generator)
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
        if asymmetric:
            out["w1_zp"] = t(E, N, triton.cdiv(K, group_size), dtype=torch.uint8)
            out["w2_zp"] = t(E, K // 2, triton.cdiv(N, group_size),
                             dtype=torch.uint8)
        out["group_size"] = group_size
    else:
        raise SystemExit(f"moe_ab_spotcheck: unsupported kind {kind!r}")
    return out


def build_case(m, t, which, generator):
    """Share activations, output storage and routing across all configs."""
    E, K, N, topk = t["E"], t["K"], t["N"], t["topk"]
    total_tokens = m * topk
    down = which == "down"
    a = torch.randn((total_tokens, N) if down else (m, K),
                    dtype=torch.bfloat16, device="cuda", generator=generator)
    c = torch.empty((m, topk, K) if down else (total_tokens, 2 * N),
                    dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(m, topk, device="cuda", generator=generator)
    ids = torch.randint(0, E, (m, topk), dtype=torch.int32, device="cuda",
                        generator=generator)
    return a, c, weights, ids


def time_gemm(t, case, config, which, filter_expert, num_iters=10):
    """Time one GEMM (up or down) under `config`; returns µs per call."""
    A, C, topk_weights, topk_ids = case
    sorted_ids, expert_ids, post_pad = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], t["E"]
    )
    if which == "up":
        B = t["w1"]
        mul_routed, tk = False, t["topk"]
        b_scale, b_zp = t["w1_scale"], t.get("w1_zp")
    else:
        B = t["w2"]
        mul_routed, tk = True, 1
        b_scale, b_zp = t["w2_scale"], t.get("w2_zp")

    def run():
        invoke_fused_moe_kernel(
            A, B, None, C, None, b_scale, b_zp,
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


def down_candidates(base, group_size):
    """Try the gfx1151 tile space while retaining the up kernel's row tile."""
    bm = base["BLOCK_SIZE_M"]
    yield base
    for bn in (16, 32, 64, 128, 256):
        for bk in (32, 64, 128):
            if bk % group_size or (bm + bn) * bk * 2 > 65536:
                continue
            for nw in (1, 2, 4, 8):
                if not 256 <= bm * bn // nw <= 4096:
                    continue
                for gm in (1, 8, 32):
                    for waves in (0, 2):
                        cfg = dict(base, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                                   GROUP_SIZE_M=gm, num_warps=nw, waves_per_eu=waves)
                        if cfg != base:
                            yield cfg


def sweep_down(m, tensors, case, base, filter_expert, num_iters, out):
    """Write a separate result per size; never overwrite the serving profile."""
    path = out / f"M={m}.json"
    if path.exists():
        raise FileExistsError(path)
    measure = lambda cfg: time_gemm(tensors, case, cfg, "down", filter_expert, num_iters)
    configs = list(down_candidates(base, tensors["group_size"]))
    records = []
    print(f"M={m}: screening {len(configs)} down tiles", flush=True)
    for i, cfg in enumerate(configs, 1):
        records.append((measure(cfg), cfg))
        if i % 32 == 0:
            print(f"M={m}: {i}/{len(configs)} screened", flush=True)
    finalists = sorted(records, key=lambda pair: pair[0])[:5]
    if all(cfg != base for _, cfg in finalists):
        finalists.append(records[0])
    samples = [(cfg, []) for _, cfg in finalists]
    for round_idx in range(3):
        for cfg, times in (samples if round_idx % 2 == 0 else reversed(samples)):
            times.append(measure(cfg))
    base_us = sorted(next(times for cfg, times in samples if cfg == base))[1]
    best_cfg, best_times = min(samples, key=lambda pair: sorted(pair[1])[1])
    best_us = sorted(best_times)[1]
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(dict(M=m, candidates=len(records), baseline_us=base_us,
                       best_us=best_us, config=best_cfg, finalists=samples), f, indent=2)
        f.write("\n")
    tmp.rename(path)
    print(f"M={m}: {len(records)} configs, {base_us:.1f} → {best_us:.1f} µs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["bf16", "int4"])
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--K", type=int, required=True, help="hidden size")
    ap.add_argument("--N", type=int, required=True, help="intermediate size")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--asymmetric", action="store_true",
                    help="include packed zero points for asymmetric int4")
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--which", required=True, choices=["up", "down"])
    ap.add_argument("--configs", nargs="+", required=True,
                    help="config JSON paths to A/B (labels from basenames)")
    ap.add_argument("--up-config",
                    help="for down timing, use this up config's BLOCK_SIZE_M as runtime does")
    ap.add_argument("--filter-expert", default="False",
                    help="constexpr value to compile under (server-real: False)")
    ap.add_argument("--num-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="rerun the first config on the same inputs after the comparison")
    ap.add_argument("--sweep-down", type=Path, metavar="DIR",
                    help="sweep non-row tiles and write per-size results in DIR")
    args = ap.parse_args()
    if args.asymmetric and args.kind != "int4":
        ap.error("--asymmetric requires --kind int4")
    if args.up_config and args.which != "down":
        ap.error("--up-config requires --which down")
    if args.sweep_down:
        if not (args.kind == "int4" and args.asymmetric and args.group_size
                and args.which == "down" and args.up_config and len(args.configs) == 1):
            ap.error("--sweep-down requires asymmetric int4 down, an up config, and one baseline")
        args.sweep_down.mkdir(parents=True, exist_ok=True)

    filter_expert = args.filter_expert == "True"
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    t = build_tensors(args.kind, args.E, args.K, args.N, args.topk,
                      args.group_size, "cuda", generator, args.asymmetric)
    for m in args.batch_sizes:
        case_generator = torch.Generator(device="cuda").manual_seed(args.seed + m)
        case = build_case(m, t, args.which, case_generator)
        results = []
        up = load_config(args.up_config, m)[0] if args.up_config else None
        if args.sweep_down:
            base = dict(load_config(args.configs[0], m)[0], BLOCK_SIZE_M=up["BLOCK_SIZE_M"])
            sweep_down(m, t, case, base, filter_expert, args.num_iters, args.sweep_down)
            continue
        paths = args.configs + args.configs[:1] if args.calibrate else args.configs
        for i, path in enumerate(paths):
            cfg, tag = load_config(path, m)
            if up is not None:
                cfg = dict(cfg, BLOCK_SIZE_M=up["BLOCK_SIZE_M"])
            label = os.path.basename(path) + tag
            if i == len(args.configs):
                label += " (repeat)"
            us = time_gemm(t, case, cfg, args.which, filter_expert,
                           args.num_iters)
            results.append((label, us))
        line = " | ".join(f"{l}: {u:.1f} µs" for l, u in results)
        print(f"M={m} [{args.which}]: {line}", flush=True)


if __name__ == "__main__":
    main()
