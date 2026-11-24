#!/usr/bin/env python3
"""Benchmark ROCmFPX Q8 dense projections on the stock Qwen checkpoint.

Q8_0_ROCMFPX stores 32 signed bytes followed by one UE4M3 scale byte.
Here the same bytes are stored in separate code/scale arrays for aligned
GPU loads. The kernel dequantizes to BF16 in registers, without materializing
a BF16 weight matrix. This is a microbench, not an SGLang loader.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open


LAYERS = "model.language_model.layers."
GDN = LAYERS + "0.linear_attn."
QSA = LAYERS + "3.self_attn."
SHARED = LAYERS + "0.mlp.shared_expert."
WEIGHTS = {
    "gdn-in": tuple(GDN + "in_proj_" + p + ".weight" for p in ("qkv", "z")),
    "gdn-out": (GDN + "out_proj.weight",),
    "qsa-qkv": tuple(QSA + p + "_proj.weight" for p in ("q", "k", "v")),
    "qsa-out": (QSA + "o_proj.weight",),
    "shared-up": tuple(SHARED + p + "_proj.weight" for p in ("gate", "up")),
    "shared-down": (SHARED + "down_proj.weight",),
    "lm-head": ("lm_head.weight",),
}


def cpu_scales():
    # Upstream rocmfpx_scale_table, including its invalid-code fallback.
    return torch.tensor([
        0.0 if b > 126 else math.ldexp(
            (b & 7) if b < 8 else 8 + (b & 7), max(b // 8, 1) - 11
        ) for b in range(256)
    ], dtype=torch.float32)


def quantize(w, scales):
    n, k = w.shape
    assert k % 32 == 0
    qs = torch.empty((n, k), dtype=torch.int8)
    es = torch.empty((n, k // 32), dtype=torch.uint8)
    for start in range(0, n, 2048):
        x = w[start:start + 2048].float().reshape(-1, k // 32, 32)
        desired = x.abs().amax(dim=-1) / 127
        hi = torch.searchsorted(scales[1:127], desired).add_(1).clamp_(1, 126)
        lo = (hi - 1).clamp_(min=1)
        e = torch.where(desired - scales[lo] <= scales[hi] - desired, lo, hi)
        e = torch.where(desired > 0, e, 0)
        s = scales[e].unsqueeze(-1)
        v = x * torch.where(s > 0, s.reciprocal(), 0)
        # C lroundf: nearest, ties away from zero.
        q = (v.abs() + 0.5).floor().copysign(v).clamp_(-127, 127)
        qs[start:start + len(x)] = q.reshape(-1, k).to(torch.int8)
        es[start:start + len(x)] = e.to(torch.uint8)
    return qs, es


@triton.jit
def ue4m3(e):
    mant = e & 7
    exp = e >> 3
    frac = tl.where(exp == 0, mant, mant + 8).to(tl.float32)
    power = ((tl.maximum(exp, 1) + 116) << 23).to(tl.float32, bitcast=True)
    return tl.where(e <= 126, frac * power, 0.0)


@triton.jit
def dequant(Q, E, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + i, i < SIZE, 0).to(tl.float32)
    e = tl.load(E + i // 32, i < SIZE, 0).to(tl.int32)
    tl.store(Y + i, q * ue4m3(e), i < SIZE)


@triton.jit
def dense(X, Q, E, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    rm = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.full((16, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = (i * SPLIT + part) * BK + rk
        a = tl.load(X + rm[:, None] * K + k[None, :],
                    (rm[:, None] < M) & (k[None, :] < K), 0)
        q = tl.load(Q + rn[None, :] * K + k[:, None],
                    (rn[None, :] < N) & (k[:, None] < K), 0)
        e = tl.load(E + rn[None, :] * (K // 32) + k[:, None] // 32,
                    (rn[None, :] < N) & (k[:, None] < K), 0)
        b = (q.to(tl.float32) * ue4m3(e.to(tl.int32))).to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
    tl.store(Y + (part * M + rm[:, None]) * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def merge(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    part = tl.arange(0, SPLIT)
    v = tl.load(P + part[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < SIZE)


def reference(q, e, scales):
    return (q.float().reshape(*e.shape, 32) * scales[e.long()][..., None]).reshape(q.shape)


def check_format(scales, device):
    # Exercise all scale bytes (including invalid ones) and all signed codes.
    e = torch.arange(256, dtype=torch.int32).repeat_interleave(8).to(torch.uint8)
    q = (torch.arange(256, dtype=torch.int32) - 128).to(torch.int8).repeat(256)
    # Independent scale decode: unsigned E4M3 is half the E4M3FN value.
    enc = torch.arange(127, dtype=torch.uint8)
    torch.testing.assert_close(scales[:127], enc.view(torch.float8_e4m3fn).float() / 2,
                               rtol=0, atol=0)
    expected = reference(q, e, scales)
    result = torch.empty(q.shape, device=device, dtype=torch.float32)
    dequant[(triton.cdiv(q.numel(), 256),)](q.to(device), e.to(device), result, q.numel(), 256)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    print("Exact CPU/GPU dequant verified for 65,536 code/scale pairs", flush=True)


def timing(fn, replays, calls=12):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def load_weight(root, index, keys):
    chunks = []
    for key in keys:
        with safe_open(root / index[key], framework="pt", device="cpu") as f:
            chunks.append(f.get_tensor(key))
    return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


def bench(w, scales, ms, device, replays, generator):
    n, k = w.shape
    q, e = quantize(w, scales)
    # Small projections are repeatedly used with distinct layer weights
    # in the real model. Cycle addresses instead of measuring L2 residency.
    copies = 4 if w.numel() * 2 < 8 * 1024**2 else 1
    ws = [w.to(device, copy=True) for _ in range(copies)]
    qs = [q.to(device, copy=True) for _ in range(copies)]
    es = [e.to(device, copy=True) for _ in range(copies)]
    stride = max(1, n // 256)
    ref = reference(q[::stride], e[::stride], scales).to(device=device, dtype=torch.bfloat16)
    del q, e
    for m in ms:
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        expected = F.linear(x, ref)
        y = torch.empty((m, n), dtype=torch.bfloat16, device=device)

        def linear():
            i = linear.i % copies
            linear.i += 1
            return F.linear(x, ws[i])

        linear.i = 0

        def make_q8(bn, bk, split, warps):
            partial = torch.empty((split, m, n), dtype=torch.float32, device=device) if split > 1 else y

            def run():
                i = run.i % copies
                run.i += 1
                dense[(triton.cdiv(n, bn), split)](
                    x, qs[i], es[i], partial, m, n, k, bn, bk, split, num_warps=warps
                )
                if split > 1:
                    merge[(triton.cdiv(m*n, 256),)](partial, y, m*n, split, 256)
                return y

            run.i = 0
            return run

        candidates = []
        for bn, warps in ((16, 1), (32, 2), (64, 4)):
            for bk in (32, 64, 128):
                for split in (1, 4, 8):
                    run = make_q8(bn, bk, split, warps)
                    result = run()[:, ::stride]
                    error = (result.float() - expected.float()).norm() / expected.float().norm()
                    assert error.item() < 0.006, (n, k, m, bn, bk, split, error.item())
                    candidates.append((timing(run, max(3, replays // 3)), (bn, bk, split, warps)))
        _, tile = min(candidates)
        run = make_q8(*tile)
        bf, quant = [], []
        for round_id in range(5):
            for times, fn in (((bf, linear), (quant, run)) if round_id % 2 == 0
                              else ((quant, run), (bf, linear))):
                times.append(timing(fn, replays))
        yield dict(M=m, N=n, K=k, copies=copies, bf16_us=median(bf), q8_us=median(quant),
                   bf16_samples_us=bf, q8_samples_us=quant, tile=tile, tiles_tested=len(candidates))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--shapes", choices=WEIGHTS, nargs="+", default=list(WEIGHTS))
    ap.add_argument("--m", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--replays", type=int, default=15)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if any(not 1 <= m <= 16 for m in args.m) or args.replays < 1:
        ap.error("require 1 <= M <= 16 and positive replays")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    scales = cpu_scales()
    check_format(scales, device)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with args.out.open("x") as out:
        for shape in args.shapes:
            w = load_weight(args.checkpoint, index, WEIGHTS[shape])
            for result in bench(w, scales, args.m, device, args.replays, generator):
                result["shape"] = shape
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(json.dumps(result), flush=True)
            del w


if __name__ == "__main__":
    main()
