#!/usr/bin/env python3
"""Isolated prefill QSA QKV microbench for the gfx1151 patch-54 image.

Uses a real packed sidecar layer. The baseline is the installed patch-54
module, including its transient unpack buffer and F.linear. The candidate
dequantizes inside a tiled BF16 GEMM, with FP32 accumulation and no split-K
workspace. This does not run SGLang or replace the convergence gauntlet.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from bench_rocmfpx_q8_dense import cpu_scales, ue4m3


N, K = 13312, 2560
MS = (1024, 4096, 16384)
LAYERS = tuple(range(3, 48, 4))
PIN = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c"
# BM, BN, BK, group of M tiles, waves (Triton num_warps).
TILES = (
    (32, 64, 64, 4, 4),
    (32, 128, 64, 4, 4),
    (64, 64, 64, 4, 4),
    (64, 128, 32, 4, 4),
    (64, 128, 64, 1, 4),
    (64, 128, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (64, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (128, 128, 64, 4, 4),
    (128, 128, 64, 4, 8),
    (128, 256, 64, 4, 8),
)


@triton.jit
def q8_bf16(q, e):
    return (q.to(tl.float32) * ue4m3(e.to(tl.int32))).to(tl.bfloat16)


@triton.jit
def check_candidate_dequant(Q, E, Y, B: tl.constexpr):
    # The same group-of-32 broadcast and BF16 conversion used by the GEMM.
    block = tl.program_id(0) * B + tl.arange(0, B)
    i = block[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(Y + i, q8_bf16(tl.load(Q + i), tl.load(E + block)[:, None]))


@triton.jit
def dense_prefill(X, Q, E, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    tl.static_assert(M % (BM * GM) == 0 and N % BN == 0 and K % BK == 0)
    tl.static_assert(BK % 32 == 0)
    pid = tl.program_id(0)
    group = pid // (GM * (N // BN))
    m_tile = group * GM + pid % GM
    n_tile = (pid // GM) % (N // BN)
    rm = m_tile * BM + tl.arange(0, BM)
    rn = n_tile * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, BK // 32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(K // BK):
        a = tl.load(X + rm[:, None] * K + block * BK + rk[None, :])
        q = tl.load(Q + rn[:, None] * K + block * BK + rk[None, :])
        e = tl.load(E + rn[:, None] * (K // 32) + block * (BK // 32) + rs[None, :])
        b = q8_bf16(q.reshape(BN, BK // 32, 32), e[:, :, None]).reshape(BN, BK)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc)


def load_packed(sidecar, layer):
    name = f"model.layers.{layer}.qkv_proj"
    manifest = json.loads((sidecar / "manifest.json").read_text())
    if (manifest["format"], manifest["reference"]) != ("qsa-q8-v1", PIN):
        raise ValueError("Expected the patch-54 sidecar format and reference pin")
    with safe_open(sidecar / f"{name}.safetensors", framework="pt", device="cpu") as f:
        if set(f.keys()) != {"codes", "scales"}:
            raise ValueError("Expected exactly codes and scales")
        q, e = f.get_tensor("codes"), f.get_tensor("scales")
    hashes = manifest["packed_sha256"][name]
    for key, t, dtype, shape in (
        ("codes", q, torch.int8, (N, K)),
        ("scales", e, torch.uint8, (N, K // 32)),
    ):
        if t.dtype != dtype or t.shape != shape or not t.is_contiguous():
            raise ValueError(f"Invalid {name}.{key} layout")
        if hashlib.sha256(t.numpy()).hexdigest() != hashes[key]:
            raise ValueError(f"Hash mismatch: {name}.{key}")
    return q, e


def bitwise_equal(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Bitwise comparison requires equal shapes and dtypes")
    bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[actual.dtype]
    return torch.equal(actual.view(bits), expected.view(bits))


def check_dequant(patch54, q_cpu, e_cpu, q, e):
    # Independent reference, including the invalid-byte fallback.
    enc = torch.arange(256, dtype=torch.uint8)
    table = torch.where(enc <= 126, enc.view(torch.float8_e4m3fn).float() / 2, 0)
    assert bitwise_equal(cpu_scales(), table)
    # Exhaustively exercise both decoders with every signed code/scale pair.
    qc = torch.arange(-128, 128, dtype=torch.int32).to(torch.int8).repeat(256)
    ec = enc.repeat_interleave(8)
    expected = qc.float() * table[ec.long()].repeat_interleave(32)
    qg, eg = qc.to(q.device), ec.to(e.device)
    full = torch.empty_like(qg, dtype=torch.float32)
    rounded = torch.empty_like(qg, dtype=torch.bfloat16)
    patch54.unpack[(triton.cdiv(qc.numel(), 1024),)](qg, eg, full, qc.numel(), 1024)
    check_candidate_dequant[(ec.numel() // 8,)](qg, eg, rounded, 8)
    assert bitwise_equal(full.cpu(), expected), "patch-54 FP32 dequant mismatch"
    assert bitwise_equal(rounded.cpu(), expected.bfloat16()), "tiled BF16 dequant mismatch"

    # Check both decoders on the actual sidecar too, reusing one BF16 buffer.
    # No decoded weights remain on the device during either timing arm.
    w = torch.empty_like(q, dtype=torch.bfloat16)
    for start in range(0, N, 512):
        sl = slice(start, start + 512)
        ref = (q_cpu[sl].float().reshape(-1, K // 32, 32) *
               table[e_cpu[sl].long()][..., None]).reshape(-1, K).bfloat16()
        size = q[sl].numel()
        patch54.unpack[(triton.cdiv(size, 1024),)](q[sl], e[sl], w[sl], size, 1024)
        assert bitwise_equal(w[sl].cpu(), ref), f"patch-54 BF16 mismatch at row {start}"
        check_candidate_dequant[(e[sl].numel() // 8,)](q[sl], e[sl], w[sl], 8)
        assert bitwise_equal(w[sl].cpu(), ref), f"tiled BF16 mismatch at row {start}"


def make_native(x, q, e, tile):
    m = x.shape[0]
    bm, bn, bk, gm, warps = tile

    def run():
        y = x.new_empty((m, N))
        dense_prefill[(m // bm * (N // bn),)](
            x, q, e, y, m, N, K, bm, bn, bk, gm, num_warps=warps,
        )
        return y

    return run


def capture(run):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


def warm(run, count=2):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(count):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def timing(run, repeats):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def measure(run, repeats):
    # Separate graph pools so only one projection output is captured per arm.
    # Peak allocated deltas exclude already-resident activations/weights and
    # measure the PyTorch allocator, not external HIP/BLAS workspace or RSS.
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    eager_us = timing(run, repeats)
    eager_peak = torch.cuda.max_memory_allocated() - resident

    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph, output = capture(run)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    graph_us = timing(graph.replay, repeats)
    graph_peak = torch.cuda.max_memory_allocated() - resident
    del graph, output
    return dict(eager_us=eager_us, graph_us=graph_us,
                eager_peak_extra_bytes=eager_peak, graph_peak_extra_bytes=graph_peak)


def checked_replay(run, expected):
    graph, out = capture(run)
    # Poison the captured output so a non-executing replay cannot pass.
    out.fill_(float("nan"))
    graph.replay()
    for a, b in zip(out.split(256), expected.split(256), strict=True):
        assert bitwise_equal(a, b), "graph replay changed BF16 GEMM output"


def matmul_error(got, want):
    # Evaluate every result but cap float temporaries at 256 rows each.
    sums = []
    for a, b in zip(got.split(256), want.split(256), strict=True):
        b = b.float()
        d = a.float() - b
        sums.append(torch.stack((d.square().sum(), b.square().sum(), d.abs().max())))
    errors, refs, max_abs = torch.stack(sums).T.cpu().tolist()
    rel_l2 = math.sqrt(math.fsum(errors) / math.fsum(refs))
    return rel_l2, max(max_abs)


def check_matmul(baseline, native):
    # Replay each arm with at most two full projection outputs on device.
    want = baseline()
    checked_replay(baseline, want)
    got = native()
    rel_l2, max_abs = matmul_error(got, want)
    del want
    assert rel_l2 < 0.006, f"relative L2 {rel_l2} exceeds patch-54 tolerance"
    checked_replay(native, got)
    return rel_l2, max_abs


def bench_m(linear, q, e, m, args, rng):
    x = torch.randn((m, K), dtype=torch.bfloat16, device=q.device, generator=rng)
    free, total = torch.cuda.mem_get_info()

    def baseline():
        return linear(x)[0]

    warm(baseline)
    sweep = []
    for tile in TILES:
        run = make_native(x, q, e, tile)
        warm(run)
        samples = [timing(run, args.repeats) for _ in range(3)]
        sweep.append(dict(tile=tile, eager_us=median(samples), eager_samples_us=samples))
    tile = min(sweep, key=lambda t: t["eager_us"])["tile"]
    native = make_native(x, q, e, tile)
    rel_l2, max_abs = check_matmul(baseline, native)
    samples = {"patch54": [], "native": []}
    order = (("patch54", baseline), ("native", native))
    for i in range(args.rounds):
        for name, run in (order if i % 2 == 0 else reversed(order)):
            samples[name].append(measure(run, args.repeats))
    ratios = {}
    medians_us = {}
    for mode in ("eager_us", "graph_us"):
        medians_us[mode] = {name: median(r[mode] for r in rows)
                            for name, rows in samples.items()}
        ratios[mode] = medians_us[mode]["native"] / medians_us[mode]["patch54"]
    return dict(kind="measurement", M=m, N=N, K=K, tile=tile, sweep=sweep,
                rel_l2=rel_l2, max_abs=max_abs, graph_replay_exact=True,
                x_bytes=x.nbytes, output_bytes=m*N*x.element_size(),
                patch54_unpack_bytes=N*K*x.element_size(), device_free_bytes_before=free,
                device_total_bytes=total, samples=samples, medians_us=medians_us,
                native_over_patch54=ratios,
                latency_reduction_pct={mode: (1 - ratio) * 100 for mode, ratio in ratios.items()},
                eager_5pct_pass=ratios["eager_us"] <= 0.95,
                graph_5pct_pass=ratios["graph_us"] <= 0.95)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecar", type=Path)
    ap.add_argument("--layer", type=int, choices=LAYERS, default=3)
    ap.add_argument("--m", type=int, choices=MS, nargs="+", default=MS)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True, help="New JSONL file (must not exist)")
    args = ap.parse_args()
    if min(args.repeats, args.rounds) < 1 or len(set(args.m)) != len(args.m):
        ap.error("require positive repeats/rounds and unique M values")
    if not torch.version.hip:
        ap.error("run in the patch-54 ROCm image on gfx1151")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx1151":
        ap.error("this benchmark requires gfx1151")

    from sglang.srt.model_executor import qsa_native_q8 as patch54

    if (patch54.N, patch54.K, patch54.PIN) != (N, K, PIN):
        raise RuntimeError("Installed patch-54 projection shape or reference pin changed")
    if torch.cuda.tunable.tuning_is_enabled():
        ap.error("set PYTORCH_TUNABLEOP_TUNING=0 to avoid tuning during this benchmark")
    torch.set_num_threads(4)
    q_cpu, e_cpu = load_packed(args.sidecar, args.layer)
    q, e = q_cpu.cuda(), e_cpu.cuda()
    linear = patch54.QSAQ8Linear(q, e)
    rng = torch.Generator(device=q.device).manual_seed(0)
    with args.out.open("x") as out, torch.inference_mode():
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            print(line, file=out, flush=True)
            print(line, flush=True)

        check_dequant(patch54, q_cpu, e_cpu, q, e)
        del q_cpu, e_cpu
        emit(dict(kind="metadata", device=props.name, arch=props.gcnArchName,
                  torch=torch.__version__, triton=triton.__version__, hip=torch.version.hip,
                  layer=args.layer, sidecar=str(args.sidecar), pin=PIN, ms=args.m,
                  packed_bytes=q.nbytes + e.nbytes, dequant_pairs_exact=65536,
                  sidecar_bf16_weights_exact=q.numel(), repeats=args.repeats, rounds=args.rounds,
                  tunableop_enabled=torch.cuda.tunable.is_enabled(),
                  tunableop_file=torch.cuda.tunable.get_filename()))
        rows = []
        for m in args.m:
            row = bench_m(linear, q, e, m, args, rng)
            emit(row)
            rows.append(row)
        full = set(args.m) == set(MS)
        emit(dict(kind="summary", all_required_m=full,
                  eager_5pct_across_required_m=full and all(r["eager_5pct_pass"] for r in rows),
                  graph_5pct_across_required_m=full and all(r["graph_5pct_pass"] for r in rows),
                  serving_gauntlet_run=False))


if __name__ == "__main__":
    main()
