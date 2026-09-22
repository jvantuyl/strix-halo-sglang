#!/usr/bin/env python3
"""Kernel parity tests for the strix-halo-sglang gfx1151 patches.

Run inside the strix-halo-sglang:dev image with the GPU attached (safe
beside a running server; everything here uses tiny buffers):

  docker run --rm --device=/dev/kfd --device=/dev/dri \
      --group-add video --group-add render --security-opt seccomp=unconfined \
      -v $HOME/strix-halo-sglang/tools:/tests:ro \
      strix-halo-sglang:dev python3 /tests/test_qwen38_rocm.py

Covers:
  1. patch 12 -- Triton fused-MoE GPTQ/AWQ kernel WITH zero points vs a
     reference dequantized matmul (asymmetric int4, group 32: the exact
     layout of Qwen3.8-Flash-Next-AWQ-INT4).
  2. patch 11 -- CPU-side PLE gather vs torch reference (bf16 and fp8 tables,
     out-of-range ids must produce zero rows).
  3. patch 11 -- kda qwen38_qsa_sm121 Triton decode kernel vs reference
     attention over packed selected KV.
  4. patch 11 -- qsa_fast_topk (JIT -> sgl_kernel -> reference chain) vs the
     fixed-width reference.
  5. patch 16 -- dense wNa16 load-time dequant (symmetric / asymmetric, with
     and without actorder g_idx) vs a plain reference, plus F.linear output.
  6. patch 17 -- SGLANG_MOE_CONFIG_DIR search path (flat and
     configs/triton_<ver>/ layouts, missing directories, precedence).
  7. patch 18 -- split (atomics-free) HyperConnection mix selected on HIP,
     vs the fp32 reference, and bit-identical across repeated launches.
"""
from __future__ import annotations

import sys

import torch
import triton.language as tl

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. fused-MoE zero points (patch 12)
# ---------------------------------------------------------------------------
def test_moe_zp():
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    E, K, N, G = 8, 256, 512, 32
    M, topk = 17, 2
    kg = K // G
    dev = "cuda"

    # Build quantized weights that EXACTLY represent dequant values with
    # per-(expert, group, n) zero points, stored in the loader's layout:
    # qweights int32 [E, K//8, N] (8 nibbles along K), scales bf16
    # [E, K//G, N], zero points int32 [E, K//G, N//8] (8 nibbles along N).
    q = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=dev)
    zp = torch.randint(3, 12, (E, kg, N), dtype=torch.int32, device=dev)
    scale = torch.rand(E, kg, N, device=dev) * 0.01 + 0.005
    dequant = (
        q.view(E, N, kg, G).float() - zp.float().permute(0, 2, 1).unsqueeze(-1)
    ) * scale.float().permute(0, 2, 1).unsqueeze(-1)
    dequant = dequant.reshape(E, N, K).to(torch.bfloat16)

    # pack weights: byte[e, n, k//2] = q(k) | q(k+1) << 4
    w_uint8 = q.to(torch.uint8).view(E, N, K // 2, 2)
    w_packed = (w_uint8[..., 0] | (w_uint8[..., 1] << 4)).contiguous()  # [E, N, K/2]

    # pack zp the way the checkpoint stores it: int32 [E, kg, N//8], 8 nibbles
    # along N little-first, then run the PATCHED conversion on it.
    zp_nib = zp.view(E, kg, N // 8, 8).to(torch.int64)
    factor = torch.tensor([1, 16, 256, 4096, 65536, 1 << 20, 1 << 24, 1 << 28], device=dev, dtype=torch.int64)
    zp_int32 = (zp_nib * factor.view(1, 1, 1, 8)).sum(-1).to(torch.int32)
    zp_stored = zp_int32.contiguous()

    # scales in stored layout [E, kg, N] bf16
    scale_stored = scale.contiguous().to(torch.bfloat16)

    # --- apply the patched scheme conversion (same code as patch 12) ---
    zp_bytes = zp_stored.view(torch.uint8).transpose(1, 2).contiguous()  # [E, N/2, kg]
    scale_conv = scale_stored.transpose(1, 2).contiguous()  # [E, N, kg]

    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E, (M, topk), device=dev, dtype=torch.int32)
    topk_weights = torch.ones(M, topk, device=dev, dtype=torch.bfloat16)

    block_size = 16
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_size, E)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    em = sorted_ids.shape[0]

    def run(zp_arg):
        out = torch.zeros(em, N, device=dev, dtype=torch.bfloat16)
        invoke_fused_moe_kernel(
            a, w_packed, None, out, None, scale_conv, zp_arg,
            topk_weights, topk_ids, sorted_ids, expert_ids, num_post,
            False, topk, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=True, per_channel_quant=False,
            block_shape=[0, G],
        )
        torch.cuda.synchronize()
        # the kernel writes row `offs_token` = flattened token id t*topk+j
        # (sorted_token_ids holds token ids, not sorted slot numbers)
        return out[: M * topk].float()

    ref = torch.zeros(M * topk, N, device=dev, dtype=torch.float32)
    for t in range(M):
        for j in range(topk):
            ref[t * topk + j] = dequant[topk_ids[t, j]].float() @ a[t].float()

    got = run(zp_bytes)
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=2e-2, rtol=2e-2)
    check("moe_zp_parity", ok, f"max abs err {err:.4g}")

    # negative control: the stored [E, kg, N/8] int32 layout viewed as bytes
    # without the transpose must NOT match (proves the patch-12 relayout is
    # what makes the kernel read the right zero points)
    zp_wrong = zp_stored.view(torch.uint8).contiguous()
    err_wrong = (run(zp_wrong) - ref).abs().max().item()
    check("moe_zp_negative_control", err_wrong > 0.1,
          f"untransposed zp max abs err {err_wrong:.4g} (expected large)")

    # no-zp path (symmetric checkpoints): kernel subtracts constant 8
    dequant8 = ((q.view(E, N, kg, G).float() - 8.0)
                * scale.float().permute(0, 2, 1).unsqueeze(-1)).reshape(E, N, K)
    ref8 = torch.zeros(M * topk, N, device=dev, dtype=torch.float32)
    for t in range(M):
        for j in range(topk):
            ref8[t * topk + j] = dequant8[topk_ids[t, j]] @ a[t].float()
    got8 = run(None)
    err8 = (got8 - ref8).abs().max().item()
    check("moe_nozp_parity", torch.allclose(got8, ref8, atol=2e-2, rtol=2e-2),
          f"max abs err {err8:.4g}")


# ---------------------------------------------------------------------------
# 2. CPU-side PLE gather (patch 11)
# ---------------------------------------------------------------------------
def test_ple_cpu_gather():
    from sglang.srt.models.qwen4_exp import (
        _STRIX_PLE_CPU_GATHER,
        _gather_ple_embedding_on_cpu,
    )

    check("ple_cpu_gather_enabled_on_hip", _STRIX_PLE_CPU_GATHER,
          "SGLANG_PLE_UVA_GATHER must be unset and torch.version.hip set")

    class _Shard:
        org_vocab_start_index = 100
        org_vocab_end_index = 1100

    class _Emb:
        shard_indices = _Shard()
        embedding_dim = 160
        weight = torch.nn.Parameter(
            torch.randn(1000, 160, dtype=torch.bfloat16), requires_grad=False
        )

    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        emb = _Emb()
        emb.weight.data = torch.randn(1000, 160).to(dtype)
        ids = torch.randint(50, 1150, (37,), device="cuda")
        out = torch.empty(37, 160, dtype=torch.bfloat16, device="cuda")
        _gather_ple_embedding_on_cpu(emb, ids, out)
        torch.cuda.synchronize()
        ref = torch.zeros(37, 160, dtype=torch.bfloat16)
        host_ids = ids.cpu()
        mask = (host_ids >= 100) & (host_ids < 1100)
        local = torch.where(mask, host_ids - 100, torch.zeros_like(host_ids))
        rows = emb.weight.data[local].to(torch.bfloat16)
        ref[mask] = rows[mask]
        ok = torch.equal(out.cpu(), ref)
        check(f"ple_cpu_gather_{str(dtype).split('.')[-1]}", ok)


# ---------------------------------------------------------------------------
# 3. QSA decode Triton kernel (patch 11)
# ---------------------------------------------------------------------------
def test_qsa_decode():
    from sglang.kernels.kda_kernels.qwen38_qsa_sm121.kernel import (
        qwen38_qsa_sm121,
    )

    torch.manual_seed(7)
    bs, nq, nk, d = 8, 24, 2, 256
    dev = "cuda"
    cu_list = [0]
    lens = [511, 1024, 77, 2048, 63, 1, 1500, 2055]
    for l in lens:
        cu_list.append(cu_list[-1] + l)
    cu_k = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    total = cu_list[-1]
    packed_k = torch.randn(total, nk, d, device=dev, dtype=torch.bfloat16)
    packed_v = torch.randn(total, nk, d, device=dev, dtype=torch.bfloat16)
    q = torch.randn(bs, nq, d, device=dev, dtype=torch.bfloat16)
    cu_q = torch.arange(bs + 1, dtype=torch.int32, device=dev)
    softmax_scale = d**-0.5

    out = qwen38_qsa_sm121(q, packed_k, packed_v, cu_q, cu_k, softmax_scale)
    torch.cuda.synchronize()

    ref = torch.empty_like(out)
    for b in range(bs):
        lo, hi = cu_list[b], cu_list[b + 1]
        kb = packed_k[lo:hi].transpose(0, 1).float()  # [nk, len, d]
        vb = packed_v[lo:hi].transpose(0, 1).float()
        qb = q[b].float()  # [nq, d]
        # merge kv heads: each kv head serves nq/nk q heads
        per = nq // nk
        outs = []
        for h in range(nk):
            attn = torch.softmax(
                qb[h * per : (h + 1) * per] @ kb[h].transpose(0, 1) * softmax_scale,
                dim=-1,
            )
            outs.append(attn @ vb[h])
        ref[b] = torch.cat(outs, dim=0).to(torch.bfloat16)
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
    check("qsa_decode_parity", ok, f"max abs err {err:.4g}")


# ---------------------------------------------------------------------------
# 4. qsa_fast_topk chain (patch 11)
# ---------------------------------------------------------------------------
def test_fast_topk():
    from sglang.srt.layers.attention.qsa.kernel import (
        _qsa_fixed_width_topk,
        qsa_fast_topk,
    )

    torch.manual_seed(3)
    rows, width, topk = 64, 4096, 512
    dev = "cuda"
    logits = torch.randn(rows, width, device=dev, dtype=torch.float32)
    starts = torch.randint(0, 256, (rows,), device=dev, dtype=torch.int32)
    # len <= width - start: the real path never lets a window exceed the row
    ends = starts + torch.randint(512, 1024, (rows,), device=dev, dtype=torch.int32)
    ends = torch.minimum(ends, torch.full_like(ends, width))
    got = qsa_fast_topk(logits, starts, ends, topk)
    ref = _qsa_fixed_width_topk(logits, (ends - starts).to(torch.int32), starts, topk)
    mism = [
        r for r, (g, rr) in enumerate(zip(got.cpu(), ref.cpu()))
        if not torch.equal(torch.sort(g[g >= 0])[0], torch.sort(rr[rr >= 0])[0])
    ]
    # NOTE: deliberately do NOT invoke the JIT kernel here to see which path
    # served the call -- on RDNA3.5 its out-of-bounds reads poison the HIP
    # queue for every later kernel in the process (that is exactly why patch
    # 11 skips it on HIP).
    check("fast_topk_chain", not mism, f"mismatched rows {mism[:8]}")


# ---------------------------------------------------------------------------
# 5. dense wNa16 load-time dequant (patch 16)
# ---------------------------------------------------------------------------
def test_wna16_dense_dequant():
    """Unpack of compressed-tensors pack-quantized int4 vs a plain reference.

    Covers symmetric / asymmetric (packed zero points along N) with and
    without an actorder g_idx; the layout of every W4A16 dense Linear
    (attention q/k/v/o, shared experts) in public Qwen3.x checkpoints.
    """
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from compressed_tensors.quantization import ActivationOrdering
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16 as wna16,
    )

    if not getattr(wna16, "_WNA16_DENSE_FALLBACK", False):
        check("wna16_dense_dequant", False, "patch 16 fallback flag not present")
        return

    n, k, group = 256, 1024, 128
    ngroups = k // group
    for symmetric in (True, False):
        for actorder in (False, True):
            torch.manual_seed(5)
            q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
            scale = (torch.rand(n, ngroups) * 0.05 + 0.01).to(torch.bfloat16)
            zp = None if symmetric else torch.randint(-8, 8, (n, ngroups), dtype=torch.int8)
            g_idx = (torch.randperm(k, dtype=torch.int32) % ngroups) if actorder else None
            gi = g_idx.long() if actorder else torch.arange(k) // group
            zref = (torch.zeros(n, ngroups) if zp is None else zp.float())[:, gi]
            ref = (q.float() - zref) * scale.float()[:, gi]

            layer = torch.nn.Module()
            p = lambda t: torch.nn.Parameter(t.cuda(), requires_grad=False)  # noqa: E731
            layer.register_parameter("weight_packed", p(pack_to_int32(q, 4)))
            layer.register_parameter("weight_scale", p(scale))
            layer.register_parameter("weight_shape", p(torch.tensor([n, k])))
            if zp is not None:
                layer.register_parameter("weight_zero_point", p(pack_to_int32(zp, 4, packed_dim=0)))
            if actorder:
                layer.register_parameter("weight_g_idx", p(g_idx))
            scheme = wna16.CompressedTensorsWNA16(
                strategy="group", num_bits=4, group_size=group, symmetric=symmetric,
                actorder=ActivationOrdering.GROUP if actorder else None,
            )
            scheme.process_weights_after_loading(layer)
            w = layer.weight.float().cpu()
            # bf16 rounding of (q - zp) * scale is the only allowed difference
            tol = ref.abs().max().item() * 2 ** -7
            err = (w - ref).abs().max().item()
            x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda")
            y = scheme.apply_weights(layer, x, bias=None).float().cpu()
            yref = torch.nn.functional.linear(x.float().cpu(), ref)
            rel = ((y - yref).norm() / yref.norm()).item()
            check(
                f"wna16_dense_dequant sym={symmetric} actorder={actorder}",
                set(layer._parameters) == {"weight"} and err <= tol and rel < 2e-2,
                f"max|dw|={err:.2e} (tol {tol:.2e}) rel(y)={rel:.2e}",
            )


# ---------------------------------------------------------------------------
# 6. SGLANG_MOE_CONFIG_DIR search path (patch 17)
# ---------------------------------------------------------------------------
def test_moe_config_dir():
    """Mounted tile configs win over the builtin tree; missing dirs are skipped.

    Uses a fabricated (E, N) so the builtin tree never has a match, and the
    layout the tuner writes (flat json in a directory) plus upstream's
    configs/triton_<ver>/ layout.
    """
    import json
    import os
    import tempfile

    import triton
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as cfg
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    # get_moe_configs reads the published server args (deterministic flag).
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")

    E, N, dtype, group = 7, 11, "int4_w4a16", 32
    name = cfg.get_config_file_name(E, N, dtype, [0, group])
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"

    def lookup(env):
        cfg.get_moe_configs.cache_clear()
        old = os.environ.pop("SGLANG_MOE_CONFIG_DIR", None)
        if env is not None:
            os.environ["SGLANG_MOE_CONFIG_DIR"] = env
        try:
            return cfg.get_moe_configs(E, N, dtype, 0, group)
        finally:
            os.environ.pop("SGLANG_MOE_CONFIG_DIR", None)
            if old is not None:
                os.environ["SGLANG_MOE_CONFIG_DIR"] = old

    with tempfile.TemporaryDirectory() as flat, tempfile.TemporaryDirectory() as tree:
        with open(os.path.join(flat, name), "w") as f:
            json.dump({"1": {"BLOCK_SIZE_M": 16, "tag": "flat"}}, f)
        os.makedirs(os.path.join(tree, "configs", version_dir))
        with open(os.path.join(tree, "configs", version_dir, name), "w") as f:
            json.dump({"1": {"BLOCK_SIZE_M": 16, "tag": "tree"}}, f)

        try:
            unset = lookup(None)
            check("moe_config_dir unset -> builtin fallback", unset is None, repr(unset))
            missing = lookup("/nonexistent-moe-configs")
            check("moe_config_dir missing dir skipped", missing is None, repr(missing))
            got = lookup(flat)
            check("moe_config_dir flat layout", got == {1: {"BLOCK_SIZE_M": 16, "tag": "flat"}}, repr(got))
            got = lookup(tree)
            check("moe_config_dir configs/triton_ver layout", got == {1: {"BLOCK_SIZE_M": 16, "tag": "tree"}}, repr(got))
            got = lookup(os.pathsep.join(["/nonexistent-moe-configs", tree, flat]))
            check("moe_config_dir first match wins", got == {1: {"BLOCK_SIZE_M": 16, "tag": "tree"}}, repr(got))
        except Exception as e:  # noqa: BLE001
            check("moe_config_dir", False, f"{type(e).__name__}: {e}")
        finally:
            cfg.get_moe_configs.cache_clear()
            reset_context()


def test_hc_mix_deterministic():
    """The split HC mix is selected on HIP, matches the reference, and repeats.

    Model shape (4 x 2560 hidden, rank 320) at decode row counts. The
    reference is the torch formula in fp32; tolerance covers bf16 rounding of
    the intermediate mix weights (the persistent kernel sits at the same
    distance from it).
    """
    import torch.nn.functional as F
    from sglang.srt.layers import hc_mix_triton as m

    check("hc_mix split variant selected on HIP", m._use_split_mix())

    torch.manual_seed(0)
    hc, hs, lowrank = 4, 2560, 320
    dev = "cuda"
    w_down = (torch.randn(lowrank, hc * hs, device=dev) * 0.02).to(torch.bfloat16)
    w_up = (torch.randn(hc * hs, lowrank, device=dev) * 0.05).to(torch.bfloat16)
    for rows in (1, 5, 16):
        x = torch.randn(rows, hc * hs, device=dev).to(torch.bfloat16)
        check(f"hc_mix rows={rows} fused path supported", m.fused_hc_mix_supported(x, w_down, w_up))
        out = m.fused_hc_mix(x, w_down, w_up, hc, hs)
        xf = x.float()
        t = F.silu(F.linear(xf, w_down.float()) / hc)
        gate = torch.sigmoid(F.linear(t, w_up.float())).unflatten(-1, (hc, hs))
        ref = (gate * xf.unflatten(-1, (hc, hs))).mean(dim=-2)
        err = (out.float() - ref).abs().max().item()
        check(f"hc_mix rows={rows} vs fp32 reference", err < 2e-2, f"max abs err {err:.5f}")
        repeats = 100
        same = sum(int(torch.equal(m.fused_hc_mix(x, w_down, w_up, hc, hs), out)) for _ in range(repeats))
        check(f"hc_mix rows={rows} bit-identical across launches", same == repeats, f"{same}/{repeats}")


if __name__ == "__main__":
    test_ple_cpu_gather()
    test_qsa_decode()
    test_fast_topk()
    test_moe_zp()
    test_wna16_dense_dequant()
    test_moe_config_dir()
    test_hc_mix_deterministic()
    print()
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PARITY TESTS PASSED")
