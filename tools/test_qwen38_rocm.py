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
  8. patch 19 -- GPTQ/AWQ MoE kernel with K not a multiple of BLOCK_SIZE_K
     (partial last block) vs the dequantized reference.
  9. patch 20 -- dense decode top-k (graph-capturable replacement for the JIT
     kernel on HIP) vs the reference set, padding, determinism, guard.
 10. patch 21 -- tie-stable block selection: torch.topk's tie order varies
     per launch on ROCm; the sort and key forms of qsa_ordered_topk agree,
     match the reference scores, stay bit-identical on tied relu-style
     scores (decode and prefill shapes), and the decode form captures into
     a CUDA graph.
 11. patch 22 -- greedy rows (top_k <= 1) get their argmax from
     sample_draft_proposal, bit-identical over launches, while the
     fast_sample draw the post-prefill draft used to take varies; the
     rejection-sampling branch of _draft_extend_for_prefill routes through
     sample_draft_proposal.
 12. patch 23 -- QSAMTPSharedSparseIndices.lookup keeps each row a valid
     prefix (tail right after the captured entries, not after the -1
     padding) so the KV gather's count-based packing sees the drafted
     positions; full rows, never-captured rows, the gather's own valid
     counts, and CUDA graph capture.
 13. patch 24 -- Triton decode MQA for the QSA indexer (TileLang is absent, so
     upstream ran the torch reference, whose cost scales with the model's
     context length): vs the reference on shuffled page tables with lengths
     from 0 to full, max_model_len past the page table, -inf mask equality,
     bit-identical over launches, CUDA graph replay, and a timing print
     against the reference at bs 20 / 131k context.
 14. patch 25 -- qsa_stable_rows_topk processes rows in slices: identical
     indices to the whole-chunk sort and to the fixed-width reference on an
     8192-row prefill chunk with tied scores, slice boundaries that do not
     divide the rows, k below and above the width; peak allocation for the
     stage bounded (a fraction of the whole-chunk sort's).
 15. patch 26 -- Triton prefill MQA for the QSA indexer (the torch reference
     materialises the per-head scores, 4x the logits, plus three copies):
     vs the reference on packed rows with empty, partial and full ranges and
     tile-edge shapes, -inf mask equality, bit-identical over launches,
     dispatch and the SGLANG_QSA_MQA_TRITON=0 override, peak allocation at
     the 128 MiB logits budget a fraction of the reference's.
 16. patch 27 -- parameters parked in pinned system memory: the pattern
     list from the env var, the parked Parameter is still a CUDA Parameter
     at the pinned address, embedding / linear forwards and a CUDA graph
     replay equal the VRAM originals, unmatched parameters untouched, torch's
     allocated VRAM drops by the moved bytes, empty pattern list is a no-op.
 17. patch 28 -- PLE short conv over the packed prefill batch (upstream pads
     every request to the longest one, three copies of
     [requests, row_width, channels]): bit-identical conv output,
     next-state and track-slot gathers vs the padded layout on mixed
     lengths (fp32 and bf16), padding rows routed to a scratch column
     without disturbing valid outputs, and peak allocation at the real
     shape (17 x 71 tokens + a 6,024-token chunk, 10,240 channels) under
     0.5 GiB where the padded layout needs 6.2.
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


def test_moe_partial_k():
    """GPTQ/AWQ MoE kernel with K not a multiple of BLOCK_SIZE_K (patch 19).

    K=320 with BLOCK_SIZE_K=128 leaves a 64-row last block, the shape the
    tuner's tp-2 shard produces for the down projection. Symmetric int4,
    group 64. Checks parity against the dequantized reference; the unmasked
    B load it replaces read past the tensor (a GPU page fault when the
    mapping ends there), which only the tuner run itself can exercise.
    """
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(1)
    E, K, N, G = 8, 320, 128, 64
    M, topk = 13, 2
    kg = K // G
    dev = "cuda"
    q = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=dev)
    scale = torch.rand(E, kg, N, device=dev) * 0.01 + 0.005
    dequant = ((q.view(E, N, kg, G).float() - 8.0)
               * scale.float().permute(0, 2, 1).unsqueeze(-1)).reshape(E, N, K)
    w_uint8 = q.to(torch.uint8).view(E, N, K // 2, 2)
    w_packed = (w_uint8[..., 0] | (w_uint8[..., 1] << 4)).contiguous()
    scale_conv = scale.to(torch.bfloat16).transpose(1, 2).contiguous()

    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E, (M, topk), device=dev, dtype=torch.int32)
    topk_weights = torch.ones(M, topk, device=dev, dtype=torch.bfloat16)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 16, E)
    em = sorted_ids.shape[0]
    for block_k in (128, 256):
        config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": block_k,
                  "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2}
        out = torch.zeros(em, N, device=dev, dtype=torch.bfloat16)
        invoke_fused_moe_kernel(
            a, w_packed, None, out, None, scale_conv, None,
            topk_weights, topk_ids, sorted_ids, expert_ids, num_post,
            False, topk, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=True, per_channel_quant=False,
            block_shape=[0, G],
        )
        torch.cuda.synchronize()
        got = out[: M * topk].float()
        ref = torch.zeros(M * topk, N, device=dev, dtype=torch.float32)
        for t in range(M):
            for j in range(topk):
                ref[t * topk + j] = dequant[topk_ids[t, j]] @ a[t].float()
        err = (got - ref).abs().max().item()
        check(f"moe_partial_k K={K} BLOCK_K={block_k}",
              torch.allclose(got, ref, atol=2e-2, rtol=2e-2), f"max abs err {err:.4g}")


def test_qsa_dense_topk():
    """Decode block selection on HIP (patch 20): dense top-k vs the reference.

    Rows of 30, 511, 600 and 2152 compressed blocks in a 4096-wide -inf
    padded logits tensor (what qsa_mqa_decode returns), budget 512: same set
    as the reference, -1 padding for short rows, bit-identical over repeats,
    and the JIT kernel stays off HIP unless SGLANG_QSA_TOPK_JIT=1.
    """
    import os

    from sglang.srt.layers.attention.qsa.kernel import (
        _qsa_fixed_width_topk,
        qsa_dense_topk,
        qsa_jit_topk_allowed,
    )

    old = os.environ.pop("SGLANG_QSA_TOPK_JIT", None)
    try:
        check("qsa_jit_topk_allowed false on HIP", not qsa_jit_topk_allowed())
        os.environ["SGLANG_QSA_TOPK_JIT"] = "1"
        check("qsa_jit_topk_allowed override", qsa_jit_topk_allowed())
    finally:
        os.environ.pop("SGLANG_QSA_TOPK_JIT", None)
        if old is not None:
            os.environ["SGLANG_QSA_TOPK_JIT"] = old

    torch.manual_seed(0)
    dev = "cuda"
    W, topk = 4096, 512
    lengths = torch.tensor([30, 511, 600, 2152], dtype=torch.int32, device=dev)
    logits = torch.full((4, W), float("-inf"), device=dev)
    for r, L in enumerate(lengths.tolist()):
        logits[r, :L] = torch.randn(L, device=dev)
    ref = _qsa_fixed_width_topk(logits, lengths, torch.zeros(4, dtype=torch.int32, device=dev), topk)
    got = qsa_dense_topk(logits, lengths, topk)
    check("qsa_dense_topk shape", tuple(got.shape) == (4, topk) and got.dtype == torch.int32, str(got.shape))
    for r, L in enumerate(lengths.tolist()):
        want = set(ref[r].tolist())
        have = set(got[r].tolist())
        pad_ok = (got[r] == -1).sum().item() == max(0, topk - L)
        check(f"qsa_dense_topk L={L} set matches reference", have == want and pad_ok,
              f"{len(have ^ want)} differing, pad {(got[r] == -1).sum().item()}")
    same = sum(int(torch.equal(qsa_dense_topk(logits, lengths, topk), got)) for _ in range(50))
    check("qsa_dense_topk bit-identical across launches", same == 50, f"{same}/50")


# ---------------------------------------------------------------------------
# 10. tie-stable QSA top-k (patch 21)
# ---------------------------------------------------------------------------
def _tied_scores(rows, width, lengths, dev):
    """relu-style indexer scores: ~half exact zeros, bf16-rounded, -inf tail."""
    x = torch.relu(torch.randn(rows, width, device=dev) - 0.3)
    x = x.to(torch.bfloat16).float()
    cols = torch.arange(width, device=dev).unsqueeze(0)
    return x.masked_fill(cols >= lengths.unsqueeze(1), float("-inf"))


def test_qsa_topk_ties():
    """Block selection with tied scores (patch 21).

    Indexer scores are relu sums, so exact ties are common; torch.topk on
    ROCm orders tied entries differently per launch. The sort and key forms
    of qsa_ordered_topk must agree, select the same scores as the reference,
    break ties toward the lower index, be bit-identical over repeats, and the
    key form must capture into a CUDA graph.
    """
    from sglang.srt.layers.attention.qsa.kernel import (
        _qsa_fixed_width_topk,
        qsa_dense_topk,
        qsa_ordered_topk,
        qsa_stable_rows_topk,
    )

    torch.manual_seed(0)
    dev = "cuda"
    topk = 512

    # decode shape: (bs, 4096) dense rows of various lengths
    lengths = torch.tensor([30, 511, 600, 2152, 4096], dtype=torch.int32, device=dev)
    logits = _tied_scores(5, 4096, lengths, dev)
    zeros = (logits == 0).sum().item()
    check("tied test data has exact zeros", zeros > 1000, f"{zeros} zeros")
    a = qsa_ordered_topk(logits, topk, use_sort=True)
    b = qsa_ordered_topk(logits, topk, use_sort=False)
    check("qsa_ordered_topk sort == key indices", torch.equal(a, b),
          f"{(a != b).sum().item()} differ")
    ref_vals = torch.topk(logits, topk, dim=1).values
    check("qsa_ordered_topk selects the reference scores",
          torch.equal(torch.gather(logits, 1, a), ref_vals))
    # ties toward the lower index: within each run of equal scores, indices ascend
    va = torch.gather(logits, 1, a)
    tie = va[:, 1:] == va[:, :-1]
    check("qsa_ordered_topk ties break toward the lower index",
          bool((a[:, 1:] > a[:, :-1])[tie].all()))
    for use_sort in (True, False):
        first = qsa_ordered_topk(logits, topk, use_sort=use_sort)
        same = sum(int(torch.equal(qsa_ordered_topk(logits, topk, use_sort=use_sort), first)) for _ in range(30))
        check(f"qsa_ordered_topk use_sort={use_sort} bit-identical", same == 30, f"{same}/30")
    plain = torch.topk(logits, topk, dim=1).indices
    varies = sum(int(not torch.equal(torch.topk(logits, topk, dim=1).indices, plain)) for _ in range(30))
    print(f"       (plain torch.topk on the same tied rows: {varies}/30 launches differ)")

    got = qsa_dense_topk(logits, lengths, topk)
    ref = _qsa_fixed_width_topk(logits, lengths, torch.zeros(5, dtype=torch.int32, device=dev), topk)
    ok = True
    for r, L in enumerate(lengths.tolist()):
        w = min(L, topk)
        ok &= torch.equal(torch.gather(logits[r], 0, got[r, :w].long()),
                          torch.gather(logits[r], 0, ref[r, :w].long()))
        ok &= bool((got[r, w:] == -1).all()) and bool((got[r, :w] >= 0).all())
    check("qsa_dense_topk tied rows: reference scores + padding", ok)
    same = sum(int(torch.equal(qsa_dense_topk(logits, lengths, topk), got)) for _ in range(30))
    check("qsa_dense_topk tied rows bit-identical", same == 30, f"{same}/30")
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        qsa_dense_topk(logits, lengths, topk)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        g_out = qsa_dense_topk(logits, lengths, topk)
    graph.replay()
    torch.cuda.synchronize()
    check("qsa_dense_topk graph replay matches eager", torch.equal(g_out, got))

    # prefill shape: packed rows with non-zero starts and growing lengths
    rows, width = 300, 700
    starts = torch.cat([torch.zeros(150, dtype=torch.int32, device=dev),
                        torch.full((150,), 300, dtype=torch.int32, device=dev)])
    lengths = torch.cat([torch.arange(150, device=dev, dtype=torch.int32) * 2,
                         torch.arange(150, device=dev, dtype=torch.int32) * 2 + 100])
    full = torch.relu(torch.randn(rows, width, device=dev) - 0.3).to(torch.bfloat16).float()
    for k in (512, 64):
        got = qsa_stable_rows_topk(full, lengths, starts, k)
        ref = _qsa_fixed_width_topk(full, lengths, starts, k)
        ok = tuple(got.shape) == (rows, k) and got.dtype == torch.int32
        for r in range(rows):
            w = min(int(lengths[r]), k)
            s0 = int(starts[r])
            ok &= bool((got[r, w:] == -1).all()) and bool((got[r, :w] >= 0).all())
            ok &= bool((got[r, :w] < int(lengths[r])).all())
            ok &= torch.equal(full[r, s0 + got[r, :w].long()], full[r, s0 + ref[r, :w].long()])
        check(f"qsa_stable_rows_topk k={k}: reference scores, relative indices, padding", ok)
        same = sum(int(torch.equal(qsa_stable_rows_topk(full, lengths, starts, k), got)) for _ in range(30))
        check(f"qsa_stable_rows_topk k={k} bit-identical", same == 30, f"{same}/30")


# ---------------------------------------------------------------------------
# 11. greedy first draft under rejection sampling (patch 22)
# ---------------------------------------------------------------------------
def test_spec_draft_greedy():
    """Draft proposal for greedy rows (patch 22).

    HIP turns speculative_use_rejection_sampling on, so every draft goes
    through sample_draft_proposal, which hands a top_k <= 1 row its argmax.
    The post-prefill draft (_draft_extend_for_prefill) called fast_sample
    directly and drew a random first draft token at temperature 0. Check the
    proposal is the argmax and bit-identical for greedy rows on a sharp but
    not one-hot distribution where fast_sample visibly varies, and that the
    rejection-sampling branch of _draft_extend_for_prefill now goes through
    sample_draft_proposal.
    """
    import importlib.util

    from sglang.srt.speculative.spec_utils import fast_sample, sample_draft_proposal

    torch.manual_seed(0)
    dev = "cuda"
    bs, vocab = 6, 4096
    logits = torch.randn(bs, vocab, device=dev) * 3
    # a runner-up within ~1.5 nats of the top so a random draw misses it often
    top = logits.argmax(dim=-1)
    logits.scatter_(1, ((top + 1) % vocab).unsqueeze(1), logits.max(dim=1, keepdim=True).values - 1.5)
    temps = torch.ones(bs, 1, device=dev)
    top_ks = torch.tensor([1, 1, 1, 50, 1, 50], device=dev, dtype=torch.int32)
    greedy = top_ks <= 1

    probs, p, idx = sample_draft_proposal(logits, temps, top_ks)
    check("sample_draft_proposal greedy rows pick the argmax",
          torch.equal(idx.view(-1)[greedy], top[greedy]))
    check("sample_draft_proposal returns the argmax probability",
          torch.allclose(p.view(-1)[greedy], probs.max(dim=1).values[greedy]))
    same = sum(int(torch.equal(sample_draft_proposal(logits, temps, top_ks)[2].view(-1)[greedy], top[greedy]))
               for _ in range(30))
    check("sample_draft_proposal greedy rows bit-identical", same == 30, f"{same}/30")
    misses = sum(int((fast_sample(probs, num_samples=1)[1].view(-1)[greedy] != top[greedy]).any())
                 for _ in range(30))
    print(f"       (fast_sample on the same greedy rows misses the argmax in {misses}/30 draws)")

    spec = importlib.util.find_spec("sglang.srt.speculative.eagle_worker_v2")
    src = open(spec.origin).read()
    start = src.index("def _draft_extend_for_prefill(")
    end = src.find("\n    def ", start + 1)
    body = src[start:end if end > 0 else None]
    rs = body[body.index("use_rejection_sampling = get_spec()"):]
    branch = rs[: rs.index("else:")]
    check("_draft_extend_for_prefill rejection-sampling branch uses sample_draft_proposal",
          "sample_draft_proposal(" in branch and "fast_sample(" not in branch)


# ---------------------------------------------------------------------------
# 12. MTP shared sparse indices: tail placement (patch 23)
# ---------------------------------------------------------------------------
def test_qsa_mtp_tail():
    """QSAMTPSharedSparseIndices.lookup rows stay a valid prefix (patch 23).

    The KV gather (_compact_kv) packs column c only if c < valid_count, so
    the drafted positions appended by lookup have to follow the captured
    row's valid entries. Rows: a short capture with -1 padding, a full row,
    and a never-captured row (zeros, captured_len 1). At each draft step the
    row must be captured entries, then the tail positions <= current, then
    -1; the FA2 valid-count kernel's count must equal the prefix length; and
    the lookup must capture into a CUDA graph.
    """
    from sglang.srt.layers.attention.qsa.sparse_attn import qwen_sparse_fa2_cu_seqlens_triton
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QSAMTPSharedSparseIndices

    dev = "cuda"
    topk, tail_width = 51, 4
    state = QSAMTPSharedSparseIndices(layer_ids=[0], num_requests=3, token_topk=topk,
                                      tail_width=tail_width, device=dev)
    short = torch.full((topk,), -1, dtype=torch.int32, device=dev)
    short[:19] = torch.arange(19, device=dev, dtype=torch.int32)
    full = torch.arange(100, 100 + topk, device=dev, dtype=torch.int32)
    state.capture(torch.stack([short, full]), torch.tensor([1, 2], device=dev),
                  torch.tensor([19, 2000], device=dev), layer_id=0)
    reqs = torch.tensor([1, 2, 0], device=dev, dtype=torch.int32)
    captured = [short, full, torch.zeros(topk, dtype=torch.int32, device=dev)]
    bases = [19, 2000, 1]
    width = topk + tail_width
    for step in range(tail_width + 1):
        pos = torch.tensor([b + step for b in bases], device=dev, dtype=torch.int64)
        out = state.lookup(reqs, pos, layer_id=0)
        ok = tuple(out.shape) == (3, width)
        counts = torch.empty(3, dtype=torch.int32, device=dev)
        cu_k = torch.empty(4, dtype=torch.int32, device=dev)
        qwen_sparse_fa2_cu_seqlens_triton(pos.to(torch.int32) + 1, out, counts, cu_k, 3, width)
        for r in range(3):
            n = int((captured[r] >= 0).sum())
            t = min(step + 1, tail_width)
            want = torch.cat([captured[r][:n],
                              torch.arange(bases[r], bases[r] + t, device=dev, dtype=torch.int32),
                              torch.full((width - n - t,), -1, dtype=torch.int32, device=dev)])
            ok &= torch.equal(out[r], want)
            ok &= int(counts[r]) == n + t
        check(f"mtp lookup step {step}: captured, tail, -1 prefix layout; gather count matches", ok,
              "" if ok else f"got {out[:, :24].tolist()} ... counts {counts.tolist()}")
    check("mtp lookup leaves the stored selection untouched",
          torch.equal(state.indices[0, 1, :topk], short) and torch.equal(state.indices[0, 2, :topk], full))

    pos = torch.tensor([21, 2002, 3], device=dev, dtype=torch.int64)
    eager = state.lookup(reqs, pos, layer_id=0)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        state.lookup(reqs, pos, layer_id=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = state.lookup(reqs, pos, layer_id=0)
    graph.replay()
    torch.cuda.synchronize()
    check("mtp lookup graph replay matches eager", torch.equal(g_out, eager))
    pos.fill_(0); pos += torch.tensor([19, 2000, 1], device=dev)
    graph.replay()
    torch.cuda.synchronize()
    check("mtp lookup graph replay follows the position buffer",
          torch.equal(g_out, state.lookup(reqs, pos, layer_id=0)))


# ---------------------------------------------------------------------------
# 13. Triton decode MQA for the QSA indexer (patch 24)
# ---------------------------------------------------------------------------
def test_qsa_mqa_triton():
    """Length-bounded decode MQA (patch 24) vs the torch reference.

    Compressed page size 16, 4 index heads of 128 (the Qwen3.8 layout).
    Shuffled page tables; context lengths of 0, 1, page fractions, tile
    edges and full rows; a max_model_len wider than the page table. Finite
    scores within 1e-3 of the reference, identical -inf masks, bit-identical
    over 20 launches, graph replay equal to eager and following the
    context_lens buffer, and the dispatcher picks the Triton path unless
    SGLANG_QSA_MQA_TRITON=0.
    """
    import math
    import os

    from sglang.srt.layers.attention.qsa import mqa

    torch.manual_seed(0)
    dev = "cuda"
    heads, hd, page, ratio = 4, 128, 16, 4

    def make(bs, max_pages, lens, max_model_len=None):
        npages = bs * max_pages + 3
        kc = torch.randn(npages, page, 1, hd, device=dev, dtype=torch.bfloat16)
        pt = torch.randperm(npages, device=dev)[: bs * max_pages].reshape(bs, max_pages).to(torch.int32)
        q = torch.randn(bs, heads, hd, device=dev, dtype=torch.bfloat16)
        ln = torch.tensor(lens, dtype=torch.int32, device=dev)
        return q, kc, pt, ln, max_model_len or max_pages * page

    check("qsa_mqa_decode: TileLang absent, Triton path used",
          not mqa.HAS_TILELANG and mqa.qsa_mqa_triton_allowed())
    for bs, max_pages, lens, mml in [
        (7, 8, [0, 1, 17, 63, 64, 65, 128], None),
        (3, 512, [8192, 5000, 1], None),
        (2, 8, [100, 128], 200),
        (20, 2048, [64] * 20, None),
    ]:
        q, kc, pt, ln, m = make(bs, max_pages, lens, mml)
        ref = mqa.torch_qsa_mqa_decode(q, kc, pt, ln, m)
        got = mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)
        fin = torch.isfinite(ref)
        mask_ok = torch.equal(fin, torch.isfinite(got)) and got.shape == ref.shape
        close = bool(torch.allclose(ref[fin], got[fin], atol=1e-3, rtol=1e-4)) if fin.any() else True
        err = (ref[fin] - got[fin]).abs().max().item() if fin.any() else 0.0
        same = all(torch.equal(mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m), got) for _ in range(20))
        check(f"triton_qsa_mqa_decode bs={bs} width={m}: reference match, mask, bit-identical",
              mask_ok and close and same, f"max err {err:.2e}, mask {mask_ok}, identical {same}")
    q, kc, pt, ln, m = make(4, 64, [10, 500, 1024, 0])
    via_dispatch = mqa.qsa_mqa_decode(q, kc, pt, ln, m)
    check("qsa_mqa_decode dispatches to the Triton kernel",
          torch.equal(via_dispatch, mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)))
    old = os.environ.get("SGLANG_QSA_MQA_TRITON")
    try:
        os.environ["SGLANG_QSA_MQA_TRITON"] = "0"
        check("SGLANG_QSA_MQA_TRITON=0 selects the reference", not mqa.qsa_mqa_triton_allowed())
    finally:
        os.environ.pop("SGLANG_QSA_MQA_TRITON", None)
        if old is not None:
            os.environ["SGLANG_QSA_MQA_TRITON"] = old

    eager = mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)
    graph.replay()
    torch.cuda.synchronize()
    check("triton_qsa_mqa_decode graph replay matches eager", torch.equal(g_out, eager))
    ln.copy_(torch.tensor([1024, 3, 700, 64], dtype=torch.int32, device=dev))
    graph.replay()
    torch.cuda.synchronize()
    check("triton_qsa_mqa_decode graph replay follows the context_lens buffer",
          torch.equal(g_out, mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m)))

    def timeit(fn, n=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        ts.sort()
        return ts[len(ts) // 2]

    max_pages = math.ceil(131072 / ratio / page)
    q, kc, pt, ln, m = make(20, max_pages, [64] * 20)
    t_ref = timeit(lambda: mqa.torch_qsa_mqa_decode(q, kc, pt, ln, m))
    t_tri = timeit(lambda: mqa.triton_qsa_mqa_decode(q, kc, pt, ln, m))
    print(f"       (decode MQA, bs 20, 131k context, 256-token rows: reference {t_ref:.3f} ms, Triton {t_tri:.3f} ms per layer)")
    check("triton_qsa_mqa_decode faster than the reference at bs 20 / 131k", t_tri < t_ref)


# ---------------------------------------------------------------------------
# 14. row-sliced prefill block selection (patch 25)
# ---------------------------------------------------------------------------
def test_qsa_topk_slices():
    """Row-sliced qsa_stable_rows_topk (patch 25).

    The prefill block selection sorted the whole 8192-row chunk in one
    stable sort, about 1 GiB of transient buffers per QSA layer at a few
    thousand blocks per row. Sorting row slices must give the same indices
    (the selection is per row) and bound the stage's peak allocation.
    """
    from sglang.srt.layers.attention.qsa import kernel as K

    check("patch 25 present (QSA_TOPK_SLICE_ELEMENTS)", hasattr(K, "QSA_TOPK_SLICE_ELEMENTS"))
    if not hasattr(K, "QSA_TOPK_SLICE_ELEMENTS"):
        return
    torch.manual_seed(0)
    dev = "cuda"

    def whole_chunk(logits, lengths, starts, topk):
        # patch 21's single-call form, kept here as the reference
        rows, width = logits.shape
        k = min(topk, width)
        s = starts.to(device=dev, dtype=torch.int64).unsqueeze(1)
        n = lengths.to(device=dev, dtype=torch.int64).unsqueeze(1)
        cols = torch.arange(width, device=dev).unsqueeze(0)
        inside = (cols >= s) & (cols < s + n)
        order = K.qsa_ordered_topk(logits.masked_fill(~inside, float("-inf")), k, use_sort=True)
        keep = torch.arange(k, device=dev).unsqueeze(0) < n.clamp_max(k)
        out = torch.where(keep, order - s, torch.full_like(order, -1)).to(torch.int32)
        if k < topk:
            out = torch.cat([out, out.new_full((rows, topk - k), -1)], dim=1)
        return out

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        return out, (torch.cuda.max_memory_allocated() - base) / 2**20

    # realistic prefill chunk: 8192 query rows, ~6.6k blocks, tied relu scores
    for rows, width, topk in [(8192, 6656, 2048), (1023, 1000, 2048), (300, 700, 64), (7, 5000, 512)]:
        starts = torch.randint(0, max(1, width // 4), (rows,), device=dev, dtype=torch.int32)
        lengths = torch.minimum(torch.randint(0, width, (rows,), device=dev, dtype=torch.int32), width - starts)
        logits = torch.relu(torch.randn(rows, width, device=dev) - 0.3).to(torch.bfloat16).float()
        ref, p_ref = peak(lambda: whole_chunk(logits, lengths, starts, topk))
        got, p_got = peak(lambda: K.qsa_stable_rows_topk(logits, lengths, starts, topk))
        slices = -(-rows // max(1, K.QSA_TOPK_SLICE_ELEMENTS // width))
        check(f"qsa_stable_rows_topk sliced == whole-chunk ({rows}x{width}, k={topk}, {slices} slices)",
              torch.equal(got, ref), f"{(got != ref).sum().item()} differ")
        fixed = K._qsa_fixed_width_topk(logits, lengths, starts, topk) if rows <= 1023 else None
        if fixed is not None:
            ok = True
            for r in range(rows):
                w = min(int(lengths[r]), topk)
                s0 = int(starts[r])
                ok &= bool((got[r, w:] == -1).all())
                ok &= torch.equal(logits[r, s0 + got[r, :w].long()], logits[r, s0 + fixed[r, :w].long()])
            check(f"qsa_stable_rows_topk sliced selects the reference scores ({rows}x{width})", ok)
        if rows == 8192:
            print(f"       (8192x{width} chunk: whole-chunk sort peak {p_ref:.0f} MiB, sliced {p_got:.0f} MiB)")
            check("sliced top-k peak allocation under a third of the whole-chunk sort", p_got < p_ref / 3)
    same = sum(int(torch.equal(K.qsa_stable_rows_topk(logits, lengths, starts, topk), got)) for _ in range(20))
    check("qsa_stable_rows_topk sliced bit-identical", same == 20, f"{same}/20")


# ---------------------------------------------------------------------------
# 15. Triton prefill MQA for the QSA indexer (patch 26)
# ---------------------------------------------------------------------------
def test_qsa_mqa_prefill_triton():
    """Prefill indexer scoring (patch 26) vs the torch reference.

    Packed rows with per-row [start, end) key ranges, including empty, full
    and tile-edge cases. Finite scores within 1e-3 of the reference (bf16
    products are exact in fp32; only the summation order differs), identical
    -inf masks, bit-identical over launches, dispatcher picks the Triton path
    unless SGLANG_QSA_MQA_TRITON=0, and the stage's peak allocation at the
    upstream 128 MiB logits budget is a fraction of the reference's.
    """
    import os

    from sglang.srt.layers.attention.qsa import mqa

    check("patch 26 present (triton_qsa_mqa_prefill)", hasattr(mqa, "triton_qsa_mqa_prefill"))
    if not hasattr(mqa, "triton_qsa_mqa_prefill"):
        return
    torch.manual_seed(0)
    dev = "cuda"

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        return out, (torch.cuda.max_memory_allocated() - base) / 2**20

    def ranges(rows, keys):
        ends = torch.randint(0, keys + 1, (rows,), device=dev, dtype=torch.int32)
        starts = (ends - torch.randint(0, keys + 1, (rows,), device=dev, dtype=torch.int32)).clamp_min(0)
        if rows > 3:
            starts[0], ends[0] = 0, keys          # full row
            starts[1], ends[1] = 0, 0             # empty row
            starts[2], ends[2] = keys, keys       # empty row at the end
        return starts, ends

    for rows, keys in [(3072, 10715), (8192, 2048), (1000, 333), (65, 64), (64, 65), (7, 5), (1, 1)]:
        q = torch.randn(rows, 4, 128, device=dev).to(torch.bfloat16)
        k = torch.randn(keys, 1, 128, device=dev).to(torch.bfloat16)
        starts, ends = ranges(rows, keys)
        ref, p_ref = peak(lambda: mqa.torch_qsa_mqa_prefill(q, k, starts, ends))
        got, p_got = peak(lambda: mqa.triton_qsa_mqa_prefill(q, k, starts, ends))
        fin = torch.isfinite(ref)
        err = (ref[fin] - got[fin]).abs().max().item() if fin.any() else 0.0
        check(f"triton_qsa_mqa_prefill {rows}x{keys}: -inf mask equals the reference",
              torch.equal(torch.isinf(ref), torch.isinf(got)))
        check(f"triton_qsa_mqa_prefill {rows}x{keys}: finite scores within 1e-3", err < 1e-3, f"max abs err {err:.2e}")
        if rows == 3072:
            print(f"       (3072x{keys} chunk, the 128 MiB logits budget: reference peak {p_ref:.0f} MiB, Triton {p_got:.0f} MiB)")
            check("triton_qsa_mqa_prefill peak allocation under a quarter of the reference", p_got < p_ref / 4)
    same = sum(int(torch.equal(mqa.triton_qsa_mqa_prefill(q, k, starts, ends), got)) for _ in range(20))
    check("triton_qsa_mqa_prefill bit-identical", same == 20, f"{same}/20")

    q = torch.randn(64, 4, 128, device=dev).to(torch.bfloat16)
    k = torch.randn(300, 1, 128, device=dev).to(torch.bfloat16)
    s = torch.zeros(64, dtype=torch.int32, device=dev)
    e = torch.full((64,), 300, dtype=torch.int32, device=dev)
    check("qsa_mqa_prefill dispatches to the Triton path",
          torch.equal(mqa.qsa_mqa_prefill(q, k, s, e), mqa.triton_qsa_mqa_prefill(q, k, s, e)))
    prev = os.environ.get("SGLANG_QSA_MQA_TRITON")
    os.environ["SGLANG_QSA_MQA_TRITON"] = "0"
    try:
        check("SGLANG_QSA_MQA_TRITON=0 sends prefill to the reference",
              torch.equal(mqa.qsa_mqa_prefill(q, k, s, e), mqa.torch_qsa_mqa_prefill(q, k, s, e)))
    finally:
        if prev is None:
            del os.environ["SGLANG_QSA_MQA_TRITON"]
        else:
            os.environ["SGLANG_QSA_MQA_TRITON"] = prev


# ---------------------------------------------------------------------------
# 16. parameters parked in pinned system memory (patch 27)
# ---------------------------------------------------------------------------
def test_host_parked_params():
    """Weights in pinned system memory aliased as CUDA tensors (patch 27).

    A small model with an embedding, a two-layer 'vision' stack and an
    lm_head: park the first two, check the Parameters stay CUDA Parameters
    at the pinned address, forwards and a CUDA graph replay equal the VRAM
    originals, lm_head is untouched, torch's allocated VRAM drops by the
    moved bytes, the buffers are exactly the parameter sizes, a second
    model sharing the Parameter parks nothing again, replacing a parked
    Parameter frees its host buffer, and an empty pattern list moves
    nothing.
    """
    try:
        from sglang.srt.model_executor import host_parked_params as hp
    except ImportError:
        check("patch 27 present (host_parked_params)", False)
        return
    check("patch 27 present (host_parked_params)", True)
    torch.manual_seed(0)
    # Inference only, as in the server: an autograd graph would keep the
    # replaced Parameter's storage alive through the saved tensors.
    torch.set_grad_enabled(False)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4096, 256, dtype=torch.bfloat16)
            self.visual = torch.nn.Sequential(
                torch.nn.Linear(128, 256, dtype=torch.bfloat16),
                torch.nn.Linear(256, 128, dtype=torch.bfloat16),
            )
            self.lm_head = torch.nn.Linear(256, 4096, bias=False, dtype=torch.bfloat16)

    m = M().cuda()
    ids = torch.randint(0, 4096, (20,), device="cuda")
    x = torch.randn(8, 128, device="cuda").to(torch.bfloat16)
    ref_e, ref_v = m.embed_tokens(ids), m.visual(x)
    ref_h = m.lm_head(ref_e)
    head_ptr = m.lm_head.weight.data_ptr()
    check("host_parked_patterns parses the env format",
          hp.host_parked_patterns(" embed_tokens.weight, visual. ,") == ["embed_tokens.weight", "visual."]
          and hp.host_parked_patterns("") == [])
    check("empty pattern list moves nothing", hp.park_model_parameters(m, []) == 0)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    expect = sum(p.numel() * p.element_size() for n, p in m.named_parameters() if "embed_tokens.weight" in n or "visual." in n)
    moved = hp.park_model_parameters(m, ["embed_tokens.weight", "visual."])
    torch.cuda.synchronize()
    check("moved bytes match the selected parameters", moved == expect, f"{moved} vs {expect}")
    check("torch allocated VRAM dropped by the moved bytes",
          before - torch.cuda.memory_allocated() >= moved * 0.95,
          f"{(before - torch.cuda.memory_allocated()) / 2**20:.1f} MiB freed of {moved / 2**20:.1f}")
    w = m.embed_tokens.weight
    check("parked weight is still a CUDA Parameter",
          isinstance(w, torch.nn.Parameter) and w.is_cuda and w.dtype == torch.bfloat16)
    parked = [(n, p) for n, p in m.named_parameters() if "embed_tokens.weight" in n or "visual." in n]
    check("parked weights resolve to a host buffer at their address",
          all(hp.is_parked(p) and hp.parked_buffer(p).data_ptr() == p.data_ptr() for _, p in parked)
          and not hp.is_parked(m.lm_head.weight))
    check("buffers are exactly the parameter sizes (no power-of-two rounding)",
          all(hp.parked_buffer(p).nbytes == p.numel() * p.element_size() for _, p in parked))
    shared = torch.nn.Module()
    shared.embed_tokens = m.embed_tokens
    check("a second model sharing the Parameter parks nothing again",
          hp.park_model_parameters(shared, ["embed_tokens.weight"]) == 0)
    check("embedding from parked weight equals the VRAM original", torch.equal(m.embed_tokens(ids), ref_e))
    check("vision stack from parked weights equals the VRAM original", torch.equal(m.visual(x), ref_v))
    check("lm_head untouched", m.lm_head.weight.data_ptr() == head_ptr and torch.equal(m.lm_head(ref_e), ref_h))
    import gc, weakref
    del parked  # the list above holds the Parameter that is about to be replaced
    gone = weakref.ref(hp.parked_buffer(m.visual[0].bias))
    m.visual[0].bias = torch.nn.Parameter(torch.zeros(256, dtype=torch.bfloat16, device="cuda"))
    gc.collect()
    check("replacing a parked Parameter frees its host buffer", gone() is None)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        m.embed_tokens(ids)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        out = m.embed_tokens(ids)
    g.replay()
    torch.cuda.synchronize()
    check("graph replay reads the parked weight", torch.equal(out, ref_e))


def test_ple_short_conv_packed():
    """PLE short conv over the packed prefill batch (patch 28)."""
    try:
        from sglang.srt.models.qwen4_exp import _packed_short_conv
    except ImportError:
        check("patch 28 present (_packed_short_conv)", False)
        return
    check("patch 28 present (_packed_short_conv)", True)
    import torch.nn.functional as F
    torch.manual_seed(0)
    K, d = 4, 3
    S = (K - 1) * d

    def padded_reference(x, state, weight, lengths, req_indices, token_offsets, row_width):
        R, C = lengths.shape[0], x.shape[1]
        padded_seq = x.new_zeros((R, row_width, C))
        padded_seq[req_indices, token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(conv_input, weight, bias=None, dilation=d, groups=C).transpose(1, 2)
        state_cols = torch.arange(S, device=x.device, dtype=torch.long)

        def gather_at(offsets):
            return conv_input.gather(
                2, (offsets.unsqueeze(1) + state_cols.unsqueeze(0)).unsqueeze(1).expand(-1, C, -1))

        return conv_output[req_indices, token_offsets], gather_at

    def batch_meta(lengths):
        pos = torch.arange(int(lengths.sum()), device=lengths.device)
        qsl = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)])
        req = (torch.searchsorted(qsl, pos, right=True) - 1).clamp(0, lengths.shape[0] - 1)
        off = pos - qsl.index_select(0, req)
        return req, off, off < lengths.index_select(0, req)

    C = 64
    for dtype in (torch.float32, torch.bfloat16):
        for lens in ([1, 7, 300, 2, 64], [71] * 17 + [6024], [5], [1, 1, 1], [777]):
            lengths = torch.tensor(lens, device="cuda")
            req, off, valid = batch_meta(lengths)
            x = torch.randn(int(lengths.sum()), C, device="cuda").to(dtype)
            state = torch.randn(len(lens), C, S, device="cuda").to(dtype)
            w = (torch.randn(C, 1, K, device="cuda") * 0.3).to(dtype)
            ref_y, ref_g = padded_reference(x, state, w, lengths, req, off, max(lens))
            y, g = _packed_short_conv(x, state, w, d, S, lengths, req, off, valid)
            track = lengths // 2
            check(f"packed short conv == padded layout ({str(dtype).split('.')[-1]}, {len(lens)} reqs, longest {max(lens)})",
                  torch.equal(y, ref_y) and torch.equal(g(lengths), ref_g(lengths)) and torch.equal(g(track), ref_g(track)),
                  f"max abs err {(y.float() - ref_y.float()).abs().max().item():.2e}")

    lengths = torch.tensor([3, 5], device="cuda")
    req, off, valid = batch_meta(lengths)
    x = torch.randn(12, 16, device="cuda")
    req = torch.cat([req, torch.full((4,), 1, device="cuda")])
    off = torch.cat([off, torch.arange(5, 9, device="cuda")])
    valid = torch.cat([valid, torch.zeros(4, dtype=torch.bool, device="cuda")])
    state = torch.randn(2, 16, S, device="cuda")
    w = torch.randn(16, 1, K, device="cuda")
    y, g = _packed_short_conv(x, state, w, d, S, lengths, req, off, valid)
    ref_y, ref_g = padded_reference(x[:8], state, w, lengths, req[:8], off[:8], 5)
    check("padding rows leave valid outputs and next state untouched",
          torch.equal(y[:8], ref_y) and torch.isfinite(y[8:]).all().item() and torch.equal(g(lengths), ref_g(lengths)))

    C = 10240
    lengths = torch.tensor([71] * 17 + [6024], device="cuda")
    req, off, valid = batch_meta(lengths)
    x = torch.randn(int(lengths.sum()), C, device="cuda").to(torch.bfloat16)
    state = torch.randn(18, C, S, device="cuda").to(torch.bfloat16)
    w = (torch.randn(C, 1, K, device="cuda") * 0.3).to(torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y, g = _packed_short_conv(x, state, w, d, S, lengths, req, off, valid)
    g(lengths)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    padded = 3 * 18 * 6024 * C * 2
    check("peak allocation at the real mixed shape under 0.5 GiB", peak < 0.5 * 2**30,
          f"{peak / 2**20:.0f} MiB (padded layout: {padded / 2**30:.1f} GiB)")


if __name__ == "__main__":
    test_ple_cpu_gather()
    test_qsa_decode()
    test_fast_topk()
    test_moe_zp()
    test_wna16_dense_dequant()
    test_moe_config_dir()
    test_hc_mix_deterministic()
    test_moe_partial_k()
    test_qsa_dense_topk()
    test_qsa_topk_ties()
    test_spec_draft_greedy()
    test_qsa_mtp_tail()
    test_qsa_mqa_triton()
    test_qsa_topk_slices()
    test_qsa_mqa_prefill_triton()
    test_host_parked_params()
    test_ple_short_conv_packed()
    print()
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PARITY TESTS PASSED")
