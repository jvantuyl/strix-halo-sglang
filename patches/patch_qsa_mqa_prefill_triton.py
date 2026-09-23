#!/usr/bin/env python3
"""gfx1151 patch 26: Triton prefill MQA for the QSA indexer, output-only memory.

`qsa_mqa_prefill` scores every compressed block of a packed prefill batch
against every query row for the indexer's block selection. Without TileLang
(patch 24) the torch reference ran: `einsum("mhd,nd->mnh")` materialises a
fp32 `[rows, keys, heads]` tensor, `relu` copies it, then the head sum, the
scale and the `-inf` mask each copy the `[rows, keys]` result. Upstream
slices the rows so the *logits* stay under 128 MiB per call, but the
reference's intermediates are 4x that plus three more copies: about 1.15 GiB
live per QSA layer per prefill chunk. An allocator snapshot of a 43k-token
prefill on this box (patches <= 25) put 1,152 MiB of the 2,370 MiB transient
peak in this one call, on top of ~90 GiB of resident weights and pools.

The Triton kernel here does one program per (64-row, 64-key tile): a tile
with no key inside any of its rows' `[start, end)` stores `-inf` and exits
after two small loads; a live tile loads its keys once and, per head in a
fixed order, takes `tl.dot` of the bf16 queries and keys with fp32
accumulation (bf16 products are exact in fp32; only the summation order
differs from the reference GEMM, a few ULP), applies relu, sums the heads,
scales and stores scores or `-inf`. Same contract as the reference (fp32
`[rows, keys]`, `-inf` outside each row's range), the logits are the only
allocation, no atomics (bit-identical across launches). Dispatched whenever
TileLang is unavailable on a CUDA/HIP device; `SGLANG_QSA_MQA_TRITON=0`
(patch 24's switch) forces the reference for A/B. Anchors on patch 24's text.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qsa/mqa.py"
text = open(p).read()

# ---- kernel + wrapper, inserted before the prefill dispatcher ---------------
anchor = "\n\ndef qsa_mqa_prefill(\n"
assert text.count(anchor) == 1, "qsa/mqa.py: qsa_mqa_prefill anchor not found"
assert "def triton_qsa_mqa_decode(" in text, "qsa/mqa.py: patch 24 not applied"
kernel = '''

@triton.jit
def _qsa_mqa_prefill_kernel(
    Q,
    K,
    RowStarts,
    RowEnds,
    Logits,
    rows,
    keys,
    scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_lm,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ok = m < rows
    n_ok = n < keys
    starts = tl.load(RowStarts + m, mask=m_ok, other=0)
    ends = tl.load(RowEnds + m, mask=m_ok, other=0)
    valid = m_ok[:, None] & n_ok[None, :] & (n[None, :] >= starts[:, None]) & (
        n[None, :] < ends[:, None]
    )
    out_ptr = Logits + m[:, None] * stride_lm + n[None, :]
    store_mask = m_ok[:, None] & n_ok[None, :]
    if tl.sum(valid.to(tl.int32)) > 0:
        d = tl.arange(0, HEAD_DIM)
        # keys as [HEAD_DIM, BLOCK_N] so the per-head product is a plain dot
        kt = tl.load(
            K + n[None, :] * stride_kn + d[:, None], mask=n_ok[None, :], other=0.0
        )
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for h in tl.static_range(HEADS):
            q = tl.load(
                Q + m[:, None] * stride_qm + h * stride_qh + d[None, :],
                mask=m_ok[:, None],
                other=0.0,
            )
            s = tl.dot(q, kt, out_dtype=tl.float32)
            acc += tl.maximum(s, 0.0)
        out = tl.where(valid, acc / scale, float("-inf"))
        tl.store(out_ptr, out, mask=store_mask)
    else:
        pad = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
        tl.store(out_ptr, pad, mask=store_mask)


def triton_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """gfx1151 patch 26: packed variable-length prefill MQA whose only
    allocation is the fp32 logits; same output as ``torch_qsa_mqa_prefill``.

    One program per (BLOCK_M rows, BLOCK_N keys). Tiles with no key inside
    any row's ``[start, end)`` store ``-inf`` without touching q or k; live
    tiles load their keys once and score the heads in fp32 in a fixed order.
    No atomics: bit-identical across launches.
    """
    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    rows, heads, head_dim = q.shape
    keys = k.shape[0]
    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    if rows == 0 or keys == 0:
        return logits
    if head_dim & (head_dim - 1):
        raise ValueError(f"Triton QSA prefill MQA needs a power-of-two head_dim, got {head_dim}")
    if q.stride(-1) != 1:
        q = q.contiguous()
    k2 = k[:, 0]
    if k2.stride(-1) != 1:
        k2 = k2.contiguous()
    row_starts = row_starts.to(device=q.device, dtype=torch.int32)
    row_ends = row_ends.to(device=q.device, dtype=torch.int32)
    block_m, block_n = 64, 64
    grid = (triton.cdiv(rows, block_m), triton.cdiv(keys, block_n))
    _qsa_mqa_prefill_kernel[grid](
        q,
        k2,
        row_starts,
        row_ends,
        logits,
        rows,
        keys,
        float(score_scale or math.sqrt(head_dim)),
        q.stride(0),
        q.stride(1),
        k2.stride(0),
        logits.stride(0),
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return logits
'''
text = text.replace(anchor, kernel + anchor, 1)

# ---- dispatch ---------------------------------------------------------------
old = """    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
"""
assert text.count(old) == 1, "qsa/mqa.py: qsa_mqa_prefill dispatch anchor not found"
new = """    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    if q.is_cuda and qsa_mqa_triton_allowed():
        # gfx1151 patch 26: the torch reference materialises the per-head
        # scores (4x the logits) plus three more copies per prefill chunk.
        return triton_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
"""
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 26 (qsa mqa prefill triton) applied")
