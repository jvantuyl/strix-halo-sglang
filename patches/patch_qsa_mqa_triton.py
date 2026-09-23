#!/usr/bin/env python3
"""gfx1151 patch 24: length-bounded Triton decode MQA for the QSA indexer.

`qsa_mqa_decode` scores every compressed block of a request against the
query for the indexer's block selection. Upstream has a TileLang kernel for
it and a torch reference; TileLang is not installed on ROCm (nor on every
CUDA build), so the reference ran on every decode step of all 12 QSA layers.
The reference gathers the *whole* page-table window, `context_len / ratio`
blocks × head_dim, converts it to fp32 and runs an einsum over it, then masks
past the real length: work proportional to the model's context length, not
the request's. At 131k context that is 32,768 blocks per row; measured per
layer per step on this box, 256-token sequences: 1.8 ms at bs 20 with a 32k
context, 6.0 ms with 131k. Times 12 layers, that was the whole drop from
85–97 to 68–72 tok/s at 16–20 streams when the default context moved from
32k to 131k (bs 1 barely moves: under 1 ms per step either way).

The Triton kernel here does one program per (row, 64-block tile): tiles past
the row's `context_len` store `-inf` and exit after one scalar load; live
tiles gather their pages, compute `relu(q·k)` per head in fp32 with a fixed
reduction order, sum the heads, scale, and store scores or `-inf`. Same
contract as the reference (fp32 `[batch, max_model_len]`, `-inf` at or past
`context_len` and past the page table), one launch instead of fill + gather
+ einsum + mask + copy, no atomics (run-to-run bit-identical), static
pointers only (graph-capturable). It is dispatched whenever TileLang is
unavailable on a CUDA/HIP device; `SGLANG_QSA_MQA_TRITON=0` forces the
reference for A/B. TileLang, when present, still wins the dispatch.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qsa/mqa.py"
text = open(p).read()

# ---- imports ---------------------------------------------------------------
old = """import math
from typing import Optional

import torch

try:
    import flashinfer.comm  # noqa: F401
"""
assert text.count(old) == 1, "qsa/mqa.py: import anchor not found"
new = """import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    import flashinfer.comm  # noqa: F401
"""
text = text.replace(old, new, 1)

# ---- kernel + wrapper, inserted before the prefill dispatcher ---------------
anchor = "\n\ndef qsa_mqa_prefill(\n"
assert text.count(anchor) == 1, "qsa/mqa.py: qsa_mqa_prefill anchor not found"
kernel = '''

def qsa_mqa_triton_allowed() -> bool:
    """gfx1151 patch 24: the Triton decode MQA stands in for TileLang unless
    SGLANG_QSA_MQA_TRITON=0 asks for the torch reference."""
    return os.environ.get("SGLANG_QSA_MQA_TRITON", "1") != "0"


@triton.jit
def _qsa_mqa_decode_kernel(
    Q,
    KCache,
    PageTable,
    ContextLens,
    Logits,
    total_blocks,
    max_model_len,
    scale,
    stride_qb,
    stride_qh,
    stride_kp,
    stride_ks,
    stride_pt,
    stride_lb,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    start = tile * BLOCK_N
    n = start + tl.arange(0, BLOCK_N)
    in_width = n < max_model_len
    out_ptr = Logits + row * stride_lb + n
    ctx = tl.load(ContextLens + row)
    live = tl.minimum(ctx, total_blocks)
    if start < live:
        valid = in_width & (n < live)
        pages = tl.load(
            PageTable + row * stride_pt + n // PAGE_SIZE, mask=valid, other=0
        ).to(tl.int64)
        slots = n % PAGE_SIZE
        d = tl.arange(0, HEAD_DIM)
        k_ptr = (
            KCache
            + pages[:, None] * stride_kp
            + slots[:, None] * stride_ks
            + d[None, :]
        )
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for h in tl.static_range(HEADS):
            q = tl.load(Q + row * stride_qb + h * stride_qh + d).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1)
            acc += tl.maximum(s, 0.0)
        out = tl.where(valid, acc / scale, float("-inf"))
        tl.store(out_ptr, out, mask=in_width)
    else:
        pad = tl.full([BLOCK_N], float("-inf"), tl.float32)
        tl.store(out_ptr, pad, mask=in_width)


def triton_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """gfx1151 patch 24: paged decode MQA with work bounded by each row's
    context length; same output as ``torch_qsa_mqa_decode``.

    One program per (row, BLOCK_N blocks). Tiles at or past the row's length
    store ``-inf`` after a single scalar load, so a long model context costs
    only the padding stores; live tiles gather their pages once, score the
    heads in fp32 in a fixed order and store. No atomics, static pointers:
    bit-identical across launches and capturable into CUDA graphs.
    """
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch, heads, head_dim = q.shape
    page_size = int(k_cache.shape[1])
    total_blocks = int(page_table.shape[1]) * page_size
    logits = torch.empty(
        (batch, max_model_len), dtype=torch.float32, device=q.device
    )
    if batch == 0 or max_model_len == 0:
        return logits
    if head_dim & (head_dim - 1):
        raise ValueError(f"Triton QSA decode MQA needs a power-of-two head_dim, got {head_dim}")
    if k_cache.stride(-1) != 1:
        k_cache = k_cache.contiguous()
    page_table = page_table.to(device=q.device, dtype=torch.int32)
    context_lens = context_lens.to(device=q.device, dtype=torch.int32)
    block_n = 64
    grid = (batch, triton.cdiv(max_model_len, block_n))
    _qsa_mqa_decode_kernel[grid](
        q,
        k_cache,
        page_table,
        context_lens,
        logits,
        total_blocks,
        max_model_len,
        float(score_scale or math.sqrt(head_dim)),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return logits
'''
text = text.replace(anchor, kernel + anchor, 1)

# ---- dispatch ---------------------------------------------------------------
old = """    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )
"""
assert text.count(old) == 1, "qsa/mqa.py: qsa_mqa_decode dispatch anchor not found"
new = """    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    if q.is_cuda and qsa_mqa_triton_allowed():
        # gfx1151 patch 24: without TileLang the torch reference gathers the
        # whole context window per row on every decode step.
        return triton_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )
"""
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 24 (qsa mqa triton) applied")
