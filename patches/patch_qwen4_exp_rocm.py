#!/usr/bin/env python3
"""gfx1151 fixes for Qwen4-Exp (Qwen3.8-Flash-Next) on ROCm (patch 11).

Three edits, each asserted so an upstream move fails the build loudly:

1. models/qwen4_exp.py -- PLE embedding gather. Upstream's pinned/file backends
   gather with a Triton kernel that dereferences the HOST pointer of the table
   from the device (UVA). That requires the GPU to read host memory through the
   IOMMU's page tables with XNACK-style fault handling -- exactly what Strix
   Halo cannot do here (IOMMU is on for the NPU and turns host accesses into a
   10-60 GB/s path; the GPU must not touch host memory in the hot path at
   all). Gather on the CPU instead: D2H the ids, index_select on the
   pinned/mmap'd host table, copy bf16 rows back. Like the device kernel, this
   emits raw bf16 WITHOUT weight_scale (the consumer multiplies it; see the
   Qwen4ExpPLELayer forwards). SGLANG_PLE_UVA_GATHER=1 restores upstream.

2. layers/attention/qwen_sparse_attn_backend.py -- QSA decode. Upstream only
   routes decode to the pure-Triton qwen38_qsa_sm121 kernel on NVIDIA SM121;
   everything else lands on flash_attn_varlen_func (a CUDA binary). On HIP,
   bind the same Triton kernel directly: the ops.attention wrapper is behind a
   hard CapabilityRequirement.cuda(min_sm=(12,1)) registry gate, so we call the
   kda kernel module and enforce its shape contract ourselves, falling back to
   flash_attn for calls outside the contract. SGLANG_QSA_DECODE_FLASH_ATTN=1
   restores the flash_attn route.

3. layers/attention/qsa/kernel.py -- qsa_fast_topk. The top-k 512 path calls
   the JIT CUDA kernel with no fallback; if the JIT toolchain cannot build for
   this HIP target the server dies on the first decode. Catch the failure and
   fall through to sgl_kernel.fast_topk_v2, then to the reference path.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# ---------------------------------------------------------------------------
# 1. models/qwen4_exp.py -- CPU-side PLE gather
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/models/qwen4_exp.py"
text = open(p).read()

old = "import math\nfrom contextlib import nullcontext"
assert text.count(old) == 1, "qwen4_exp.py: import anchor not found"
text = text.replace(old, "import math\nimport os\nfrom contextlib import nullcontext", 1)

old = "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):"
assert text.count(old) == 1, "qwen4_exp.py: class anchor not found"
helper = '''# gfx1151 patch 11: the pinned/file PLE gather kernel dereferences the HOST
# pointer of the table from the device (UVA). On Strix Halo the GPU must not
# touch host memory in the hot path (IOMMU on for the NPU turns such accesses
# into a 10-60 GB/s path; XNACK off), so gather on the CPU instead: copy the
# ids to the host, index_select on the pinned/mmap'd table, copy bf16 rows
# back. Like the device kernel, this emits raw bf16 WITHOUT weight_scale --
# the consumer multiplies it (see the PLE layer forwards).
# SGLANG_PLE_UVA_GATHER=1 restores the upstream device-side gather.
_STRIX_PLE_CPU_GATHER = (
    torch.version.hip is not None
    and os.environ.get("SGLANG_PLE_UVA_GATHER", "0") != "1"
)


def _gather_ple_embedding_on_cpu(
    embedding: "Qwen4ExpPinnedHostEmbedding",
    flat_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    start = embedding.shard_indices.org_vocab_start_index
    end = embedding.shard_indices.org_vocab_end_index
    host_ids = flat_ids.detach().to("cpu")
    in_range = (host_ids >= start) & (host_ids < end)
    local_idx = torch.where(in_range, host_ids - start, torch.zeros_like(host_ids))
    rows = embedding.weight.data.index_select(0, local_idx).to(torch.bfloat16)
    rows = torch.where(in_range.unsqueeze(1), rows, torch.zeros_like(rows))
    output.view(-1, embedding.embedding_dim).copy_(rows)


'''
text = text.replace(old, helper + old, 1)

old = """            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self.weight.data_ptr(),
                flat_ids,
                output,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
            )
"""
assert text.count(old) == 1, "qwen4_exp.py: gather launch anchor not found"
new = """            if _STRIX_PLE_CPU_GATHER:
                _gather_ple_embedding_on_cpu(self, flat_ids, output)
            else:
                _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                    self.weight.data_ptr(),
                    flat_ids,
                    output,
                    embedding_dim=self.embedding_dim,
                    tp_vocab_start=self.shard_indices.org_vocab_start_index,
                    tp_vocab_end=self.shard_indices.org_vocab_end_index,
                    is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                    BLOCK_D=self._block_d,
                )
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# 2. qwen_sparse_attn_backend.py -- QSA decode on HIP
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = "import logging\nimport math\nfrom copy import copy"
assert text.count(old) == 1, "qsa backend: import anchor not found"
text = text.replace(old, "import logging\nimport math\nimport os\nfrom copy import copy", 1)

old = """@lru_cache(maxsize=1)
def _resolve_flash_attn_varlen_func():
"""
assert text.count(old) == 1, "qsa backend: resolver anchor not found"
fallback_fn = '''def _resolve_flash_attn_varlen_fallback():
    """Resolve a plain flash_attn varlen decode call (FA2, then FA4 cute)."""
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import (
            flash_attn_varlen_func as cute_varlen_func,
        )

        def flash_attn_varlen_func(*args, **kwargs):
            output = cute_varlen_func(*args, **kwargs)
            # The cute interface returns (out, lse); lse is None here.
            return output[0] if isinstance(output, tuple) else output

        return flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "QSA decode requires flash_attn (FA2) or flash-attn-4 "
            "(FA4 cute) for its packed varlen fallback."
        ) from exc


@lru_cache(maxsize=1)
def _resolve_flash_attn_varlen_func():
'''
text = text.replace(old, fallback_fn, 1)

old = """    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
"""
assert text.count(old) == 1, "qsa backend: sm121 route anchor not found"
new = """    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
    if torch.version.hip is not None and os.environ.get(
        "SGLANG_QSA_DECODE_FLASH_ATTN", "0"
    ) != "1":
        # gfx1151 patch 11: route QSA decode to the pure-Triton kernel the
        # SM121 path uses. The ops.attention wrapper refuses non-(12,1)
        # devices (hard CapabilityRequirement in the kernel registry), so bind
        # the kda Triton kernel directly and enforce its shape contract here,
        # falling back to flash_attn for calls outside the contract.
        from sglang.kernels.kda_kernels.qwen38_qsa_sm121.kernel import (
            qwen38_qsa_sm121 as _kda_qsa_run,
        )

        _logged_paths = set()

        def _log_path_once(path_name, detail):
            if path_name not in _logged_paths:
                _logged_paths.add(path_name)
                logging.getLogger(__name__).info(
                    "QSA decode on HIP: using %s (%s)", path_name, detail
                )

        def _qwen38_qsa_hip_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=0,
            softmax_scale=1.0,
            causal=True,
            **_,
        ):
            _, num_q_heads, head_dim = q.shape
            num_kv_heads = k.shape[1]
            contract_ok = (
                q.is_cuda
                and q.dtype == torch.bfloat16
                and head_dim == 256
                and (num_q_heads, num_kv_heads) in ((12, 1), (24, 2))
                and k.shape[2] == head_dim
                and v.shape == k.shape
                and max_seqlen_q == 1
                and 0 < max_seqlen_k <= 2055
                and q.is_contiguous()
                and k.is_contiguous()
                and v.is_contiguous()
                and cu_seqlens_q.dtype == torch.int32
                and cu_seqlens_k.dtype == torch.int32
            )
            if contract_ok:
                _log_path_once(
                    "Triton qwen38_qsa kernel",
                    f"heads=({num_q_heads},{num_kv_heads}) head_dim={head_dim}",
                )
                return _kda_qsa_run(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale
                )
            _log_path_once(
                "flash_attn varlen fallback",
                f"heads=({num_q_heads},{num_kv_heads}) head_dim={head_dim} "
                f"dtype={q.dtype} max_seqlen_q={max_seqlen_q} "
                f"max_seqlen_k={max_seqlen_k}",
            )
            return _resolve_flash_attn_varlen_fallback()(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return _qwen38_qsa_hip_varlen
"""
text = text.replace(old, new, 1)

old = """    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func as cute_varlen_func

        def flash_attn_varlen_func(*args, **kwargs):
            output = cute_varlen_func(*args, **kwargs)
            # The cute interface returns (out, lse); lse is None here.
            return output[0] if isinstance(output, tuple) else output

        return flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "QSA decode requires flash_attn (FA2) or flash-attn-4 "
            "(FA4 cute) for its packed varlen fallback."
        ) from exc
"""
assert text.count(old) == 1, "qsa backend: flash_attn chain anchor not found"
text = text.replace(old, "    return _resolve_flash_attn_varlen_fallback()\n", 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# 3. qsa/kernel.py -- fast_topk fallback chain
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/kernel.py"
text = open(p).read()

old = "from __future__ import annotations\n\nfrom typing import Optional"
assert text.count(old) == 1, "qsa kernel: import anchor not found"
text = text.replace(
    old,
    "from __future__ import annotations\n\nimport logging\nimport os\nfrom typing import Optional",
    1,
)

old = "def qsa_fast_topk("
assert text.count(old) == 1, "qsa kernel: def anchor not found"
helper = '''def _qsa_fixed_width_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Reference top-k with fixed-width output: short rows pad with -1."""
    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


'''
text = text.replace(old, helper + old, 1)

old = """        if topk == 512:
            # Prefer the JIT kernel: it ships with the sglang python package,
            # so top-k 512 works regardless of the installed sgl_kernel version.
            from sglang.kernels.ops.elementwise.fast_topk import fast_topk

            return fast_topk(logits, lengths, topk=512, row_starts=starts)
"""
assert text.count(old) == 1, "qsa kernel: jit topk anchor not found"
new = """        if topk == 512:
            # Prefer the JIT kernel: it ships with the sglang python package,
            # so top-k 512 works regardless of the installed sgl_kernel version.
            # gfx1151 patch 11: on RDNA3.5 the JIT kernel is UNSAFE -- it can
            # read out of bounds and then either raise IndexError or silently
            # return wrong rows, and its bad reads poison the HIP queue so a
            # later unrelated kernel dies with HSA_STATUS_ERROR_EXCEPTION
            # (measured on gfx1151). sgl_kernel.fast_topk_v2 asserts
            # topk==2048. So on HIP skip straight to the reference path;
            # SGLANG_QSA_TOPK_JIT=1 re-enables the JIT for experiments.
            if not (
                torch.version.hip is not None
                and os.environ.get("SGLANG_QSA_TOPK_JIT", "0") != "1"
            ):
                try:
                    from sglang.kernels.ops.elementwise.fast_topk import fast_topk

                    return fast_topk(logits, lengths, topk=512, row_starts=starts)
                except Exception as exc:  # JIT compile can raise many error types
                    logging.getLogger(__name__).warning(
                        "qsa_fast_topk: JIT fast_topk unavailable (%s); falling "
                        "back to sgl_kernel/reference",
                        exc,
                    )
"""
text = text.replace(old, new, 1)

old = """        raise ValueError(
            f"QSA top-k {topk} is unsupported by sgl_kernel; "
            f"supported values are {supported_topk}"
        )
"""
assert text.count(old) == 1, "qsa kernel: unsupported-topk anchor not found"
text = text.replace(
    old, "        return _qsa_fixed_width_topk(logits, lengths, starts, topk)\n", 1
)

old = """    # CPU/reference path mirrors the CUDA operator's fixed-width, relative output.
    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output
"""
assert text.count(old) == 1, "qsa kernel: reference block anchor not found"
text = text.replace(
    old,
    """    # CPU/reference path mirrors the CUDA operator's fixed-width, relative output.
    return _qsa_fixed_width_topk(logits, lengths, starts, topk)
""",
    1,
)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# 4. kda qwen38_qsa_sm121 kernel -- shared-memory budget on RDNA3.5
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py"
text = open(p).read()

old = """    use_bk32 = (num_kv_heads == 1 and batch < 12) or (num_kv_heads == 2 and batch < 4)
    block_kv = 32 if use_bk32 else 64
    stages = 3 if use_bk32 else 2
"""
assert text.count(old) == 1, "kda qsa kernel: schedule anchor not found"
new = """    use_bk32 = (num_kv_heads == 1 and batch < 12) or (num_kv_heads == 2 and batch < 4)
    block_kv = 32 if use_bk32 else 64
    stages = 3 if use_bk32 else 2
    # gfx1151 patch 11: RDNA3.5 caps workgroup shared memory at 64KB and the
    # BKV=64/2-stage GB10 schedule needs 72KB (OutOfResources on gfx1151).
    # The BKV=32/3-stage schedule fits and is what small batches already use.
    if torch.version.hip is not None:
        block_kv = 32
        stages = 3
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

print("patch 11 (qwen4_exp rocm) applied")
