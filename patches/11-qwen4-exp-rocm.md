# Patch 11 — Qwen4-Exp (Qwen3.8-Flash-Next) on ROCm/gfx1151

Upstream's Qwen4-Exp support (PR #37500 + PLE offload #37068) assumes an
NVIDIA datacenter/AIO part in three places. Applied by
`patches/patch_qwen4_exp_rocm.py` with asserted anchors.

## 1. CPU-side PLE gather (`models/qwen4_exp.py`)

The `pinned` and `file` PLE offload backends gather with a Triton kernel that
dereferences the **host pointer** of the table from the device (UVA). On Strix
Halo the GPU must not touch host memory in the hot path: the IOMMU is on for
the NPU (ASR), which turns device→host accesses into a 10–60 GB/s path, and
XNACK-style fault handling is off. The first gather would fault or crawl.

The patch gathers on the CPU instead: D2H the ids, `index_select` on the
pinned/mmap'd table, copy bf16 rows back. Numerically identical to the device
kernel (raw bf16 out, no `weight_scale` — the consumer applies it). The file
backend's prefetcher (WILLNEED/madvise) still runs, so page-cache readahead still
hides most of the fault latency.

Escape hatch: `SGLANG_PLE_UVA_GATHER=1` restores the upstream device gather.

## 2. QSA decode via the pure-Triton kernel (`qwen_sparse_attn_backend.py`)

Upstream routes QSA decode to `qwen38_qsa_sm121_varlen` only on NVIDIA SM121;
every other device lands on `flash_attn_varlen_func` (CUDA binary). On HIP the
patch binds the same pure-Triton kernel (`kda_kernels/qwen38_qsa_sm121`,
specialized for Qwen3.8's shapes: head_dim 256, 12:1/24:2 heads, batch ≤ 128,
selected KV ≤ 2055) directly — the `ops.attention` wrapper is behind a hard
`CapabilityRequirement.cuda(min_sm=(12,1))` registry gate — and enforces the
shape contract itself, falling back to flash_attn for out-of-contract calls.
The first call of each kind logs which path it took
(`QSA decode on HIP: using Triton qwen38_qsa kernel (...)` or
`... using flash_attn varlen fallback (...)`), so a real run can be verified
from the server log.

Escape hatch: `SGLANG_QSA_DECODE_FLASH_ATTN=1` restores the flash_attn route.

## 3. `qsa_fast_topk` fallback chain (`qsa/kernel.py`)

The top-k 512 path calls the JIT CUDA kernel with no fallback. On gfx1151
that kernel is **unsafe**: it reads out of bounds and then either raises
IndexError or silently returns wrong rows, and the bad reads poison the HIP
queue so a later unrelated kernel dies with `HSA_STATUS_ERROR_EXCEPTION`
(measured). `sgl_kernel.top_k.fast_topk_v2` asserts topk=2048 only. So on HIP
the patch skips the JIT entirely and uses the fixed-width reference path
(rows ≤ 128, one small `torch.topk` per row).
`SGLANG_QSA_TOPK_JIT=1` re-enables the JIT for experiments.

## 4. QSA decode shared-memory budget (`kda_kernels/qwen38_qsa_sm121`)

The GB10 launch schedule uses BLOCK_KV=64 with 2 stages for larger batches,
which needs 72 KB of workgroup shared memory. RDNA3.5 caps workgroups at
64 KB, so those launches fail with `OutOfResources` on gfx1151. On HIP the
kernel always takes the BLOCK_KV=32 / 3-stage schedule (measured to fit and
match the reference on hardware).
