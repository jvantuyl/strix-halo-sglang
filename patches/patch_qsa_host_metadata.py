#!/usr/bin/env python3
"""gfx1151 patch 39: build QSA speculative metadata from host-known lengths.

py-spy on the production scheduler (2026-11-20, mission decode+prefill) put
75% of the scheduler thread's busy time inside one chain:
run_batch -> _draft_extend_for_decode -> prepare_for_draft_extend ->
init_forward_metadata -> _metadata_from_forward_batch -> _speculative_row_to_request.

That chain runs on the plan stream once per MTP round (and once per prefill
chunk's draft-extend). Its extend branch called
``int(repeats.sum().item())`` on a device tensor — a full-stream sync in the
middle of batch preparation. Under the overlap scheduler the thread does this
while forwards are still in flight, so the sync waits them out: ~1 s per
1024-token prefill chunk (the mission's prefill cadence is thread-bound, not
GPU-bound) and a share of every decode round. The amdgpu_top "oscillation"
and the scheduler thread being ~95% busy while 31 cores idle both trace back
here.

The host already knows the extend lengths — the overlap FutureMap publishes
``extend_seq_lens_cpu`` (and ``req_pool_indices_cpu``) with the batch, the
same routing patch 37 gave seq_lens_cpu. So:

1. ``_speculative_row_to_request``: when ``extend_seq_lens_cpu`` is present,
   build the row->request mapping entirely on the host (torch CPU ops, one
   H2D move at the end). The device-tensor fallback path is kept for callers
   without the CPU list.
2. The chunk-prefill forward path's ``req_pool_indices.tolist()``: prefer the
   published ``req_pool_indices_cpu`` (a CPU-tensor ``.tolist()`` does not
   touch the compute stream).

No numerics change: the mapping values are identical, only where they are
computed.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            # DP token padding rows belong to no request; alias them to request row 0,
            # as the CUDA-graph layout does, so gathers stay in bounds.
            repeats = extend_seq_lens[:batch_size].to(dtype=torch.long)
            real_rows = int(repeats.sum().item())
            if real_rows > num_rows:
                raise ValueError(
                    "QSA speculative query rows are fewer than the extend "
                    f"mapping: rows={num_rows}, mapped={real_rows}"
                )
            row_to_request = torch.repeat_interleave(
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                ),
                repeats,
            )
            padding = num_rows - real_rows
            if padding:
                row_to_request = torch.cat(
                    [
                        row_to_request,
                        row_to_request.new_zeros(padding),
                    ]
                )
            return row_to_request"""
new = """        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            # gfx1151 patch 39: the overlap scheduler already publishes the
            # extend lengths on the host. Build the mapping there — a
            # .item() on the device tensor would sync this (plan-)stream
            # against whatever forward is still in flight, ~1 s per
            # 1024-token prefill chunk and part of every decode round.
            repeats_cpu = torch.tensor(
                list(extend_seq_lens_cpu[:batch_size]), dtype=torch.long
            )
            real_rows = int(repeats_cpu.sum())
            if real_rows > num_rows:
                raise ValueError(
                    "QSA speculative query rows are fewer than the extend "
                    f"mapping: rows={num_rows}, mapped={real_rows}"
                )
            row_to_request = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.long),
                repeats_cpu,
            )
            padding = num_rows - real_rows
            if padding:
                row_to_request = torch.cat(
                    [
                        row_to_request,
                        torch.zeros(padding, dtype=torch.long),
                    ]
                )
            return row_to_request.to(forward_batch.req_pool_indices.device)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            # DP token padding rows belong to no request; alias them to request row 0,
            # as the CUDA-graph layout does, so gathers stay in bounds.
            repeats = extend_seq_lens[:batch_size].to(dtype=torch.long)
            real_rows = int(repeats.sum().item())
            if real_rows > num_rows:
                raise ValueError(
                    "QSA speculative query rows are fewer than the extend "
                    f"mapping: rows={num_rows}, mapped={real_rows}"
                )
            row_to_request = torch.repeat_interleave(
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                ),
                repeats,
            )
            padding = num_rows - real_rows
            if padding:
                row_to_request = torch.cat(
                    [
                        row_to_request,
                        row_to_request.new_zeros(padding),
                    ]
                )
            return row_to_request"""
assert text.count(old) == 1, "row_to_request extend anchor not found (once expected)"
text = text.replace(old, new)

old = """        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = ["""
new = """        req_indices_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
        if req_indices_cpu is not None:
            # gfx1151 patch 39: a CPU-tensor .tolist() does not sync the
            # compute stream; the device readback did, once per QSA layer
            # per prefill chunk.
            req_indices = (
                req_indices_cpu.tolist()
                if torch.is_tensor(req_indices_cpu)
                else list(req_indices_cpu)
            )
        else:
            req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = ["""
assert text.count(old) == 1, "req_indices anchor not found (once expected)"
text = text.replace(old, new)

open(p, "w").write(text)

print("patch_qsa_host_metadata: ok")
