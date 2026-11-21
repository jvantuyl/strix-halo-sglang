#!/usr/bin/env python3
"""gfx1151 patch 41: QSA prefill MQA inputs from host mirrors, no readbacks.

Third in the patch 37/39/40 series (scheduler thread de-serialization). After
40 removed the rope-cache readback, a py-spy capture during a cold 64k-token
prefill showed the next dominator at 82% of thread samples:
`QSAIndexerMetadata.get_prefill_mqa_inputs`, whose

    sequence_lengths_list = sequence_lengths.tolist()

is a device->host sync per QSA layer per prefill chunk — and the chunk
prefill kernel wrapper `sparse_gqa_fwd_interface_triton_ck` computes its
launch-grid bound as `int((cu_q[1:] - cu_q[:-1]).max().item())`, another
readback per layer per chunk. Sequential syncs in the same per-layer chain
mask each other, which is why 39 and 40 each moved the wall only slightly.

Both values are host-known on the serving path:

* `QSAIndexerMetadata.sequence_lengths` for a non-speculative forward is
  exactly `forward_batch.seq_lens` — whose overlap-published mirror is
  `forward_batch.seq_lens_cpu`;
* the chunk kernel's `max_q` is `max(extend_seq_lens)`, and the caller
  already builds the host list `extend_seq_lens_cpu`.

Fixes:

1. `QSAIndexerMetadata` (msgspec struct in qsa/metadata.py) gains
   `sequence_lengths_cpu` (default None; existing construction sites are
   unaffected). The non-speculative branch of `_metadata_from_forward_batch`
   passes it from `forward_batch.seq_lens_cpu`; speculative-paged and
   graph-capture constructions deliberately leave it None (paged rows are
   per-token, not per-request, and `get_prefill_mqa_inputs` is a
   non-speculative path — a wrong mirror would be silently wrong, so it is
   never set where it is not exact).
2. `get_prefill_mqa_inputs` uses the mirror when present (tensor or list),
   falling back to the original `.tolist()` readback.
3. `sparse_gqa_fwd_interface_triton_ck` accepts `max_q=None` and skips the
   device readback when the caller supplies it; the QSA backend's
   chunk-prefill caller passes `max(extend_seq_lens_cpu)`.

No numerics change: identical values, host-sourced.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# --- 1. Struct field + host-mirror read (qsa/metadata.py) ----------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/metadata.py"
text = open(p).read()

old = """        parts = []
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        sequence_lengths_list = sequence_lengths.tolist()"""
new = """        parts = []
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        # gfx1151 patch 41: the overlap scheduler publishes seq_lens_cpu and
        # this metadata carries it for non-speculative forwards; a device
        # .tolist() here synced the compute stream per layer per chunk.
        if self.sequence_lengths_cpu is not None:
            mirror = self.sequence_lengths_cpu
            sequence_lengths_list = (
                [int(x) for x in mirror.tolist()]
                if isinstance(mirror, torch.Tensor)
                else [int(x) for x in mirror]
            )
        else:
            sequence_lengths_list = sequence_lengths.tolist()"""
assert text.count(old) == 1, "get_prefill_mqa_inputs anchor not found"
text = text.replace(old, new)

old = """    extend_rope_matrix: Optional[torch.Tensor] = None
    graph_ring_group_locs: Optional[torch.Tensor] = None

    def get_seqlens_int32(self) -> torch.Tensor:"""
new = """    extend_rope_matrix: Optional[torch.Tensor] = None
    graph_ring_group_locs: Optional[torch.Tensor] = None
    # gfx1151 patch 41: host mirror of sequence_lengths for non-speculative
    # forwards (forward_batch.seq_lens_cpu); None keeps the .tolist()
    # fallback. Never set on speculative-paged metadata, whose
    # sequence_lengths rows are per-token rather than per-request.
    sequence_lengths_cpu: Optional[object] = None

    def get_seqlens_int32(self) -> torch.Tensor:"""
assert text.count(old) == 1, "QSAIndexerMetadata field-tail anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 2. Pass the mirror from the non-spec construction (backend) ----------
p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
            req_pool_indices=row_req_pool_indices,"""
new = """            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
            req_pool_indices=row_req_pool_indices,
            # gfx1151 patch 41: exact host mirror on the non-speculative
            # branch (per-request rows); speculative-paged rows are
            # per-token, so the mirror stays None there by design.
            sequence_lengths_cpu=getattr(
                forward_batch, "seq_lens_cpu", None
            ),"""
assert text.count(old) == 1, "non-spec QSAIndexerMetadata construction anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 3. Chunk-prefill kernel grid bound from the host ---------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/sparse_attn.py"
text = open(p).read()

old = """def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())"""
new = """def sparse_gqa_fwd_interface_triton_ck(
    q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q=None
):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    # gfx1151 patch 41: callers with host-known extend lengths pass max_q;
    # the device readback is the fallback for callers that do not.
    if max_q is None:
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())"""
assert text.count(old) == 1, "sparse_attn signature anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 4. Caller passes the host max ---------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            layer.scaling,
        )"""
new = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            layer.scaling,
            # gfx1151 patch 41: host-known grid bound; avoids a device
            # readback of the cumsum per layer per chunk.
            max_q=max(extend_lens),
        )"""
assert text.count(old) == 1, "backend triton_ck call anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

print("patch_qsa_prefill_host_mirrors: ok")
