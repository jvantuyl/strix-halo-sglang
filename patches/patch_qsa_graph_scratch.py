#!/usr/bin/env python3
"""Patch 15: a dedicated QSA packed-KV scratch for captured decode graphs.

``QwenSparseAttnBackend._get_fa2_scratch`` hands out one growable pair of
packed K/V buffers per (heads, head_dim, dtype, device). The decode graphs are
captured with the scratch sized for ``cuda_graph_max_tokens * topk`` rows and
their kernels bake in those addresses. An eager decode step for a batch larger
than any captured graph asks for more rows, so the method allocates a bigger
pair and drops the old one; from then on every replay writes into freed
memory. Nothing else allocates on the capture stream, so the write stays
silent until ``torch.cuda.empty_cache()`` (``/flush_cache``) unmaps the block
and the next replay dies with ``Memory access fault ... Page not present``.

Key the scratch on ``metadata.is_cuda_graph`` as well. Graph captures always
ask for the same capacity, so their pair is allocated once and never
replaced; eager decode grows its own pair freely.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """        self._fa2_scratch: Dict[
            Tuple[int, int, torch.dtype, torch.device],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
"""
assert text.count(old) == 1, "qwen_sparse_attn_backend.py: _fa2_scratch declaration anchor not found"
new = """        self._fa2_scratch: Dict[
            Tuple[int, int, torch.dtype, torch.device, bool],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
"""
text = text.replace(old, new, 1)

old = """    def _get_fa2_scratch(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (num_kv_heads, head_dim, dtype, device)
        buffers = self._fa2_scratch.get(key)
        if buffers is None or buffers[0].shape[0] < capacity:
            shape = (capacity, num_kv_heads, head_dim)
"""
assert text.count(old) == 1, "qwen_sparse_attn_backend.py: _get_fa2_scratch anchor not found"
new = """    def _get_fa2_scratch(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        is_cuda_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Captured graphs bake in the scratch addresses, so they get their own
        # pair: an eager decode step larger than any graph must not replace it.
        key = (num_kv_heads, head_dim, dtype, device, is_cuda_graph)
        buffers = self._fa2_scratch.get(key)
        if buffers is None or buffers[0].shape[0] < capacity:
            if is_cuda_graph and buffers is not None:
                raise RuntimeError(
                    "QSA CUDA graph scratch cannot grow after capture: "
                    f"have {buffers[0].shape[0]} rows, need {capacity}"
                )
            shape = (capacity, num_kv_heads, head_dim)
"""
text = text.replace(old, new, 1)

# Call site 1: FlashInfer/TRT-LLM paged decode packing.
old = """        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
"""
assert text.count(old) == 1, "qwen_sparse_attn_backend.py: trtllm scratch call anchor not found"
new = """        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
            is_cuda_graph=metadata.is_cuda_graph,
        )
"""
text = text.replace(old, new, 1)

# Call site 2: FA2 varlen decode packing.
old = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
"""
assert text.count(old) == 1, "qwen_sparse_attn_backend.py: fa2 scratch call anchor not found"
new = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
            is_cuda_graph=metadata.is_cuda_graph,
        )
"""
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 15 (dedicated QSA graph scratch) applied")
