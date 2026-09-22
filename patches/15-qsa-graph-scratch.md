# Patch 15: a dedicated QSA packed-KV scratch for captured decode graphs

**File:** `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_graph_scratch.py`](patch_qsa_graph_scratch.py)
**Upstream:** not gfx1151-specific; any Qwen3.8 (QSA) deployment with
`--cuda-graph-max-bs-decode` below `--max-running-requests` is exposed.

## Symptom

With decode graphs for bs ≤ 8 and 12–16 concurrent streams, the scheduler
died with

```
Memory access fault by GPU node-1 ... on address 0x7f.... Reason: Page not present or supervisor privilege.
```

reproducibly within two rounds of a benchmark loop, with and without the
tuned MoE tiles and with and without `expandable_segments`. Pure eager
(`--disable-cuda-graph`) and pure graph (graphs for every batch size the
scheduler can form) runs were clean, and under `AMD_SERIALIZE_KERNEL=3` the
fault surfaced at the first kernel after a graph replay.

## Two contributing factors

Neither alone crashes. The reproduction needed both:

1. **Eager decode above the graph range frees a buffer the graphs use.**
   `QwenSparseAttnBackend._get_fa2_scratch` hands out one growable pair of
   packed K/V buffers per `(num_kv_heads, head_dim, dtype, device)`:

   ```python
   buffers = self._fa2_scratch.get(key)
   if buffers is None or buffers[0].shape[0] < capacity:
       buffers = (torch.empty(shape, ...), torch.empty(shape, ...))
       self._fa2_scratch[key] = buffers
   ```

   Graph capture sizes it for `cuda_graph_max_tokens × topk` rows and the
   captured `qwen_sparse_kv_extraction_compact_triton` / attention kernels bake
   in those addresses. An eager decode step for a batch larger than any
   captured graph asks for `batch × topk` rows, so the method allocates a
   bigger pair and drops the old one. Prefill never touches this scratch
   (`forward_extend` has its own varlen path), which is why pure-graph runs
   were clean.

2. **`torch.cuda.empty_cache()` unmaps the freed block.** PyTorch's caching
   allocator is stream-affine and nothing else allocates on the capture
   stream, so after (1) the graphs write into a freed-but-still-mapped block
   and nothing notices. `/flush_cache` (which the benchmark called between
   concurrency rounds) calls `empty_cache()`, the block's pages are released,
   and the next replay faults. Without the flush, three rounds of
   16-stream → 8-stream ran clean and a greedy probe returned byte-identical
   text before and after.

Found with `torch.cuda.memory._record_memory_history()` plus an allocator
snapshot taken inside `flush_cache` right before `empty_cache()`: the faulting
address fell in a 16 MiB block that was `active_allocated` from startup
through the 8-stream graph round (allocation stack:
`_get_fa2_scratch ← _forward_paged_attention ← forward_decode ← … ←
decode_cuda_graph_runner.run_once ← capture_one`) and `inactive` after the
first 16-stream eager round.

## Fix

Key the scratch on `metadata.is_cuda_graph` too. Captures always request the
same capacity (`cuda_graph_max_tokens × topk`, or
`cuda_graph_max_tokens × stride` on the TRT-LLM paged path), so the graph pair
is allocated once and never replaced; a growth request against it raises
instead of silently re-allocating. Eager decode grows its own pair freely.
Both call sites pass the flag. Memory cost: one extra pair for eager decode
(2 × `max_running_requests × topk × kv_heads × head_dim × 2 B`; 2 × 32 MiB
at 16 requests here).

## Verification

Debug container, graphs for bs ≤ 8, `--max-running-requests 16`, real
`empty_cache()` between rounds:

| | before | after |
|---|---|---|
| 16 streams → flush → 8 streams | fault on the first replay, every time | 3 rounds clean |
| greedy bs=1 probe after each round | n/a (dead) | identical |
| graph scratch pair in the allocator snapshot after an eager round | `inactive` (freed) | `active_allocated`, separate 2 × 32 MiB eager pair |

The launchers no longer need `--max-running-requests` tied to
`--cuda-graph-max-bs-decode`; the defaults keep them equal (20/20) because the
extra graphs cost 0.3 GB and nothing else.
