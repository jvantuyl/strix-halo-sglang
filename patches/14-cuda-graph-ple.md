# Patch 14: CUDA graphs with the CPU-side PLE gather

**Files:** `python/sglang/srt/models/qwen4_exp.py`,
`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
**Script:** [`patch_cuda_graph_ple.py`](patch_cuda_graph_ple.py) (requires
patch 11; anchors on its gather branch)

## Problem

Patch 11 gathers PLE n-gram rows on the host (`_gather_ple_embedding_on_cpu`):
ids to the CPU, `index_select` on the mmap'd table, rows back to the GPU.
None of that can be recorded into a CUDA graph, so Qwen3.8-Flash-Next ran
with `--disable-cuda-graph` and paid eager dispatch for 48 layers on every
decode step.

## Fix

Upstream already gives each PLE layer a static per-batch-size prefetch buffer
under capture (`_graph_prefetch_buffers[lookup_tokens]`), filled on a side
stream by `start_prefetch` and read by the graph. This patch keeps the graph
reading that buffer and fills it from the host *before* each replay:

1. `Qwen4ExpPinnedHostEmbedding.gather` skips the host gather while
   `torch.cuda.is_current_stream_capturing()`; the capture-mode buffer is
   zeroed on allocation so the capture pass reads defined memory. The two
   warmup passes before capture still run the real gather.
2. `Qwen4ExpPLELayer.fill_graph_prefetch_buffer(ple_batch, forward_batch)`
   recomputes the n-gram ids for the padded batch (the same GPU ops the graph
   recorded) and runs the host gather into the captured buffer.
3. `Qwen4ExpModel.prepare_graph_replay(forward_batch)` builds the PLE batch
   and fills every PLE layer; `Qwen4ExpForConditionalGeneration` forwards it.
4. `DecodeCudaGraphRunner` keeps the capture-time `ForwardBatch` of every
   batch size (it wraps the static input buffers that `load_batch` refreshes)
   and calls the model's `prepare_graph_replay` with it, between
   `load_batch` and `backend.replay`. Models without the method are
   untouched.

The hook is a no-op with the UVA gather (`SGLANG_PLE_UVA_GATHER=1`) or for
models without a PLE layer, including the MTP draft model
(`Qwen4ExpForCausalLMMTP` clears `ple_layer_ids`). Target-verify graphs go
through the same `_prepare_ple_batch` mode branch that was captured.

## Result

Qwen3.8-Flash-Next-AWQ-INT4, bs=1 decode: 11.2 → 12.7 tok/s; 4/8 concurrent
unchanged (25.9 / 42.9 tok/s aggregate). Graph capture for bs 1–8 costs
0.39 GB. Profiling the scheduler during graph decode shows ~98% of host time
in the D2H copy of the ids, i.e. waiting for the previous replay: decode is
now GPU-bound (~75 ms/token) and the remaining lever is kernel tuning (the
`E=512, N=640, int4_w4a16` MoE config is still the default).

Verified on the dummy-weight mini config: decode graphs bs 1–8, a padded
batch (5 requests → bs 8), and MTP (`--speculative-algorithm NEXTN`) with
target-verify and draft graphs.
