# Patch 39: QSA speculative metadata built from host-known lengths

**File:** `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_host_metadata.py`](patch_qsa_host_metadata.py)
**Test:** [`../tests/check_qsa_host_metadata.py`](../tests/check_qsa_host_metadata.py)

## Symptom

Watching `amdgpu_top` during the 64k-token mission workload showed GPU
utilization oscillating with idle periods during prefill, where earlier runs
had looked pegged. The decode-round probes (patch 35 event stream) showed no
stall storm — the idle lives *inside* the prefill cadence and as a diffuse
share of every decode round.

## Measurement

`py-spy record` (non-blocking, 100 Hz) on the production scheduler over a
mission window: the scheduler thread was **~95% busy** — not waiting — and
**75% of its busy time** sat in one chain:

```
run_batch → _draft_extend_for_decode → prepare_for_draft_extend
  → init_forward_metadata → _metadata_from_forward_batch
  → _speculative_row_to_request          ← leaf, self-time
```

`_speculative_row_to_request`'s extend branch computed
`int(repeats.sum().item())` on a device tensor. `prepare_for_draft_extend`
runs batch preparation on a separate plan stream; a `.item()` there blocks
the scheduler thread until that stream (and its event dependencies) drain —
i.e. it waits out whatever forward is still in flight. For a prefill chunk's
draft-extend that is the remaining GPU time of the 1024-token chunk itself:
the thread stalls up to ~1 s per chunk (py-spy: ~21 s of a 30 s window in
this chain while the mission prefilled), the chunk cadence becomes
thread-bound (~1.55 s/chunk vs ~1.4 s of GPU), and the GPU idles visibly.
The same sync costs a share of every decode round.

The other two per-forward syncs of the same class turned out to be cold:
the CUDA-graph update path stages via `.cpu()` only as a documented
non-serving fallback (`_can_replay_with_gpu_kernels` takes the replay path),
and `_qsa_build_write_plan` is sync-free by design. A third one —
`req_pool_indices.tolist()` in the chunk-prefill `forward_extend` path —
runs per QSA layer per chunk and reads a value the overlap scheduler
already publishes on the host.

## Fix

Both sites now read the host-published values instead of round-tripping
through the device:

1. `_speculative_row_to_request`: when `forward_batch.extend_seq_lens_cpu`
   is present (the overlap FutureMap publishes it — the same routing patch
   37 gave `seq_lens_cpu`), the row→request mapping is built with torch CPU
   ops and moved to the device once, at the end. No `.item()`, no device
   dispatch chain. The device-tensor path is kept as fallback for callers
   without the CPU list.
2. The chunk-prefill `forward_extend` gather uses
   `forward_batch.req_pool_indices_cpu` when present; a CPU-tensor
   `.tolist()` does not touch the compute stream.

No numerics change — the mapping values are identical, only where they are
computed.

## Verification

* `tests/check_qsa_host_metadata.py` (in-image, no GPU): the host path and
  the device fallback path produce identical mappings for decode-like,
  prefill-chunk-like, multi-request and zero-extend shapes, including
  padding; out-of-range `num_rows` still raises; the host branch contains
  no `sum().item()`.
* End-to-end (2026-11-20): mission prefill cadence and decode gap p50
  re-measured after deploy; py-spy re-run confirms the chain leaves the
  profile's top spots. Results recorded in `bench/results.md`.
