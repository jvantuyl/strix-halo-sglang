# Patch 40: QSA indexer rope cache sized from a host bound

**Files:** `python/sglang/srt/layers/attention/qsa/metadata.py`,
`python/sglang/srt/layers/attention/qsa/qsa_indexer.py`,
`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_rope_host_bound.py`](patch_qsa_rope_host_bound.py)
**Test:** [`../tests/check_qsa_rope_host_bound.py`](../tests/check_qsa_rope_host_bound.py)

## Symptom

After patch 39 removed the draft-extend metadata sync, a py-spy capture
during a cold 64k-token prefill showed the scheduler thread's next
dominator: **83% of thread samples** inside the QSA indexer's rope path —

```
self_attention → _compute_qsa_topk_indices → project_qk → apply_rope
```

## Cause

`qsa_indexer.project_qk` (fused-prep path) and `qsa_indexer.apply_rope`
both size the lazy rotary cache with

```python
self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
```

`positions.max().item()` is a device reduction + D2H sync **per call** —
per QSA layer per prefill chunk (12 layers × 63 chunks per 64k-token
prompt) and several times per decode round. Same class as the syncs patches
37 and 39 removed: the scheduler thread blocks on a value the host already
knows.

## Fix

The rope cache stays lazy (base.py's `_ensure_cos_sin_cache_length` is
grow-only and computes incrementally), but the per-call device readback is
replaced by a monotone host-side bound:

* `qsa/metadata.py` gains `QSA_HOST_MAX_POSITION` (module-level holder) and
  two helpers: `record_qsa_host_max_position(seq_lens_cpu, draft_window)`
  and `_host_position_bound(positions)` — both files already import from
  this module, so no new import edges.
* `QwenSparseAttnBackend` records `max(seq_lens_cpu) + draft_window` in
  `_metadata_from_forward_batch` and `_replay_cuda_graph_metadata` — the two
  entry points every serving forward passes through before any layer runs.
  The draft window mirrors `_speculative_max_row_length`'s bound: verify
  and draft rows position past the pre-write sequence length.
* `project_qk` / `apply_rope` call
  `_ensure_cos_sin_cache_length(_host_position_bound(positions))`. Until a
  bound is recorded (boot paths without the overlap publication) the helper
  falls back to the original device readback, so correctness never depends
  on the hint; an over-large bound only grows the cache slightly earlier,
  and a stale bound cannot corrupt numbers (the cache index would fail
  loudly, not silently).

No numerics change: cache contents are identical for every position used.

## Verification

* `tests/check_qsa_rope_host_bound.py` (in-image, no GPU): holder
  monotonicity, `max + draft_window` semantics, list/tensor/None inputs,
  fallback precedence, both indexer sites converted, both backend entry
  points recording.
* End-to-end after deploy: prefill cadence re-measured against the patch-39
  numbers and the Nov-14 baseline; py-spy prefill capture re-run. Results in
  `bench/results.md`.
