# Patch 41: QSA prefill MQA inputs from host mirrors

**Files:** `python/sglang/srt/layers/attention/qsa/metadata.py`,
`python/sglang/srt/layers/attention/qsa/sparse_attn.py`,
`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_prefill_host_mirrors.py`](patch_qsa_prefill_host_mirrors.py)
**Test:** [`../tests/check_qsa_prefill_host_mirrors.py`](../tests/check_qsa_prefill_host_mirrors.py)

## Symptom

After patch 40 removed the indexer rope readback, a py-spy capture during a
cold 64k-token prefill showed the next dominator at **82% of scheduler
thread samples**: `QwenSparseAttnMetadata.get_prefill_mqa_inputs` — and the
64k prefill wall barely moved (83.0 vs 84.4 s). Sequential per-layer syncs
mask each other: removing one exposes the next, and the wall only drops
when the chain is clear.

## Cause

Two more device readbacks in the same per-layer-per-chunk chain:

1. `get_prefill_mqa_inputs`:
   `sequence_lengths_list = sequence_lengths.tolist()` — a device→host
   sync per QSA layer per prefill chunk. For a non-speculative forward the
   metadata's `sequence_lengths` is exactly `forward_batch.seq_lens`, whose
   overlap-published mirror is `forward_batch.seq_lens_cpu`.
2. `sparse_gqa_fwd_interface_triton_ck`:
   `max_q = int((cu_q[1:] - cu_q[:-1]).max().item())` — a device readback
   for the launch grid's row bound, while the caller (`forward_extend`)
   already holds the host list `extend_seq_lens_cpu`, whose max is the same
   number.

## Fix

1. `QSAIndexerMetadata` (msgspec struct in `qsa/metadata.py` — the class
   `get_prefill_mqa_inputs` belongs to) gains `sequence_lengths_cpu = None`
   as the last field; existing construction sites are unaffected. The
   non-speculative branch of `_metadata_from_forward_batch` passes it from
   `forward_batch.seq_lens_cpu`. The speculative-paged and graph-capture
   constructions deliberately leave it None: paged rows are per-token
   rather than per-request, so a `seq_lens_cpu` mirror would be silently
   wrong there — `get_prefill_mqa_inputs` is a non-speculative path, and
   the None fallback keeps the original `.tolist()` behavior anywhere the
   mirror is not exact.
2. `get_prefill_mqa_inputs` prefers the mirror (tensor or list); fallback
   unchanged.
3. The kernel wrapper accepts `max_q=None`; the backend's chunk-prefill
   caller passes `max(extend_lens)` from the host list.

No numerics change.

## Verification

* `tests/check_qsa_prefill_host_mirrors.py` (in-image, no GPU): struct
  field + default, mirror-preferred read logic, kernel kwarg + fallback,
  backend sites, behavioral mirror semantics.
* End-to-end after deploy: cold 64k prefill wall and chunk cadence vs the
  patch-39/40 numbers (84.4/83.0 s, tok/s p50 712) and the Nov-14 baseline
  (98 s, 679 tok/s); py-spy capture re-run. Results in `bench/results.md`.
