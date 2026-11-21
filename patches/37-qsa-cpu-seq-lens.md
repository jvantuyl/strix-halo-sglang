# Patch 37: QSA opts into seq_lens_cpu — sync-free speculative metadata

**File:** `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
**Script:** [`patch_qsa_cpu_seq_lens.py`](patch_qsa_cpu_seq_lens.py)
**Test:** [`../tests/check_qsa_cpu_seq_lens.py`](../tests/check_qsa_cpu_seq_lens.py)

## Symptom

Every boot logs, once:

```
QSA speculative metadata without CPU request lengths: the
token-slot bound reads them back once per forward
```

That is not just a warning: in the speculative forward modes (TARGET_VERIFY
and DRAFT_EXTEND_V2) the QSA metadata builder needs a host-side over-bound on
each row's token-slot gather width. `_speculative_max_row_length()` prefers
the sync-free `seq_lens_cpu.max() + draft_window`, but the speculative
ForwardBatches carried `seq_lens_cpu=None`, so the fallback ran:
`int(sequence_lengths.max())` on a device tensor — a stream sync plus a D2H
copy per forward, four per MTP round (3 draft + 1 verify). A stream sync
serializes the host against everything the GPU still has queued.

## Root cause

`decide_needs_cpu_seq_lens()` (`managers/overlap_utils.py`) gates whether the
overlap scheduler's FutureMap publishes `seq_lens_cpu`/`seq_lens_sum` to the
next round's batches. It is an OR over the backends of the spec-v2 chain
(target, draft, draft-extend — see `EagleWorkerV2.spec_v2_attn_backends`).
Both QSA backend classes declared `needs_cpu_seq_lens = False`, mirroring
upstream's DSA backend — which can afford the opt-out because its metadata
path is fully device-side. QSA's is not: its speculative branch consumes the
host copy when present and pays a per-forward readback when absent. The
backend's own comment ("the graphed decode path never reads it, so opting out
is safe") was true for plain decode and false for the speculative path.

Meanwhile the eagle worker already materializes `new_seq_lens.to("cpu")` once
per round for the draft path (its comment carries the TODO "one D2H per
round (TODO: async or upper-bound estimate)").

## Change

The main `QwenSparseAttnBackend` declares `needs_cpu_seq_lens = True`. The
multi-step draft backend keeps False: only one backend needs to opt in for
the OR to go True, and the draft decode steps genuinely never read it.

## Expected effect

The per-forward readback disappears from target-verify and draft-extend; the
sync-free host bound takes over. In the 2026-11-19 decode profile the
speculative forwards showed ~66 GPU-idle stalls of 10–185 ms over ~81 MTP
rounds under rocprofv3 — profiler overhead inflates those, so the production
win is smaller (a 2026-11-20 event-stream probe of production decode showed
tight 144 ± 15 ms rounds with no stalls), but the readback itself is real and
this removes it at the cost of nothing: the host copy already existed.
