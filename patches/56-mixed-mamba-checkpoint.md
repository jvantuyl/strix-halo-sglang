# 56 — Preserve Mamba prefill checkpoints in MIXED batches

On SGLang `70b5b03e`, `prepare_for_extend` picks a ping-pong slot and
records the checkpoint depth before the scheduler decides whether a prefill
chunk will run alongside decoding requests. `mix_with_running` calls
`merge_batch`, which clears that tracking metadata. The mixed forward then
skips the checkpoint write, but the radix cache can still take ownership of
the slot at the recorded depth. A later prefix hit can restore stale state
without an error.

`patch_mixed_mamba_checkpoint.py` changes only `schedule_batch.py`. It
preserves the three device tensors (slot IDs, mask, sequence lengths) and
two optional host mirrors (prefill mask and sequence lengths) across the
mixed merge for non-lazy `extra_buffer`. Decode-tail rows keep their own
slot IDs for backend translation and graph inputs, but are padded with
`False` masks and `-1` sequence lengths, so they never write checkpoints
through the prefill path. Missing host mirrors stay missing; decode-only
bookkeeping stays cleared.

The tail uses the running batch's prepared track indices if present. For
NEXTN/EAGLE, where track indices can instead be prepared inside the verify
forward, it gathers the request's current ping-pong position from the pool
mapping without mutating the running batch or the requests. The verify
setter shares that lookup, including the `-1` no-op destination for a
freed row whose old slot might have been reused. The host position tensor
is pinned only on supported devices.

The backport follows [#39526](https://github.com/sgl-project/sglang/pull/39526)
for live tail slot IDs and the pure lookup, and the prefill-only restore
scope in [#45435](https://github.com/sgl-project/sglang/pull/45435). It does
not backport `extra_buffer_lazy`: that strategy also needs to undo decode
boundary preallocation and handle the overlapped finish path. Generic
merges, result processing and allocators are unchanged.

CPU regression:

```sh
python3 tests/check_mixed_mamba_checkpoint.py /sgl-workspace/sglang
```

The test executes the patched scheduler and backend method bodies with CPU
tensors and fixture pools, without importing the model runtime. It checks
the prepared-index and NEXTN paths, optional mirrors, freed rows, and
that a boundary decode row is not a writer. It runs the backend track plan
and SSM copy and verifies that the actual donation and copy-on-write
selection methods identify the written state. It does not execute a
model kernel, full radix traversal, CUDA graphs, or GPU generation.
