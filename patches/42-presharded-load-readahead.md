# Patch 42: presharded reload — staged pinned reads, one file ahead

**File:** `python/sglang/srt/model_loader/loader.py`
**Script:** [`patch_presharded_load_readahead.py`](patch_presharded_load_readahead.py)
**Test:** [`../tests/check_presharded_load_readahead.py`](../tests/check_presharded_load_readahead.py)

## Symptom

Every restart reloads 75.1 GiB from the presharded dump in ~101 s — a flat
**758 MiB/s** (patch 31's progress line), while the same NVMe sustains
**4.2 GiB/s** with O_DIRECT reads.

## Cause (measured, not guessed)

* `safe_open(..., framework="pt").get_tensor()` returns **zero-copy views
  into the file's mmap** (verified: the returned tensor's data pointer lies
  inside the file's mapping in `/proc/self/maps`). Nothing is read at
  "read" time.
* The page-in therefore happens inside the consumer's
  `param_data.copy_(tensor)` — and `cudaMemcpy` from a file-backed mapping
  pin-faults the source pages one 4 KiB page at a time: measured
  **0.79 GiB/s** cold. That is the whole bottleneck.
* The GPU copy itself is free once the source is pinned: **66 GiB/s** from
  pinned memory.
* Kernel readahead hints don't help here: `POSIX_FADV_WILLNEED` on this
  kernel returns instantly and warms nothing (measured), and buffered
  reads cap at ~1.15 GiB/s single-stream (the page-cache copy is the
  limiter, not the device).

A v1 of this patch (fadvise WILLNEED|SEQUENTIAL + one-file-ahead
`get_tensor` prefetch) bought nothing for exactly these reasons: the
fadvise did nothing and the "prefetch" only materialized views. It is kept
out of the history; this is the real fix.

## Fix

Stage each dump file in pinned host memory on a background thread running
one file ahead of the apply loop:

1. `_pinned_views_from_buffer(path, keys, direct=True)`: parses the
   safetensors header, allocates one pinned buffer, `os.preadv`s the whole
   file into it with **O_DIRECT** (8 MiB chunks; the EOF remainder below
   512 bytes goes through a buffered fd because O_DIRECT needs 512-byte
   alignment of offset, length and buffer address — the pinned base is
   page-aligned, so whole-file-from-0 keeps every chunk aligned), and
   returns `torch.frombuffer` views over the buffer for the plan's keys.
2. The consumer's `copy_()` now transfers from pinned memory at PCIe
   speed; the NVMe read of the next file overlaps it on the prefetch
   thread (`_prefetched_file_reads`, one-deep, failures surface via
   `fut.result()` in order).
3. Fallback ladder on any staging error (O_DIRECT EINVAL/EIO, exotic
   dtypes, pin_memory unavailable): buffered-into-pinned, then the original
   `safe_open` mmap views — each step logs a warning so a silent fallback
   can't hide a perf regression.

Extra host memory: one staged shard (~1.3 GiB pinned). Tensors, order and
verification semantics unchanged.

## Hashing audit

Load-time tensor hashing (`verify_on_load`) defaults to **off** and this
deployment does not set it; `_audit_presharded_root` is stat-only.
`hash_num_threads` only bites at dump time. If `verify_on_load` is ever
enabled, its per-file hashing sits between staging and GPU-apply on the
critical path — folding it into the prefetch stage is future work.

## Verification

* `tests/check_presharded_load_readahead.py`: behavioral (with stubbed
  staging: file N+1's read is in flight before the consumer finishes file
  N, yields stay in plan order) and a real round-trip — pinned views read
  a freshly written safetensors file byte-identically for both
  `direct=True/False`, are pinned, and survive a `gc.collect()` after all
  staging-internal references are dropped.
* End-to-end after deploy: reload progress line MiB/s vs the 758 MiB/s /
  101 s baseline (target and draft phases), recorded in `bench/results.md`.
