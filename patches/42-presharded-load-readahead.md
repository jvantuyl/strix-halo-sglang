# Patch 42: presharded reload — staged reads one file ahead, DMA through a fixed pinned stage

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
* The GPU copy itself is free once the source is resident: **66 GiB/s**
  H2D from pinned host memory, 4.79 GiB/s from a warm mmap view.
* Kernel readahead hints don't help here: `POSIX_FADV_WILLNEED` on this
  kernel returns instantly and warms nothing (measured), and buffered
  reads cap at ~1.15 GiB/s single-stream (the page-cache copy is the
  limiter, not the device). O_DIRECT `os.preadv` into a prepared buffer
  sustains **4.24 GiB/s** on one stream; more reader threads don't scale
  past ~4.4 GiB/s on this controller.

## Fix

Stage each dump file into an **aligned pageable** CPU buffer on a
background thread running one file ahead of the apply loop, and route the
consumer's GPU copies through **one fixed 64 MiB pinned stage block**:

1. `_staged_views_from_buffer(path, keys, direct)`: parses the safetensors
   header, allocates one CPU buffer of `file_size + 1024` padded to a
   512-byte boundary, `os.preadv`s the whole file from offset 0 with
   **O_DIRECT** (every chunk aligned because the base is aligned and the
   read starts at 0; the sub-512-byte tail goes through a buffered fd),
   and returns `torch.frombuffer` views over the buffer for the plan's
   keys.
2. The consumer's `copy_()` from the staged view runs at ~3.2 GiB/s
   effective; the NVMe read of the next file overlaps it on the prefetch
   thread (`_prefetched_file_reads`, one-deep, failures surface via
   `fut.result()` in order). `QWEN38_PRESHARDED_SERIAL=1` disables the
   thread (serial staging) for debugging.
3. Large copies (≥1 MiB, contiguous) go through `copy_stage`, a single
   fixed pinned block (`_copy_to_gpu`): `cudaMemcpy` from pageable memory
   stages through torch's *internal pinned cache*, which keeps one cached
   block per distinct tensor size — on this UMA box those pinned pages
   showed up against the device memory accounting and one reload left
   ~3 GiB behind, unrecoverable by `empty_cache`. One fixed block avoids
   per-size caching entirely; `torch.cuda.synchronize()` before each
   reuse guarantees the previous async H2D finished reading it.
4. Fallback ladder on any staging error (O_DIRECT EINVAL/EIO, exotic
   dtypes): buffered-into-buffer, then the original `safe_open` mmap
   views — each step logs a warning so a silent fallback can't hide a
   perf regression. `QWEN38_PRESHARDED_STAGE=0` forces the original mmap
   path.

Why pageable staging instead of pinning the whole file: **on this UMA box
pinned host allocations are charged against the device memory the KV
budget is computed from** (patch 31-era measurement: 1 GiB pinned → 0.41
GiB device-free drop), and a v2 of this patch that staged into ~1.3 GiB
pinned per file booted into *"Loaded weights leave no GPU memory for the
KV cache"*. Patch 43 replaces that accounting with the carve's own sysfs
numbers, after which the distinction stops mattering for the budget — the
fixed small stage keeps the pinned footprint at 64 MiB regardless.

## Rejected variant: preparking the bandwidth-idle params

A v4 parked `embed_tokens.weight`/`visual.*` (patch 27's set, 2.02 GiB)
*before* the reload — `param.data` pointed at a fresh `hipHostMalloc`
buffer filled CPU→CPU from the staged views, so the bytes would never
enter VRAM. Load speed and memory accounting were fine, but the MTP draft
**stopped accepting entirely** (accept rate 0.93 → 0.00, decode 18 → 8
tok/s): the draft shares embedding storage with the target, and swapping
the target's `param.data` before the dump apply left the shared original
storage unfilled. The variant is kept out of the shipped patch; the
problem it aimed at (KV-budget accounting) is properly fixed by patch 43.

## Hashing audit

Load-time tensor hashing (`verify_on_load`) defaults to **off** and this
deployment does not set it; `_audit_presharded_root` is stat-only.
`hash_num_threads` only bites at dump time. If `verify_on_load` is ever
enabled, its per-file hashing sits between staging and GPU-apply on the
critical path — folding it into the prefetch stage is future work.

## Verification

* `tests/check_presharded_load_readahead.py`: behavioral (with stubbed
  staging: file N+1's read is in flight before the consumer finishes file
  N, yields stay in plan order) and a real round-trip — staged views read
  a freshly written safetensors file byte-identically for both
  `direct=True/False`, and survive a `gc.collect()` after all
  staging-internal references are dropped; `_copy_to_gpu` behavioral
  checks for the stage and fallback paths.
* End-to-end after deploy: reload **101 s → ~26 s** target + **10 s →
  ~2.6 s** draft (≈2.9 GiB/s effective), recorded in `bench/results.md`;
  draft MTP acceptance unchanged (accept len ≥ 2.5 on real traffic), KV
  budget per patch 43.
