# Patch 44 — PLE row prefetcher: real background warm reads (DRAFTED, NOT APPLIED)

Status: **drafted 2026-11-21, not applied, not built, decision pending.**
The script exists (`patch_ple_warm_prefetch.py`) but is not in the Dockerfile
chain. It was parked when the user halted implementation pending forensics.

## What it would do

`PleFilePrefetcher` (patch 13-era) hints the page cache about the pages a
prefill-sized gather is about to read via `posix_fadvise(WILLNEED)`, one call
per page, on a background thread. On this kernel that syscall is a no-op —
measured twice (2026-11-21): on the presharded dump files during the patch-42
work, and directly on this table (2 GiB range, residency 30.1% → 30.3%
across the WILLNEED). The prefetch therefore does nothing here, and a cold
prefill pays for its cold rows synchronously inside the gather.

The draft replaces the fadvise loop with real reads: the sorted page list is
merged into contiguous spans and each span is `preadv`-ed into a 4 MiB
scratch buffer, data discarded, page cache populated. Same read volume, but
off the scheduler's critical path and at NVMe speed.

## Forensics since the draft was written (revises the rationale)

* The mapping is **4 kB pages**, not 2 MB folios (`KernelPageSize: 4 kB` in
  the scheduler's smaps) — the "100× read amplification" idea from the
  original draft notes is wrong; a cold row miss is a cheap 4 kB read.
* The mapped-in row set creeps ~30 kB/token of distinct rows (0.72 → 4.13
  GiB over an hour of probes) and the kernel's own reclaim trims it
  continuously before the 8 GiB trim budget (`PleFileRssTrimmer`, default
  `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=8.0`) ever fires. Zero trim events is
  the expected steady state on this memory-tight box.
* The constant ~19 MiB/chunk disk reads during the 62k probe were
  **first-touches of new rows**, not eviction churn: the probe content
  (shuffled JSONL preference pairs) has near-zero n-gram repetition, so every
  chunk hits ~1000 never-seen rows. Natural prose repeats n-grams, so the
  row working set is compact and reads taper within a prompt.
* Consequently the "7% prefill regression vs the 730-735 historical p50" is
  **not established** — the comparator was structurally unfair (shuffled
  content, cold boot, first run vs repeated natural-prose runs).

## Why it is parked

With first-touch as the real mechanism, the value of background warming
depends on whether representative (natural-prose) long-context prefill is
actually measurably slower than the historical baseline. Decision deferred
until a fair re-baseline exists (natural-prose corpus in `bench/decode_wall.py`,
cold-then-repeat on one boot). If the fair baseline shows no regression, this
patch should be rejected as-is; if it shows one, the draft's span-merged
background read is the leading candidate.
