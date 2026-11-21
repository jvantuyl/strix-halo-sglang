# Patch 45 — PLE gather hit statistics on the status line

Status: **shipped 2026-11-21** (script, in-image test, Dockerfile step).

## Problem

The PLE table (47.7 GiB fp8) is read through a file-backed mmap; whether a
prefill chunk is cheap depends on how many of its rows' pages are already
resident in the page cache. Until now that was only measurable out-of-band
(`sudo mincore` over smaps ranges, `read_bytes` sampling) — not something
the server reports during normal serving, and not comparable across boots.

## What it adds

A `PLE:` segment on the scheduler's `Prefill batch` status line:

```
PLE: <unique>/<rows> rows, <resident>/<pages> pages res, <err> err
```

* **rows** — row reads the batch asked of the table (one per token plus the
  n-gram window).
* **rows/rows_unique** — in-batch duplication (sorted-table locality).
* **pages** — distinct 4 KiB pages those rows touch.
* **resident** — pages that `mincore(2)` reports resident in the page cache
  at gather time. The hit prediction: resident = minor fault, non-resident
  = disk I/O.
* **err** — recording errors since boot (recording disables itself after
  the first; diagnostics must never take the server down).

Counters are per-report-interval (drained by the metrics reporter, like the
other server numbers) and cumulative-since-boot via
`PleHitStats.drain(cumulative=True)`.

## Ground truth, not estimates

Residency is `mincore(2)` on the mapping itself. Verified semantics
(`tests/check_ple_hit_stats.py`, runs in-image):

* a page just written reads resident = 1;
* untouched far pages read 0 (the immediate neighbour of a fault may read 1
  from kernel readahead — real residency, not a false positive);
* `pages_for_rows` page math checked against 160-byte rows.

## Cost

Per prefill chunk: one ids D2H copy, one `torch.unique`, and ≤ ~1100
one-page mincore syscalls (~1-2 ms on the scheduler thread vs ~1 s chunks;
the gather fault path it observes already costs ~5% of the chunk). CUDA-graph
replay never re-enters `gather()`, so decode latency is untouched. Gathers
below 512 rows are skipped.

## Forensics context (2026-11-21)

This instrument exists because the earlier "7% prefill regression" story
kept changing shape under measurement: the probe content was the dominant
variable (shuffled preference-pair text has near-zero n-gram repetition and
hits ~1000 never-seen rows per chunk, while natural prose repeats n-grams
and tapers within a prompt), the mapping's resident set is bounded by kernel
reclaim long before the 8 GiB `PleFileRssTrimmer` budget, and fadvise
WILLNEED is a no-op on this kernel. With `PLE:` on the status line, the
row-cache hit rate is a first-class server number for the re-baseline work
and for judging any future PLE change.
