# Patch 46 — PLE read-through hot-set cache (RAM MVP)

Status: **shipped 2026-11-21** (MVP, iteration 2 — replaces the shadow
spike, which measured a 100% row-level hit rate on warm traffic and
validated the SIEVE structure). Spec:
`~/.factory/specs/2026-11-21-ple-read-through-cache.md`.

## What this is

A real read-through cache in front of the PLE table's mmap: the hottest
rows live in a dense, **mlock'd** host-RAM array; gathers serve hit rows
from RAM and miss rows from the mmap exactly as before. This removes the
page-cache dependency for hot rows — the kernel page cache is a shared,
evictable, 8 GiB-budget trimmer; the hot set is private and pinned.

* `SGLANG_QWEN4_PLE_HOTSET_ROWS` (default 1048576 ≈ 152 MiB ≈ the census's
  99%-coverage point; `0` disables).
* Warm start: a keys-only checksummed snapshot beside the table
  (`<table dir>/hotset-snapshot.bin`, ~8 B/row, atomic rename, sha256
  digest, fingerprinted to the table file's size+mtime and the shard
  range). On boot the key set is restored and the payload bytes are
  re-gathered from the mmap in row-id order (a sorted ~39k-page sweep,
  a few seconds), so the cache is warm from the first prefill. Saved by
  the scheduler thread at most every 5 minutes when the size changed.
* Capacity overflow, mlock failure, or any internal error degrades to the
  stock mmap path — the cache can only fail towards today's behavior.

## Where it hooks (the production path)

On HIP the default gather is the **CPU path** (`_gather_ple_embedding_on_cpu`;
`SGLANG_PLE_UVA_GATHER=1` would restore the device-side triton kernel —
unused here). The cache intercepts exactly there:

1. Probe the batch's local row ids against the flat hash table (one
   vectorized numpy pass; the ids are already on the CPU — no new syncs).
2. Hit rows are copied from the dense payload array into the output
   (uint8 view → dtype view → bf16, byte-identical conversion to the
   mmap path); miss rows go through the stock `index_select` on the mmap.
3. Fills are **one-chunk-lag**: a batch's misses are copied from the mmap
   at the *next* batch's prep, when the kernel's own reads have already
   made those pages resident — fills are minor faults, never disk I/O.
4. CUDA-graph capture/replay never touches the cache (replay serves from
   the prefetch buffer; decode's working set is resident anyway).

## Structure (flat arrays; single owner: the scheduler forward thread)

* `keys[N]` int64 row ids, `payload[N, row_bytes]` uint8 (mlock'd),
  `visited[N]` SIEVE bits, `table[M]` slot index (M = 2N, linear probing
  with tombstones), `hand`/`tail` cursors; splitmix64-style avalanche
  before slotting (raw row ids are Zipf-correlated).
* SIEVE policy: lazy promotion on hit, FIFO insertion, hand-sweep
  eviction; eviction and insert work is capped per call; probe-exhausted
  orphans are removed and counted, never left unfindable.
* Tombstones on eviction preserve probe chains; when they exceed M/4 the
  table rebuilds vectorized from live keys (SIEVE bits reset — documented
  deviation).
* Keys already resident at fill time (two batches missed the same key
  before its fill landed) update the payload in place instead of
  re-inserting — a blind re-insert leaves a stale shadow entry.

## Tests (`tests/check_ple_hotset_shadow.py`, in-image)

Probe/payload round-trip, one-chunk-lag fill semantics, duplicate-key
rejection and update-in-place, SIEVE eviction with survivor payload
integrity, rehash correctness, capacity accounting — and the critical
one: **bit-exact output** of the cached CPU gather vs the original path
on mixed hit/miss/out-of-range batches (bitwise int16 compare, NaN-safe:
fp8 0x7F+ patterns are NaN and must round-trip identically).

## Expected effect

Census says ~90% row-level coverage at 169 MiB; the shadow measured 100%
hits on warm traffic. Steady-state gain is small (minor-fault removal,
~2-4%; cold-heavy phases ~5-8%); the main value is tail-latency
predictability and independence from page-cache pressure.

## Order-based overlap (shipped, same commit series)

`prepare()` only probes and records the batch's misses; the caller
flushes the pending fill after its own output is computed, so the
~5-20 ms of mmap gather + insert work sits outside the prep critical
path. Same thread, no races; the lag relative to the missing batch is
unchanged.

## Follow-up (scheduled)

* (none currently; snapshot persistence shipped)
