# Patch 46 — PLE hot-set shadow cache (SIEVE, measurement only)

Status: **shipped 2026-11-21** (spike, iteration 1). Spec:
`~/.factory/specs/2026-11-21-ple-read-through-cache.md`.

## What this is

The first iteration of the read-through hot-set cache: a **shadow**. It
runs the proposed flat-hash + SIEVE structure alongside the real gather
path and reports, on the `Prefill batch` status line, what a real cache
would have hit — without changing any serving behavior. Its job is to
measure the one number that decides whether the kernel work (dense-read
gather + GPU probe) is worth building: **the actual hit rate of real
traffic**.

```
hotset: <hits>/<rows> rows <pct>% sz <size> ev <evictions> err <errors>
```

* Keys are table row ids (the ids the gather already computes; the patch-45
  hook already places them on the CPU every prefill chunk — no new syncs).
* No payload storage and no table reads on fill: shadow fills are free, so
  the hit rate reflects "if this cache were real", not current I/O. The
  row-level hit rate is a conservative lower bound for the context-level
  cache (16 rows per context assembly).
* Same SIEVE policy the real cache would use: lazy promotion on hit, FIFO
  insertion, hand-sweep eviction — the numbers include eviction realism.

## Structure (all flat arrays; single-owner: the scheduler forward thread)

* `keys[N]` int64 row ids, `visited[N]` SIEVE bits, `table[M]` slot index
  (M = 2N, linear probing with tombstones), `hand`/`tail` cursors.
* Keys are re-hashed through a splitmix64-style avalanche before slotting
  (raw row ids are Zipf-correlated; the mixer makes slot placement uniform).
* Batch-vectorized operations only: probe, touch-bits, and insert are numpy
  passes; eviction work is capped per call (4096 slots) — beyond that the
  shadow stops inserting (measurement turns conservative) instead of
  stalling the scheduler thread.
* Tombstones + rehash: eviction leaves tombstones to preserve linear-probe
  chain contiguity; when tombstones exceed M/4 the table is rebuilt
  vectorized from live keys (SIEVE bits reset — documented deviation; the
  real cache will carry epochs through the rebuild).
* Capacity from `SGLANG_QWEN4_PLE_HOTSET_ROWS` (default 1048576 rows ≈ the
  99%-coverage point from the census; 0 disables).

## Correctness lessons from the spike (each found by a failing test)

1. **@triton.jit decorator capture** (patch 45's lesson, re-learned here as
   a regression check): inserted code must never land between a decorator
   and its def.
2. **Tombstone deletion**: plain `-1` on delete breaks probe chains — live
   keys become unfindable. Tombstones or backshift are mandatory.
3. **Claim contiguity**: in a vectorized batch insert, contested slots must
   admit exactly one claimant per round (first occurrence), or the winner
   leaves an empty gap that makes earlier-claimed keys unfindable. A group
   whose members ALL defer is the degenerate case that first showed up.
4. **Group-index ordering**: numpy group-by built in sorted order must be
   mapped back through the argsort permutation before use.
5. **Bounded work**: eviction and insert loops are capped; probe-exhausted
   (orphan) slots are removed and counted, never left as unfindable
   zombies; `_table_remove` never spins on a missing entry.
6. The avalanche test initially "failed" at exactly the birthday-collision
   expectation — the threshold was wrong, not the hash. Test expectations
   get derived from uniform-hashing math, not vibes.

`tests/check_ple_hotset_shadow.py` covers all of the above in-image
(Zipf replay, no-phantom-hits, SIEVE policy, capacity consistency, bounded
work, drain semantics).

## What success looks like

The shadow's cumulative hit rate over a day of real serving. The census
predicts ~90% at a 169 MiB real cache (top 69.5k contexts); the shadow
measures what serving actually does. If the number is high, iteration 2
builds the real thing (dense payload array + gather kernel reading it);
if low, we saved the kernel work and revisit admission policy.
