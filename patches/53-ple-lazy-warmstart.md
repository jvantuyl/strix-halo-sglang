# Patch 53 — keys-only hotset warm start (lazy payload refill)

**Files:** `python/sglang/srt/utils/ple_hotset.py`, `python/sglang/srt/models/qwen4_exp.py` · **Test:** `tests/check_ple_lazy_warmstart.py` (6/6 in-image, no GPU)

## Symptom

Even after patch 52 removed the mlock upfront fault, the first attach
window of the 2026-11-23 boot took 1:44: the warm start re-gathered the
snapshot's payload rows from the 47.7 GiB table mmap eagerly — ~160 MiB of
scattered 4 KiB reads before the server could serve, plus THP-granular
fault-in of the payload. Mapping and locking never required those pages.

## The patch

`restore_keys_only(keys)` replaces the eager restore. Restored keys occupy
slots (SIEVE fodder: they evict before live keys, giving the live set the
runway the 2M capacity was raised for) but hold placeholder payloads and
carry an invalid bit:

- `prepare()` maps invalid hits to **misses** — the kernel never sees an
  invalid slot (its payload bytes are zeros), so the position goes through
  the stock mmap gather exactly like a cold key.
- The deferred fill lands in `_insert_batch`'s existing resident-key path,
  which **refills the payload in place** and clears the bit. No duplicate
  slots are ever created, so a probe can never hand out the placeholder.
- Eviction of never-touched restored keys and `_rehash` keep the invalid
  accounting honest; the `prepare` mask is skipped entirely once the
  cache is fully warm (`_n_invalid == 0`).

The reads amortize over live traffic, and only rows actually touched become
resident: the hotset converges to warm under real load, holding the
touched subset rather than the whole snapshot.

## Safety

The only correctness rule is "an invalid slot must never be served", and
there is exactly one serving path (`prepare` → `slots_out` → kernel reads
`payload[slot]`); it is gated before `hits`/`visited` are computed. The
in-image test drives the real prepare/flush cycle: restored keys probe as
misses, refill creates no dupes (size unchanged), served bytes are
bit-exact vs the fill source, eviction drives `_n_invalid` to 0 without
errors, and rehash carries the bits.

## Effect

Boot no longer reads the snapshot's payload pages at attach (the 1:44
first window collapses; the draft window was already 1 s after patch 52).
Steady state converges to the same warm hotset under live traffic. The
snapshot format is unchanged (`save_hotset_snapshot` is keys-only already),
so an existing snapshot restores exactly as before.

## Trade-offs accepted

- The first minutes after a boot see the hotset filling from real traffic
  (fill burst) instead of being warm at ready — the same cold-start curve
  the cache had before patch 46, but with the key set pre-seeded.
- Restored-but-never-touched keys consume hash/slot space until evicted;
  with 2M rows and ~1M restored that is the intended runway, not a leak.
