# Patch 52 — `mlock2(MLOCK_ONFAULT)` for the PLE hot-set payload

**File:** `python/sglang/srt/utils/ple_hotset.py` · **Test:** `tests/check_ple_hotset_mlock2.py` (in-image, no GPU)

## Symptom

The 2026-11-23 boot (cap 12 + 2M-row hotset) took 5m45s instead of ~65 s.
Two silent ~2-minute windows sat exactly between the `PLE table:
file-backed mmap` log line and the `WILLNEED prefetch` line, once per
table attach (target + draft). Yesterday's boots at 1M rows: 8 s and 0 s
for the same phases; weight-load elapsed 42 s → 264 s.

## Root cause

`PleHotsetCache._mlock` (patch 46) locks the whole payload array with
plain `mlock` right after `np.zeros`/calloc. Plain mlock on untouched
pages faults every page in synchronously: at 2M rows that is 320 MiB per
hotset, twice per boot, under this host's RAM pressure — ~2 min each.
The snapshot re-gather (`~8 s`) and `torch.from_file` mmap attach (lazy,
instant) are innocent; the sparse-table `ftruncate` never even ran (the
table file was reused).

## The patch

`mlock2(addr, len, MLOCK_ONFAULT)` instead: pages become unevictable only
as the cache touches them, so the steady-state protection is identical and
only the boot-time force-fault disappears. Fallbacks preserve every earlier
behavior:

1. glibc `mlock2` symbol → ONFAULT lock;
2. missing symbol → `syscall(325, …, MLOCK_ONFAULT)` (x86_64);
3. both fail (pre-4.4 kernel) → plain `mlock`, the old upfront-fault cost;
4. that fails too → `mlocked = False`, unlocked operation, as in patch 46.

## Effect

Boot cost of the hotset at any row count drops to ~0; the 2M-row rollout
keeps its runtime benefit (12k/hr churn evidence, patch 46 docs) without
the 4-minute boot penalty. At 1M rows nothing changes measurably.

## Trade-offs accepted

- Pages are locked only once touched, so a freshly booted hotset starts
  unlocked-but-locking-on-fill; the warm-start snapshot restore faults its
  pages in at restore time and they lock then — a few hundred MiB of
  locking moves to the first seconds of serving instead of attach.
- `RLIMIT_MEMLOCK` accounting is unchanged (the same pages eventually lock).
- The syscall-number fallback is x86_64-specific; other arches fall to
  plain mlock if glibc lacks mlock2. The image is x86_64-only.
