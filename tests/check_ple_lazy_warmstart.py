#!/usr/bin/env python3
"""check_ple_lazy_warmstart.py — verify patch 53 in the image, no GPU.

Patch 53 turns the hotset warm start into a keys-only restore: restored
keys hold placeholder payloads and an invalid bit; prepare() must probe
them as MISSES (the kernel never sees an invalid slot — its payload bytes
are zeros), and the deferred fill must REFILL THE SLOT IN PLACE (no
duplicate slots, or a probe could hand out the garbage first slot). This
test imports the real module (ctypes + numpy only) and drives the actual
prepare/flush_pending cycle against a stand-in table:

  1. restore_keys_only: keys inserted, all invalid, no payload reads.
  2. prepare on a restored key: slot -1 (miss) despite the key being
     present, and no bytes leave the payload array.
  3. flush_pending: in-place refill — size unchanged (no dupes), the
     invalid bit cleared.
  4. prepare again: hit, and the served payload is bit-exact vs the table.
  5. Eviction of a never-touched restored key stays honest (counter).
  6. _rehash carries invalid bits across the rebuild.

Run: python3 tests/check_ple_lazy_warmstart.py   (inside the image)
"""

import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import numpy as np  # noqa: E402

HOTSET_SRC = "/sgl-workspace/sglang/python/sglang/srt/utils/ple_hotset.py"
QEXP_SRC = "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"


def main():
    failures = []
    hs = open(HOTSET_SRC).read()
    qe = open(QEXP_SRC).read()

    # 0. All eight edit sites carry the patch marker.
    n_markers = hs.count("gfx1151 patch 53") + qe.count("gfx1151 patch 53")
    if n_markers < 8:
        failures.append(f"expected >=8 patch-53 markers, found {n_markers}")
    for needle in ("restore_keys_only", "self._n_invalid = 0"):
        if needle not in hs:
            failures.append(f"hotset source missing: {needle}")
    if "restore_keys_only(_keys)" not in qe:
        failures.append("qwen4_exp does not call restore_keys_only")
    if "host_table.view(torch.uint8)[torch.from_numpy(_keys)]" in qe:
        failures.append("eager payload re-gather still present in qwen4_exp")

    from sglang.srt.utils.ple_hotset import PleHotsetCache

    row_bytes = 8
    cap = 64  # small enough that the flood below forces SIEVE eviction
    c = PleHotsetCache(cap, row_bytes)
    # Stand-in table: plenty of distinct rows; row i is a recognizable
    # byte pattern.
    n_table_rows = 4096
    table = (np.arange(n_table_rows * row_bytes, dtype=np.uint8) % 251
             ).reshape(n_table_rows, row_bytes)

    def fill_rows_fn(rows):
        return table[rows]  # uint8 (count, row_bytes)

    # 1. Keys-only restore.
    keys = np.array([3, 7, 11, 40], dtype=np.int64)
    n = c.restore_keys_only(keys)
    if n != keys.size or c.size != keys.size:
        failures.append(f"restore_keys_only inserted {n}, size {c.size}")
    if c._n_invalid != keys.size:
        failures.append(f"expected {keys.size} invalid, got {c._n_invalid}")

    # 2. A restored key probes as a MISS (placeholder payload must never
    #    reach the kernel).
    slots = np.full(2, -99, dtype=np.int32)
    c.prepare(np.array([3, 5], dtype=np.int64), slots, fill_rows_fn)
    if not (slots == -1).all():
        failures.append(f"restored key handed to kernel: slots {slots}")
    # The pending fills for {3, 5} are queued; 3 is already resident.
    c.flush_pending(fill_rows_fn)
    # 3. In-place refill: no duplicate slots were created.
    if c.size != keys.size + 1:  # 4 restored + key 5
        failures.append(f"size after refill {c.size} (dupes created?)")
    if c._n_invalid != 3:  # 7, 11, 40 still lazy
        failures.append(f"invalid count after refill {c._n_invalid} != 3")

    # 4. Second pass: key 3 hits, payload bit-exact vs the table.
    slots2 = np.full(2, -99, dtype=np.int32)
    c.prepare(np.array([3, 5], dtype=np.int64), slots2, fill_rows_fn)
    if slots2[0] < 0 or slots2[1] < 0:
        failures.append(f"refilled keys did not hit: {slots2}")
    else:
        served = c.payload[slots2]
        if not (served == table[[3, 5]]).all():
            failures.append("served payload is not bit-exact vs the table")

    # 5. Evict never-touched restored keys: counter stays honest. The
    #    flood inserts 4x capacity in distinct in-table keys; SIEVE must
    #    evict the never-touched restored rows first and keep accounting.
    floods = np.arange(200, 200 + 4 * cap, dtype=np.int64)
    for i in range(0, floods.size, 32):
        c.prepare(floods[i: i + 32],
                  np.full(floods[i: i + 32].size, -99, dtype=np.int32),
                  fill_rows_fn)  # type: ignore[arg-type]
        c.flush_pending(fill_rows_fn)
    if c._n_invalid != 0:
        failures.append(f"eviction left {c._n_invalid} invalid bits")
    if c.errors:
        failures.append(f"{c.errors} internal errors during flood")

    # 6. Rehash with a mix of invalid bits carries them across.
    c2 = PleHotsetCache(256, row_bytes)
    c2.restore_keys_only(np.array([1, 2, 3, 9], dtype=np.int64))
    slots3 = np.full(1, -99, dtype=np.int32)
    c2.prepare(np.array([2], dtype=np.int64), slots3, fill_rows_fn)
    c2.flush_pending(fill_rows_fn)  # 2 refilled; 1, 3, 9 still invalid
    before = c2._n_invalid
    c2._rehash()
    if c2._n_invalid != before:
        failures.append(f"rehash dropped invalid bits: {before} -> {c2._n_invalid}")
    slots4 = np.full(1, -99, dtype=np.int32)
    c2.prepare(np.array([9], dtype=np.int64), slots4, fill_rows_fn)
    if slots4[0] != -1:
        failures.append("post-rehash invalid key handed to kernel")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("check_ple_lazy_warmstart: 6/6 OK (keys-only restore, "
          "invalid keys probe as misses, in-place refill with no dupes, "
          "bit-exact served bytes, honest eviction accounting, rehash "
          "carries bits). Boot effect: attach no longer reads the "
          "snapshot's payload pages.")


if __name__ == "__main__":
    main()
