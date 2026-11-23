#!/usr/bin/env python3
"""gfx1151 patch 53: keys-only hotset warm start (lazy payload refill).

Patch 46's warm start re-gathered the snapshot's payload bytes from the
47.7 GiB table mmap at every attach: ~160 MiB of scattered 4 KiB reads,
eagerly, before serving. On the 2026-11-23 boot that cost 1:44 (cold page
cache); a page-warm boot pays ~8 s. Either way it is avoidable work:
mapping and locking never required the pages to be resident.

Now the snapshot restores KEYS ONLY. Restored keys occupy slots and act
as SIEVE fodder (they evict before live keys), but their payload bytes
stay unfilled and every such slot carries an invalid bit:

- prepare() probes restored keys as MISSES (the kernel never sees an
  invalid slot — its payload is zeros), so the position goes through the
  stock mmap gather exactly as a cold key would;
- the deferred fill lands in _insert_batch's existing resident-key path,
  which refills the payload in place and clears the bit. No duplicate
  slots are created, so a probe can never hand out the garbage slot.

The reads amortize over live traffic and only rows actually touched ever
become resident — the hotset converges to warm under real load, and its
steady-state RAM holds the touched subset instead of the whole snapshot.

Files: sglang/srt/utils/ple_hotset.py (7 edits) and
sglang/srt/models/qwen4_exp.py (warm-start block).

Run from the Dockerfile with the target tree as argv[1].
"""

import argparse
import os
import sys

HOTSET = "python/sglang/srt/utils/ple_hotset.py"
QEXP = "python/sglang/srt/models/qwen4_exp.py"

EDITS = [
    # ---- __init__: invalid-bit state -----------------------------------
    (
        HOTSET,
        """        self._interval = self._zero()
        self._cumulative = self._zero()
        self._pending_misses = np.zeros(0, dtype=np.int64)
""",
        """        self._interval = self._zero()
        self._cumulative = self._zero()
        self._pending_misses = np.zeros(0, dtype=np.int64)
        # gfx1151 patch 53: payload-validity bits. Warm-started keys hold
        # no payload bytes (see restore_keys_only); an invalid slot probes
        # as a miss until its first touch refills it in place.
        self._invalid = np.zeros(n, dtype=bool)
        self._n_invalid = 0
""",
    ),
    # ---- _insert_batch: refill clears the invalid bit ------------------
    (
        HOTSET,
        """        if new_keys.size:
            present = self._probe(new_keys)
            upd = present >= 0
            if upd.any():
                self.payload[present[upd]] = payloads[upd]
                keep = ~upd
                new_keys = new_keys[keep]
                payloads = payloads[keep]
""",
        """        if new_keys.size:
            present = self._probe(new_keys)
            upd = present >= 0
            if upd.any():
                self.payload[present[upd]] = payloads[upd]
                # gfx1151 patch 53: a refilled resident key now holds real
                # bytes — clear its lazy-restore invalid bit.
                was_inv = self._invalid[present[upd]]
                if was_inv.any():
                    self._n_invalid -= int(was_inv.sum())
                    self._invalid[present[upd]] = False
                keep = ~upd
                new_keys = new_keys[keep]
                payloads = payloads[keep]
""",
    ),
    # ---- _insert_batch: fresh claims always carry real payloads ---------
    (
        HOTSET,
        """        self.keys[slots] = new_keys
        self.visited[slots] = 0
        self.payload[slots] = payloads
        self.size += k
""",
        """        self.keys[slots] = new_keys
        self.visited[slots] = 0
        self.payload[slots] = payloads
        # gfx1151 patch 53: a fresh insert carries a real payload; the
        # slot may be recycled from a never-touched restored entry.
        was_inv = self._invalid[slots]
        if was_inv.any():
            self._n_invalid -= int(was_inv.sum())
        self._invalid[slots] = False
        self.size += k
""",
    ),
    # ---- _sieve_evict: keep the invalid count honest --------------------
    (
        HOTSET,
        """        slot = self.hand
        self.hand = (self.hand + 1) % self.n
        self._table_remove(slot)
        self.keys[slot] = -1
        self.size -= 1
        return slot, steps + 1
""",
        """        slot = self.hand
        self.hand = (self.hand + 1) % self.n
        self._table_remove(slot)
        self.keys[slot] = -1
        # gfx1151 patch 53: evicting a never-touched restored key just
        # drops its placeholder bytes; keep the count honest.
        if self._invalid[slot]:
            self._invalid[slot] = False
            self._n_invalid -= 1
        self.size -= 1
        return slot, steps + 1
""",
    ),
    # ---- prepare: invalid slots probe as misses --------------------------
    (
        HOTSET,
        """                in_window = local_ids >= 0
                uniq = np.unique(local_ids[in_window])
                hit = self._probe(uniq)
                hits = int((hit >= 0).sum())
                self.visited[hit[hit >= 0]] = 1
""",
        """                in_window = local_ids >= 0
                uniq = np.unique(local_ids[in_window])
                hit = self._probe(uniq)
                # gfx1151 patch 53: a restored key with no payload yet
                # must NOT be handed to the kernel (its bytes are zeros).
                # Probe it as a miss; the position reads from the mmap and
                # the deferred fill refills the slot in place. The branch
                # is skipped entirely once the cache is fully warm.
                if self._n_invalid:
                    inv = (hit >= 0) & self._invalid[np.maximum(hit, 0)]
                    hit[inv] = -1
                hits = int((hit >= 0).sum())
                self.visited[hit[hit >= 0]] = 1
""",
    ),
    # ---- _rehash: carry invalid bits across the rebuild ------------------
    (
        HOTSET,
        """        payloads = self.payload[first_idx]
        self.table[:] = -1
        self.keys[:] = -1
        self.visited[:] = 0
        self.size = 0
""",
        """        payloads = self.payload[first_idx]
        # gfx1151 patch 53: carry lazy-restore bits across the rebuild.
        invalid = self._invalid[first_idx]
        self.table[:] = -1
        self.keys[:] = -1
        self.visited[:] = 0
        self._invalid[:] = False
        self._n_invalid = 0
        self.size = 0
""",
    ),
    (
        HOTSET,
        """        self._bump(rehashes=1)
        if live.size:
            self._insert_batch(live, payloads)
""",
        """        self._bump(rehashes=1)
        if live.size:
            self._insert_batch(live, payloads)
            # gfx1151 patch 53: _insert_batch clears the bits for fresh
            # inserts, so re-mark the keys that were restored-invalid.
            if invalid.any():
                slots = self._probe(live[invalid])
                ok = slots >= 0
                self._invalid[slots[ok]] = True
                self._n_invalid += int(ok.sum())
""",
    ),
    # ---- restore_keys_only: the new warm-start entry point ---------------
    (
        HOTSET,
        """        self._bump(fills=inserted)
        return inserted
""",
        """        self._bump(fills=inserted)
        return inserted

    def restore_keys_only(self, keys: np.ndarray) -> int:
        \"\"\"gfx1151 patch 53: warm-start with keys but NO payload bytes.

        Restored keys occupy slots (SIEVE fodder: they evict before live
        keys, giving the live set a larger runway at capacity) but their
        payload stays a placeholder: prepare() probes them as misses until
        the first touch refills the slot in place from the mmap. The boot
        no longer re-gathers the snapshot's ~160 MiB of scattered table
        pages; the reads amortize over live traffic, and only rows that
        are actually touched become resident.
        \"\"\"
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size > self.n:
            keys = keys[: self.n]
        inserted = 0
        chunk = 1 << 16
        for i in range(0, keys.size, chunk):
            k = keys[i: i + chunk]
            placeholder = np.zeros((k.size, self.row_bytes), dtype=np.uint8)
            self._insert_batch(k, placeholder)
            slots = self._probe(k)
            ok = slots >= 0
            if ok.any():
                self._invalid[slots[ok]] = True
                self._n_invalid += int(ok.sum())
            inserted += int(ok.sum())
        return inserted
""",
    ),
    # ---- qwen4_exp: call the lazy warm start ------------------------------
    (
        QEXP,
        """                    if _keys is not None and _keys.size:
                        _order = np.argsort(_keys)
                        _keys = _keys[_order]
                        _payloads = (
                            host_table.view(torch.uint8)[
                                torch.from_numpy(_keys)
                            ]
                            .numpy()
                        )
                        self._hotset.restore(_keys, _payloads)
""",
        """                    if _keys is not None and _keys.size:
                        # gfx1151 patch 53: keys-only warm start. The
                        # eager payload re-gather read ~160 MiB of
                        # scattered pages from the table mmap at every
                        # attach (1:44 on the 2026-11-23 cold-cache boot);
                        # restored keys now refill in place on first
                        # touch, so the reads amortize over live traffic.
                        _n = self._hotset.restore_keys_only(_keys)
                        print(
                            f"[qwen4_exp] PLE hotset: warm-started {_n} "
                            "keys (payloads lazy, refilled on first touch)",
                            flush=True,
                        )
""",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    texts = {}
    for i, (rel, old, new) in enumerate(EDITS, 1):
        path = os.path.join(args.target, rel)
        if path not in texts:
            with open(path) as f:
                texts[path] = f.read()
            if "gfx1151 patch 53" in texts[path]:
                sys.exit(f"patch 53 already applied ({rel})")
        assert texts[path].count(old) == 1, f"edit {i} anchor not found in {rel}: {old[:60]!r}"
        texts[path] = texts[path].replace(old, new, 1)
    for path, text in texts.items():
        with open(path, "w") as f:
            f.write(text)
    print("patch 53 applied: hotset warm start restores keys only; "
          "payloads refill in place on first touch "
          f"({len(EDITS)} edits across {len(texts)} files)")


if __name__ == "__main__":
    main()
