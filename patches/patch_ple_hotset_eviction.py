#!/usr/bin/env python3
"""Patch 61: bounded replacement and truthful fills in the PLE row cache."""

import hashlib
import sys
from pathlib import Path


METHODS = '''    def _probe(self, uniq):
        buckets = self._buckets(uniq)
        return np.where(buckets >= 0, self.table[buckets], -1)

    def _reserve_buckets(self, table, keys):
        """Reserve a reachable hash bucket for each candidate, or return -1."""
        s = (avalanche(keys) & self.mask64).astype(np.int64)
        buckets = np.full(keys.size, -1, dtype=np.int64)
        remaining = np.arange(keys.size)
        for _ in range(64):  # same probe bound as lookup
            if not remaining.size:
                break
            sv = s[remaining]
            _, first = np.unique(sv, return_index=True)
            available = table[sv[first]]
            first = first[(available == -1) | (available == -2)]
            buckets[remaining[first]] = sv[first]
            table[sv[first]] = -3  # reserved under the cache lock
            keep = np.ones(remaining.size, dtype=bool)
            keep[first] = False
            remaining = remaining[keep]
            s[remaining] = (s[remaining] + 1) & (self.m - 1)
        return buckets

    def _evict_slots(self, count):
        """Select count distinct victims out of the dense live slot prefix."""
        slots = np.empty(count, dtype=np.int64)
        taken = steps = 0
        while taken < count:
            start = self.hand
            end = min(self.size, start + max(count - taken, 1024))
            flags = self.visited[start:end]
            fresh = np.flatnonzero(flags == 0)[:count - taken]
            if taken + fresh.size == count:
                end = start + int(fresh[-1]) + 1
                flags = self.visited[start:end]
            flags[flags == 1] = 0  # second chance on this pass
            flags[fresh] = 2  # cannot select the same slot again on wrap
            slots[taken:taken + fresh.size] = start + fresh
            taken += fresh.size
            steps += end - start
            self.hand = end % self.size
        old_buckets = self._buckets(self.keys[slots])
        assert (old_buckets >= 0).all(), "unreachable PLE victim"
        self.table[old_buckets] = -2
        # gfx1151 patch 53: never carry a victim's placeholder into a fill.
        self._n_invalid -= int(self._invalid[slots].sum())
        self._invalid[slots] = False
        self._bump(evictions=count, sweeps=steps)
        self._saved_size = -1  # a full cache can change keys without growing
        return slots

    def _insert_batch(self, new_keys, payloads, max_evict=4096):
        """Install payloads (array or row reader); return the number skipped."""
        keys = np.asarray(new_keys)
        if not keys.size:
            return 0
        if np.unique(keys).size != keys.size:
            raise ValueError("insert keys must be unique")
        present = self._probe(keys)
        updates = np.flatnonzero(present >= 0)
        fresh = np.flatnonzero(present < 0)
        free = self.n - self.size
        limit = min(self.n - updates.size, free + max_evict)
        fresh = fresh[max(0, fresh.size - limit):]  # same tail preference

        # Claim hash buckets BEFORE displacing anything: even a probe-bound
        # rejection must not evict a resident or leave a hole in the payload.
        buckets = self._reserve_buckets(self.table, keys[fresh])
        ok = buckets >= 0
        fresh, buckets = fresh[ok], buckets[ok]
        added = min(free, fresh.size)
        slots = np.empty(fresh.size, dtype=np.int64)
        slots[:added] = np.arange(self.size, self.size + added)
        existing = present[updates]
        if added < fresh.size:
            # gfx1151 patch 53: a restored placeholder may be refilling in
            # this same batch. Its slot is not available for replacement.
            self.visited[existing] = 2
            slots[added:] = self._evict_slots(fresh.size - added)
            self.visited[existing] = 1

        indices = np.concatenate((updates, fresh))
        if indices.size:
            targets = np.concatenate((existing, slots))
            rows = payloads(keys[indices]) if callable(payloads) else np.asarray(payloads)[indices]
            self.payload[targets] = rows
            # gfx1151 patch 53: refilling a resident lazy key makes it valid.
            self._n_invalid -= int(self._invalid[existing].sum())
            self._invalid[existing] = False
        self.keys[slots] = keys[fresh]
        self.visited[slots] = 0
        self.table[buckets] = slots
        self.size += added
        if fresh.size and np.count_nonzero(self.table == -2) > self.m // 4:
            self._rehash()
        return int(keys.size - indices.size)

    def _rehash(self):
        """Drop tombstones without moving payloads or dropping a live key."""
        rebuilt = np.full_like(self.table, -1)
        buckets = self._reserve_buckets(rebuilt, self.keys[:self.size])
        if (buckets < 0).any():
            return  # keep the reachable old index if this order exhausts probes
        rebuilt[buckets] = np.arange(self.size, dtype=rebuilt.dtype)
        self.table = rebuilt
        # gfx1151 patch 53: invalid bits remain attached to their original slots.
        self._bump(rehashes=1)

'''


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    path = root / "python/sglang/srt/utils/ple_hotset.py"
    reporter = root / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
    src = path.read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == (
        "19f478076cbca084fb81b2fc2f90e83a546911aeeaca4339d6f1190074f588c3"
    ), "expected patch 53 PLE cache"
    src = src.replace("        self.tail = 0\n", "", 1)
    src = src.replace("def _probe(self, uniq:", "def _buckets(self, uniq:", 1)
    src = src.replace("slot per key, -1 absent", "hash bucket per key, -1 absent", 1)
    src = src.replace("hit[idx] = tv[is_hit].astype(np.int64)", "hit[idx] = s[idx]", 1)
    src = src[:src.index("    # ---- eviction")] + METHODS + src[src.index("    # ---- public API"):]
    src = src.replace(
        "                    rows = fill_rows_fn(pending)\n"
        "                    self._insert_batch(pending, rows)\n"
        "                    self._bump(fills=int(pending.size))",
        "                    skipped = self._insert_batch(pending, fill_rows_fn)\n"
        "                    self._bump(fills=int(pending.size) - skipped, skipped=skipped)",
    ).replace(
        "            self._insert_batch(keys[i: i + chunk], payloads[i: i + chunk])\n"
        "            inserted += min(chunk, keys.size - i)",
        "            skipped = self._insert_batch(keys[i: i + chunk], payloads[i: i + chunk])\n"
        "            inserted += min(chunk, keys.size - i) - skipped",
    )
    log = reporter.read_text()
    old = '''f"sz {h['size']}, {h['errors']} err)"
            )'''
    assert log.count(old) == 1, "PLE reporter moved"
    reporter.write_text(log.replace(old, old + '''
            if h["skipped"]:
                hot += f", hotset skipped: {h['skipped']}"'''))
    path.write_text(src)


if __name__ == "__main__":
    main()
