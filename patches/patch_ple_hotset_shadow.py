"""gfx1151 patch 46: PLE hot-set shadow cache (SIEVE, keys-only, no behavior change).

SPIKE — first iteration of the read-through cache design
(~/.factory/specs/2026-11-21-ple-read-through-cache.md). This version does
NOT accelerate anything: it runs the proposed flat-hash + SIEVE structure as
a shadow alongside the real gather path and reports what a real cache would
have hit, on the Prefill status line. It exists to measure the one number
that decides whether the kernel work (dense-read gather + GPU probe) is
worth building: the actual hit rate of real traffic.

Shadow semantics:
* Keyed on table row id (the ids the gather already computes; one row is
  160 B; a real cache entry would assemble 16 of them per context, so the
  row-level hit rate is a conservative lower bound for context-level).
* No payload storage and no table reads on fill — shadow fills are free, so
  hit-rate numbers reflect "if this cache were real", not current I/O.
* Same SIEVE policy the real cache would use: lazy promotion on hit,
  FIFO + hand-sweep eviction, so the numbers include eviction realism.

Structure (all flat arrays, single-owner = the scheduler's forward thread):

    keys[N]    int64   row id per slot (-1 = empty)
    visited[N] uint8   SIEVE reference bits
    table[M]   int32   avalanche(key) -> slot, M = 2N, linear probe
    hand, tail int     eviction hand / FIFO insertion point

Env: SGLANG_QWEN4_PLE_HOTSET_ROWS (capacity, default 1048576; 0 disables).
Recording errors disable the shadow, never the server.
"""

import argparse
import os
import sys

HOTSET_MODULE = '''"""gfx1151 patch 46: shadow SIEVE cache over PLE row ids (no behavior change)."""

import os
import threading

import numpy as np

_U64 = np.uint64
_M1 = _U64(0xff51afd7ed558ccd)
_M2 = _U64(0xc4ceb9fe1a85ec53)


def avalanche(keys: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer: destroys Zipf correlation before slotting."""
    k = keys.astype(np.uint64)
    k = k ^ (k >> _U64(33))
    k = k * _M1
    k = k ^ (k >> _U64(33))
    k = k * _M2
    return k ^ (k >> _U64(33))


class PleHotsetShadow:
    """Flat hash + SIEVE over PLE row ids. Keys only; no payload, no I/O."""

    def __init__(self, capacity: int) -> None:
        n = 1
        while n < capacity:
            n <<= 1
        self.n = n
        self.m = n * 2
        self.mask = _U64(self.m - 1)
        self.keys = np.full(n, -1, dtype=np.int64)
        self.visited = np.zeros(n, dtype=np.uint8)
        self.table = np.full(self.m, -1, dtype=np.int32)
        self.hand = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.Lock()
        self._interval = self._zero()
        self._cumulative = self._zero()
        self.errors = 0
        self.enabled = True

    @staticmethod
    def _zero():
        return {
            "gathers": 0, "rows": 0, "rows_unique": 0, "hits": 0,
            "misses": 0, "inserts": 0, "evictions": 0, "sweeps": 0,
            "skipped": 0,
        }

    # ---- probe ----------------------------------------------------------
    def _probe(self, uniq: np.ndarray):
        """Vectorized linear probe. Returns (hit_slots, unused).

        Table states: -1 empty (lookup STOPS — key absent), -2 tombstone
        (lookup CONTINUES — a deleted entry may shadow a live one further
        along the chain), else slot. hit_slots[i] = slot of uniq[i] if
        present else -1.
        """
        count = uniq.size
        h = avalanche(uniq) & self.mask
        s = h.astype(np.int64)
        hit = np.full(count, -1, dtype=np.int64)
        act = np.arange(count)
        for _ in range(64):  # matches the insert probe cap; rehash bounds chains
            if act.size == 0:
                break
            tv = self.table[s[act]]
            kact = uniq[act]
            is_hit = (tv >= 0) & (self.keys[np.maximum(tv, 0)] == kact)
            if is_hit.any():
                idx = act[is_hit]
                hit[idx] = tv[is_hit].astype(np.int64)
            # empty stops the probe; tombstone (-2) and occupied-mismatch continue
            act = act[~is_hit & (tv != -1)]
            if act.size:
                s[act] = (s[act] + 1) & np.int64(self.m - 1)
        return hit, None

    # ---- eviction -------------------------------------------------------
    def _sieve_evict(self) -> int:
        """SIEVE: from hand, clear visited bits, evict first unvisited.

        Returns (slot, steps). Bounded by the caller: if the sweep would run
        away (pathological all-visited), the caller bulk-clears instead.
        """
        steps = 0
        while self.visited[self.hand]:
            self.visited[self.hand] = 0
            self.hand = (self.hand + 1) % self.n
            steps += 1
            if steps >= self.n:  # all visited: clear everything, take hand
                self.visited[:] = 0
                break
        slot = self.hand
        self.hand = (self.hand + 1) % self.n
        self._table_remove(slot)
        self.keys[slot] = -1
        self.size -= 1
        return slot, steps + 1

    def _table_remove(self, slot: int) -> None:
        k = self.keys[slot]
        if k < 0:
            return
        s = int(avalanche(np.array([k], dtype=np.int64))[0] & self.mask)
        for _ in range(self.m):  # bounded: never spin on a missing entry
            if self.table[s] == slot:
                self.table[s] = -2  # tombstone: keeps chain entries reachable
                return
            s = (s + 1) % self.m

    def _insert_batch(self, new_keys: np.ndarray, max_evict: int = 4096) -> int:
        """Insert keys at the FIFO tail; SIEVE-evict on capacity pressure.

        Vectorized: slot allocation is an arange over the tail; table
        insertion is the same multi-round linear probe as lookup. Eviction
        work is capped per call (4096 slots) — beyond that, remaining keys
        are skipped and counted. Under a cold storm the shadow then stops
        inserting (measurement turns conservative) instead of stalling the
        scheduler thread. Returns the number skipped.
        """
        new_keys = np.unique(new_keys)  # duplicates break claim contiguity
        k = new_keys.size
        if k > self.n:
            # a batch larger than capacity: keep the tail, skip the rest
            skipped = k - self.n
            new_keys = new_keys[skipped:]
            k = new_keys.size
        free = self.n - self.size
        need_evict = max(k - free, 0)  # <= size by construction now
        skipped = 0
        if need_evict > max_evict:
            skipped = need_evict - max_evict
            new_keys = new_keys[: k - skipped]
            need_evict = max_evict
        for _ in range(need_evict):
            slot, steps = self._sieve_evict()
            self._interval["evictions"] += 1
            self._interval["sweeps"] += steps
            self._cumulative["evictions"] += 1
            self._cumulative["sweeps"] += steps
        if new_keys.size == 0:
            return skipped
        count = new_keys.size
        slots = (self.tail + np.arange(count)) % self.n
        self.tail = int(slots[-1]) + 1 if count else self.tail
        if self.tail >= self.n:
            self.tail = 0
        self.keys[slots] = new_keys
        self.visited[slots] = 0
        self.size += count
        # Vectorized table insertion: multi-round probe for reusable slots
        # (-1 empty or -2 tombstone). Keys were verified absent by _probe
        # and are deduplicated below. Claim protocol preserves the linear-
        # probing contiguity invariant (no empty gap may appear on any
        # key's probe path): per round, EXACTLY ONE claimant per contested
        # slot — its first occurrence — takes it if reusable; everyone else
        # advances one slot. Contention peels off one claimant per round.
        h = (avalanche(new_keys) & self.mask).astype(np.int64)
        s = h.copy()
        remaining = np.arange(count)
        for _ in range(64):
            if remaining.size == 0:
                break
            sv = s[remaining]
            tv = self.table[sv]
            reusable = (tv == -1) | (tv == -2)
            _, first_pos = np.unique(sv, return_index=True)
            claim = np.zeros(sv.size, dtype=bool)
            claim[first_pos[reusable[first_pos]]] = True
            done = remaining[claim]
            self.table[s[done]] = slots[done]
            remaining = remaining[~claim]
            if remaining.size:
                s[remaining] = (s[remaining] + 1) % self.m
        if remaining.size:
            # probe rounds exhausted: these slots hold keys with no table
            # entry — unfindable zombies. Remove them and count as skipped.
            orphan_slots = slots[remaining]
            self.keys[orphan_slots] = -1
            self.size -= int(remaining.size)
            skipped += int(remaining.size)
        if int((self.table == -2).sum()) > self.m // 4:
            self._rehash()
        return skipped

    def _rehash(self) -> None:
        """Rebuild the table from live keys, dropping tombstones/dupes.

        Tombstones accumulate under steady evict+insert traffic and inflate
        the effective lookup load towards 1.0 (chains stretch, lookups hit
        their round cap). A full vectorized rebuild restores clean chains.
        Deviation note: SIEVE visited bits reset (recency order is not
        preserved) — acceptable for the shadow; the real cache will carry
        the epoch through the rebuild instead.
        """
        live = np.unique(self.keys[self.keys >= 0])
        self.table[:] = -1
        self.keys[:] = -1
        self.visited[:] = 0
        self.size = 0
        self.hand = 0
        self.tail = 0
        self._interval["rehashes"] = self._interval.get("rehashes", 0) + 1
        self._cumulative["rehashes"] = self._cumulative.get("rehashes", 0) + 1
        if live.size:
            self._insert_batch(live)

    # ---- batch entry point ---------------------------------------------
    def observe(self, uniq: np.ndarray) -> None:
        """Record one gather's unique row ids (shadow: keys only)."""
        if not self.enabled or uniq.size == 0:
            return
        with self._lock:
            try:
                uniq = np.unique(uniq)
                hit, _unused = self._probe(uniq)
                hits = int((hit >= 0).sum())
                self.visited[hit[hit >= 0]] = 1
                misses = uniq.size - hits
                skipped = self._insert_batch(uniq[hit < 0])
                rows = int(uniq.size)
                for dst in (self._interval, self._cumulative):
                    dst["gathers"] += 1
                    dst["rows"] += rows
                    dst["rows_unique"] += rows
                    dst["hits"] += hits
                    dst["misses"] += misses - skipped
                    dst["inserts"] += rows - hits - skipped
                    dst["skipped"] = dst.get("skipped", 0) + skipped
            except Exception:
                self.enabled = False
                self.errors += 1

    def drain(self, cumulative: bool = False):
        with self._lock:
            if cumulative:
                d = dict(self._cumulative)
                d["errors"] = self.errors
                d["enabled"] = self.enabled
                d["size"] = self.size
                return d
            out = dict(self._interval)
            self._interval = self._zero()
            out["errors"] = self.errors
            out["enabled"] = self.enabled
            out["size"] = self.size
            return out


_PLE_HOTSET = None


def get_ple_hotset():
    """Process-wide shadow instance, sized from the env (0 = disabled)."""
    global _PLE_HOTSET
    if _PLE_HOTSET is None:
        rows = int(os.environ.get("SGLANG_QWEN4_PLE_HOTSET_ROWS", "1048576"))
        _PLE_HOTSET = PleHotsetShadow(rows) if rows > 0 else None
    return _PLE_HOTSET
'''

QWEN_HOOK_ANCHOR = """        stats.record(rows, int(uniq.numel()), len(pages), resident)
    except Exception:
        stats.enabled = False
        stats.note_error()"""

QWEN_HOOK_NEW = """        stats.record(rows, int(uniq.numel()), len(pages), resident)
        try:
            hotset = _ple_hotset()
            if hotset is not None:
                hotset.observe(uniq.numpy())
        except Exception:
            pass  # shadow must never affect serving or the stats
    except Exception:
        stats.enabled = False
        stats.note_error()"""

QWEN_IMPORT_ANCHOR = """from sglang.srt.utils.ple_hit_stats import (
    PLE_HIT_STATS,
    mincore_resident,
    pages_for_rows as _ple_pages_for_rows,
)"""

QWEN_IMPORT_NEW = QWEN_IMPORT_ANCHOR + """
from sglang.srt.utils.ple_hotset import get_ple_hotset as _ple_hotset"""

REPORTER_SUFFIX_ANCHOR = "        msg += _ple_stats_msg_suffix()"

REPORTER_SUFFIX_NEW = """        msg += _ple_stats_msg_suffix()
        msg += _ple_hotset_msg_suffix()"""

HOTSET_SUFFIX_BLOCK = '''def _ple_hotset_msg_suffix() -> str:
    """gfx1151 patch 46: shadow hot-set counters on the status line."""
    try:
        from sglang.srt.utils.ple_hotset import get_ple_hotset

        hs = get_ple_hotset()
        if hs is None:
            return ""
        s = hs.drain()
        if not s["gathers"]:
            return ""
        rate = s["hits"] / max(s["rows"], 1)
        return (
            f", hotset: {s['hits']}/{s['rows']} rows {rate * 100:.0f}% "
            f"sz {s['size']} ev {s['evictions']} err {s['errors']}"
        )
    except Exception:
        return ""
'''

REPORTER_MODULE_ANCHOR = "def _ple_stats_msg_suffix() -> str:"



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang",
                    help="sglang source tree root (contains python/)")
    args = ap.parse_args()

    root = os.path.join(args.target, "python", "sglang", "srt")
    if not os.path.isdir(root):
        sys.exit(f"not an sglang tree: {args.target}")

    stats_path = os.path.join(root, "utils", "ple_hotset.py")
    assert not os.path.exists(stats_path), (
        f"ple_hotset.py already exists at {stats_path}; refusing to clobber"
    )
    with open(stats_path, "w") as f:
        f.write(HOTSET_MODULE)

    qwen = os.path.join(root, "models", "qwen4_exp.py")
    with open(qwen) as f:
        text = f.read()
    assert QWEN_HOOK_ANCHOR in text, "qwen4_exp.py hook anchor moved"
    assert QWEN_IMPORT_ANCHOR in text, "qwen4_exp.py import anchor moved"
    assert "ple_hotset" not in text, "patch 46 already applied?"
    text = text.replace(QWEN_IMPORT_ANCHOR, QWEN_IMPORT_NEW, 1)
    text = text.replace(QWEN_HOOK_ANCHOR, QWEN_HOOK_NEW, 1)
    with open(qwen, "w") as f:
        f.write(text)

    rep = os.path.join(
        root, "managers", "scheduler_components", "metrics_reporter.py")
    with open(rep) as f:
        rtext = f.read()
    assert REPORTER_MODULE_ANCHOR in rtext, "reporter module anchor moved"
    assert REPORTER_SUFFIX_ANCHOR in rtext, "reporter suffix anchor moved"
    assert "ple_hotset_msg_suffix" not in rtext, "patch 46 already applied?"
    # block goes at module level (next to the patch-45 suffix def); the
    # call goes on the prefill message. Never insert a def mid-method.
    rtext = rtext.replace(
        REPORTER_MODULE_ANCHOR,
        HOTSET_SUFFIX_BLOCK.rstrip() + "\n\n\n" + REPORTER_MODULE_ANCHOR,
        1,
    )
    rtext = rtext.replace(REPORTER_SUFFIX_ANCHOR, REPORTER_SUFFIX_NEW, 1)
    with open(rep, "w") as f:
        f.write(rtext)

    print("patch 46 applied: PLE hot-set shadow cache (measurement only)")


if __name__ == "__main__":
    main()
