"""gfx1151 patch 46: PLE read-through hot-set cache (MVP, RAM-resident).

Iteration 2 of the read-through cache design
(~/.factory/specs/2026-11-21-ple-read-through-cache.md): the shadow becomes
real. Hits are served from a dense, mlock'd RAM array; misses fall back to
today's mmap row reads inside the gather kernel; fills happen on the
scheduler thread at the NEXT chunk's prep (one-chunk lag, off the GPU path,
vectorized). No persistence yet — every restart starts cold, per the MVP
brief.

Per prefill gather (non-captured, >= 512 rows):
  1. ids come to CPU once (the sync patch 45 already pays)
  2. numpy probe of the unique LOCAL row ids -> slot per row, -1 on miss
  3. the PREVIOUS batch's misses are filled: their 160 B rows are copied
     from the mmap into the payload array (pages already resident — the
     previous chunk's kernel just read them), then SIEVE-inserted
  4. the slots tensor (host int32) goes straight into the gather kernel —
     the UMA kernel reads host tensors directly, zero copies
  5. per row: slot >= 0 -> one dense 160 B read from the cache array;
     slot < 0 -> today's mmap row read

Decode CUDA-graph replay never re-enters here; capture uses a permanent
all-miss buffer so graphs always take the mmap path.

Structure: keys[N] int64, visited[N] uint8 (SIEVE bits), payload[N,160]
uint8 (mlock'd), table[M] int32 (linear probe, tombstones, vectorized
rehash at tombstone pressure). Single owner: the scheduler forward thread.

Env: SGLANG_QWEN4_PLE_HOTSET_ROWS (capacity in rows; default 1048576 ≈
152 MiB payload; 0 disables).
"""

import argparse
import os
import sys

HOTSET_MODULE = '''"""gfx1151 patch 46: PLE read-through hot-set cache (RAM MVP)."""

import ctypes
import os
import threading

import numpy as np

_U64 = np.uint64
_M1 = _U64(0xff51afd7ed558ccd)
_M2 = _U64(0xc4ceb9fe1a85ec53)

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def avalanche(keys: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer: destroys Zipf correlation before slotting."""
    k = keys.astype(np.uint64)
    k = k ^ (k >> _U64(33))
    k = k * _M1
    k = k ^ (k >> _U64(33))
    k = k * _M2
    return k ^ (k >> _U64(33))


class PleHotsetCache:
    """Flat hash + SIEVE over PLE row ids with a dense mlock'd payload."""

    def __init__(self, capacity: int, row_bytes: int) -> None:
        n = 1
        while n < capacity:
            n <<= 1
        self.n = n
        self.m = n * 2
        self.mask64 = _U64(self.m - 1)
        self.row_bytes = int(row_bytes)
        self.keys = np.full(n, -1, dtype=np.int64)
        self.visited = np.zeros(n, dtype=np.uint8)
        self.payload = np.zeros((n, self.row_bytes), dtype=np.uint8)
        self.table = np.full(self.m, -1, dtype=np.int32)
        self.hand = 0
        self.tail = 0
        self.size = 0
        self.mlocked = False
        self._lock = threading.Lock()
        self._interval = self._zero()
        self._cumulative = self._zero()
        self._pending_misses = np.zeros(0, dtype=np.int64)
        self.errors = 0
        self.enabled = True
        self._mlock()

    def _mlock(self) -> None:
        try:
            addr = self.payload.__array_interface__["data"][0]
            page = 4096
            start = addr & ~(page - 1)
            end = (addr + self.payload.nbytes + page - 1) & ~(page - 1)
            if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(end - start)) != 0:
                raise OSError(ctypes.get_errno(), "mlock failed")
            self.mlocked = True
        except Exception:
            self.mlocked = False  # cache works unlocked; pressure may evict

    @staticmethod
    def _zero():
        return {
            "gathers": 0, "rows": 0, "rows_unique": 0, "hits": 0,
            "misses": 0, "fills": 0, "evictions": 0, "sweeps": 0,
            "skipped": 0, "rehashes": 0,
        }

    def _bump(self, **kw) -> None:
        for dst in (self._interval, self._cumulative):
            for k, v in kw.items():
                dst[k] = dst.get(k, 0) + v

    # ---- probe ----------------------------------------------------------
    def _probe(self, uniq: np.ndarray) -> np.ndarray:
        """Vectorized linear probe; slot per key, -1 absent.

        Table states: -1 empty (lookup STOPS), -2 tombstone (lookup
        CONTINUES), else slot.
        """
        count = uniq.size
        h = avalanche(uniq) & self.mask64
        s = h.astype(np.int64)
        hit = np.full(count, -1, dtype=np.int64)
        act = np.arange(count)
        for _ in range(64):
            if act.size == 0:
                break
            tv = self.table[s[act]]
            kact = uniq[act]
            is_hit = (tv >= 0) & (self.keys[np.maximum(tv, 0)] == kact)
            if is_hit.any():
                idx = act[is_hit]
                hit[idx] = tv[is_hit].astype(np.int64)
            act = act[~is_hit & (tv != -1)]
            if act.size:
                s[act] = (s[act] + 1) & np.int64(self.m - 1)
        return hit

    # ---- eviction -------------------------------------------------------
    def _sieve_evict(self):
        """SIEVE: from hand, clear visited bits, evict first unvisited."""
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
        s = int(avalanche(np.array([k], dtype=np.int64))[0] & self.mask64)
        for _ in range(self.m):  # bounded: never spin on a missing entry
            if self.table[s] == slot:
                self.table[s] = -2  # tombstone keeps chain entries reachable
                return
            s = (s + 1) % self.m

    def _insert_batch(self, new_keys: np.ndarray, payloads: np.ndarray,
                      max_evict: int = 4096) -> int:
        """Insert keys+payloads at the FIFO tail; SIEVE-evict on pressure.

        Claim protocol preserves linear-probe contiguity: per round exactly
        one claimant per contested slot (first occurrence) takes it if
        reusable; the rest advance one slot. Probe-exhausted keys are
        removed and counted as skipped. Returns the number skipped.
        """
        new_keys = np.asarray(new_keys)
        payloads = np.asarray(payloads)
        if new_keys.size and np.unique(new_keys).size != new_keys.size:
            # duplicates break claim contiguity; callers dedupe (prepare,
            # rehash). Fail loudly rather than corrupt the table silently.
            raise ValueError("insert keys must be unique")
        # Keys already resident (possible when two batches miss the same key
        # before its fill lands): update the payload in place, drop them
        # from the fresh insert. A blind re-insert would leave a stale
        # duplicate entry that shadows the fresh slot on probe.
        if new_keys.size:
            present = self._probe(new_keys)
            upd = present >= 0
            if upd.any():
                self.payload[present[upd]] = payloads[upd]
                keep = ~upd
                new_keys = new_keys[keep]
                payloads = payloads[keep]
        k = new_keys.size
        if k > self.n:
            skipped = k - self.n
            new_keys = new_keys[skipped:]
            payloads = payloads[skipped:]
            k = new_keys.size
        free = self.n - self.size
        need_evict = max(k - free, 0)  # <= size by construction
        skipped = 0
        if need_evict > max_evict:
            skipped = need_evict - max_evict
            new_keys = new_keys[skipped:]
            payloads = payloads[skipped:]
            k = new_keys.size
        for _ in range(need_evict):
            slot, steps = self._sieve_evict()
            self._bump(evictions=1, sweeps=steps)
        if k == 0:
            return skipped
        slots = (self.tail + np.arange(k)) % self.n
        self.tail = int(slots[-1]) + 1
        if self.tail >= self.n:
            self.tail = 0
        self.keys[slots] = new_keys
        self.visited[slots] = 0
        self.payload[slots] = payloads
        self.size += k
        h = (avalanche(new_keys) & self.mask64).astype(np.int64)
        s = h.copy()
        remaining = np.arange(k)
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
            self.table[s[done]] = slots[done].astype(np.int32)
            remaining = remaining[~claim]
            if remaining.size:
                s[remaining] = (s[remaining] + 1) % self.m
        if remaining.size:
            # probe rounds exhausted: keys would be unfindable zombies —
            # remove their slots and count as skipped
            orphan_slots = slots[remaining]
            self.keys[orphan_slots] = -1
            self.size -= int(remaining.size)
            skipped += int(remaining.size)
        if int((self.table == -2).sum()) > self.m // 4:
            self._rehash()
        return skipped

    def _rehash(self) -> None:
        """Rebuild table from live keys; drops tombstones and dupes.

        Duplicate slots (from earlier false-miss re-inserts) hold identical
        payloads for the same key — the first slot's payload is kept. SIEVE
        bits reset (recency not preserved) — documented deviation.
        """
        live, first_idx = np.unique(self.keys, return_index=True)
        live = live[live >= 0]
        first_idx = first_idx[self.keys[first_idx] >= 0]
        payloads = self.payload[first_idx]
        self.table[:] = -1
        self.keys[:] = -1
        self.visited[:] = 0
        self.size = 0
        self.hand = 0
        self.tail = 0
        self._bump(rehashes=1)
        if live.size:
            self._insert_batch(live, payloads)

    # ---- public API -----------------------------------------------------
    def prepare(self, local_ids: np.ndarray, slots_out: np.ndarray,
                fill_rows_fn) -> None:
        """Probe this batch, write slot indices, fill the previous misses.

        local_ids: LOCAL row ids for every gather position (out-of-window
        positions arrive as -1 and keep slot -1). slots_out: int32 array of
        the same length. fill_rows_fn(rows) -> uint8 (count, row_bytes)
        gathered view of the mmap for fills.

        Fills serve the PREVIOUS batch's misses (one-chunk lag): pages are
        resident by then (the previous kernel just read them), so fills are
        minor faults, not disk I/O.
        """
        if not self.enabled:
            slots_out[:] = -1
            return
        with self._lock:
            try:
                in_window = local_ids >= 0
                uniq = np.unique(local_ids[in_window])
                hit = self._probe(uniq)
                hits = int((hit >= 0).sum())
                self.visited[hit[hit >= 0]] = 1
                order = np.searchsorted(uniq, local_ids)
                flat_slots = np.where(
                    in_window, hit[np.minimum(order, max(uniq.size - 1, 0))],
                    -1,
                )
                slots_out[:] = flat_slots.astype(np.int32)
                # fill the previous batch's misses (one-chunk lag)
                pending = self._pending_misses
                self._pending_misses = uniq[hit < 0]
                if pending.size:
                    rows = fill_rows_fn(pending)
                    self._insert_batch(pending, rows)
                    self._bump(fills=int(pending.size))
                self._bump(
                    gathers=1, rows=int(local_ids.size),
                    rows_unique=int(uniq.size), hits=hits,
                    misses=int((hit < 0).sum()),
                )
            except Exception:
                self.enabled = False
                self.errors += 1
                slots_out[:] = -1

    def drain(self, cumulative: bool = False):
        with self._lock:
            d = dict(self._cumulative if cumulative else self._interval)
            if not cumulative:
                self._interval = self._zero()
            d["errors"] = self.errors
            d["enabled"] = self.enabled
            d["size"] = self.size
            d["mlocked"] = self.mlocked
            return d


_PLE_HOTSET = None


def get_ple_hotset():
    """Process-wide cache instance (constructed by the embedding module)."""
    return _PLE_HOTSET


def make_ple_hotset(capacity_rows: int, row_bytes: int):
    """Construct (or return) the process-wide cache; 0 capacity disables."""
    global _PLE_HOTSET
    if _PLE_HOTSET is None and capacity_rows > 0:
        _PLE_HOTSET = PleHotsetCache(capacity_rows, row_bytes)
    return _PLE_HOTSET
'''

# Status-line observability: extend patch 45's PLE suffix with the hot
# set's own counters (hits/misses of the real cache, fills, evictions).
REPORTER_OLD = """        return (
            f", PLE: {s['rows_unique']}/{s['rows']} rows, "
            f"{s['pages_resident']}/{s['pages']} pages res, "
            f"{s['errors']} err"
        )
    except Exception:
        return \"\""""

REPORTER_NEW = """        from sglang.srt.utils.ple_hotset import get_ple_hotset

        hs = get_ple_hotset()
        hot = ""
        if hs is not None:
            h = hs.drain()
            hot = (
                f", hotset: {h['hits']}/{h['hits'] + h['misses']} "
                f"({h['fills']} fills, {h['evictions']} ev, "
                f"sz {h['size']}, {h['errors']} err)"
            )
        return (
            f", PLE: {s['rows_unique']}/{s['rows']} rows, "
            f"{s['pages_resident']}/{s['pages']} pages res, "
            f"{s['errors']} err{hot}"
        )
    except Exception:
        return \"\""""

QWEN_IMPORT_ANCHOR = """from sglang.srt.utils.ple_hit_stats import (
    PLE_HIT_STATS,
    mincore_resident,
    pages_for_rows as _ple_pages_for_rows,
)"""

QWEN_IMPORT_NEW = QWEN_IMPORT_ANCHOR + """
from sglang.srt.utils.ple_hotset import (
    PleHotsetCache as _PleHotsetCache,
    make_ple_hotset as _make_ple_hotset,
)"""

# Production runs the CPU gather path on HIP (SGLANG_PLE_UVA_GATHER unset):
# host_ids -> index_select on the mmap'd table -> bf16 copy. The hot cache
# intercepts exactly there: hit rows come from the mlock'd dense array, miss
# rows from the mmap as today. Pure torch/numpy — no new kernels.
CPU_GATHER_OLD = """    host_ids = flat_ids.detach().to("cpu")
    in_range = (host_ids >= start) & (host_ids < end)
    local_idx = torch.where(in_range, host_ids - start, torch.zeros_like(host_ids))
    rows = embedding.weight.data.index_select(0, local_idx).to(torch.bfloat16)
    rows = torch.where(in_range.unsqueeze(1), rows, torch.zeros_like(rows))
    output.view(-1, embedding.embedding_dim).copy_(rows)"""

CPU_GATHER_NEW = """    host_ids = flat_ids.detach().to("cpu")
    in_range = (host_ids >= start) & (host_ids < end)
    local_idx = torch.where(in_range, host_ids - start, torch.zeros_like(host_ids))
    cache = embedding._hotset
    if (
        cache is not None
        and cache.enabled
        and host_ids.numel() >= 512
        and host_ids.numel() <= embedding._hotset_slots.shape[0]
        and not torch.cuda.is_current_stream_capturing()
    ):
        # gfx1151 patch 46: read-through hot-set cache. Probe on CPU; hit
        # rows are served from the mlock'd payload array, miss rows still
        # go through the mmap (and populate the cache one chunk later).
        n = host_ids.numel()
        slots_np = embedding._hotset_slots[:n]
        cache.prepare(
            np.where(
                in_range.numpy(), local_idx.numpy(), -1
            ),
            slots_np,
            lambda hit_rows: embedding.weight.view(torch.uint8)[hit_rows].numpy(),
        )
        slots_t = torch.from_numpy(slots_np)
        hit = slots_t >= 0
        hit_pos = torch.nonzero(hit).squeeze(1)
        miss_pos = torch.nonzero(~hit).squeeze(1)
        rows = torch.empty(
            n, embedding.embedding_dim, dtype=torch.bfloat16
        )
        if hit_pos.numel():
            hit_bytes = cache.payload[slots_np[hit_pos.numpy()]]
            rows[hit_pos] = (
                torch.from_numpy(hit_bytes)
                .view(embedding.weight.dtype)
                .to(torch.bfloat16)
            )
        if miss_pos.numel():
            rows[miss_pos] = embedding.weight.data.index_select(
                0, local_idx[miss_pos]
            ).to(torch.bfloat16)
        output.view(-1, embedding.embedding_dim).copy_(rows)
        return
    rows = embedding.weight.data.index_select(0, local_idx).to(torch.bfloat16)
    rows = torch.where(in_range.unsqueeze(1), rows, torch.zeros_like(rows))
    output.view(-1, embedding.embedding_dim).copy_(rows)"""

HOTSET_ATTACH_OLD = """        # Only the file backend has anything to prefetch (rows live on storage).
        self._file_prefetcher = make_ple_file_prefetcher(host_table)"""

HOTSET_ATTACH_NEW = """        # gfx1151 patch 46: read-through hot-set cache (RAM MVP). Only the
        # file backend needs it (the pinned backend has no fault path).
        import os as _os

        _hot_rows = int(_os.environ.get("SGLANG_QWEN4_PLE_HOTSET_ROWS", "1048576"))
        self._hotset = (
            _make_ple_hotset(
                _hot_rows, host_table.element_size() * host_table.shape[-1]
            )
            if _hot_rows > 0
            else None
        )
        self._hotset_slots = np.zeros(1 << 18, dtype=np.int32)
        # Only the file backend has anything to prefetch (rows live on storage).
        self._file_prefetcher = make_ple_file_prefetcher(host_table)"""

NP_IMPORT_OLD = "import torch"
NP_IMPORT_NEW = "import numpy as np\nimport torch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang",
                    help="sglang source tree root (contains python/)")
    args = ap.parse_args()

    root = os.path.join(args.target, "python", "sglang", "srt")
    if not os.path.isdir(root):
        sys.exit(f"not an sglang tree: {args.target}")

    stats_path = os.path.join(root, "utils", "ple_hotset.py")
    with open(stats_path, "w") as f:
        f.write(HOTSET_MODULE)

    qwen = os.path.join(root, "models", "qwen4_exp.py")
    with open(qwen) as f:
        text = f.read()
    # Idempotent re-apply: if a previous run of this patch is present,
    # unapply its qwen4_exp.py edits first (the module file is rewritten
    # from scratch below; the reporter edit is reversed the same way).
    if "ple_hotset" in text:
        text = text.replace(NP_IMPORT_NEW, NP_IMPORT_OLD, 1)
        text = text.replace(QWEN_IMPORT_NEW, QWEN_IMPORT_ANCHOR, 1)
        text = text.replace(HOTSET_ATTACH_NEW, HOTSET_ATTACH_OLD, 1)
        text = text.replace(CPU_GATHER_NEW, CPU_GATHER_OLD, 1)
    rep = os.path.join(root, "managers", "scheduler_components",
                       "metrics_reporter.py")
    with open(rep) as f:
        rtext = f.read()
    if REPORTER_NEW in rtext:
        rtext = rtext.replace(REPORTER_NEW, REPORTER_OLD, 1)
        with open(rep, "w") as f:
            f.write(rtext)
    for name, anchor in (
        ("import", QWEN_IMPORT_ANCHOR),
        ("cpu-gather", CPU_GATHER_OLD),
        ("attach", HOTSET_ATTACH_OLD),
    ):
        assert anchor in text, f"qwen4_exp.py {name} anchor moved"

    # numpy import: qwen4_exp.py starts `import torch` — add numpy above it.
    # Anchor conservatively on the first occurrence of a bare import torch.
    text = text.replace(NP_IMPORT_OLD, NP_IMPORT_NEW, 1)
    text = text.replace(QWEN_IMPORT_ANCHOR, QWEN_IMPORT_NEW, 1)
    text = text.replace(CPU_GATHER_OLD, CPU_GATHER_NEW, 1)
    text = text.replace(HOTSET_ATTACH_OLD, HOTSET_ATTACH_NEW, 1)
    rep = os.path.join(root, "managers", "scheduler_components", "metrics_reporter.py")
    with open(rep) as f:
        rtext = f.read()
    assert REPORTER_OLD in rtext, "metrics_reporter.py reporter anchor moved"
    rtext = rtext.replace(REPORTER_OLD, REPORTER_NEW, 1)
    with open(rep, "w") as f:
        f.write(rtext)
    with open(qwen, "w") as f:
        f.write(text)

    print("patch 46 applied: PLE hot-set cache MVP (RAM, cold on restart)")


if __name__ == "__main__":
    main()
