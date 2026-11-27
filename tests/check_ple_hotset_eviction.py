"""Patch 61 regressions. Needs NumPy, no torch, model weights, or device."""

import ast
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SRC = Path(os.environ.get("PLE_HOTSET_SRC", "/sgl-workspace/sglang/python/sglang/srt/utils/ple_hotset.py"))
spec = importlib.util.spec_from_file_location("ple_hotset_under_test", SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Cache = module.PleHotsetCache


def payload(keys):
    # Every test row has unique bytes, including keys above 255.
    return np.asarray(keys, dtype="<i8").reshape(-1, 1).view(np.uint8)


def prepare(cache, keys):
    keys = np.asarray(keys, dtype=np.int64)
    slots = np.empty(keys.size, dtype=np.int32)
    cache.prepare(keys, slots, payload)
    hit = slots >= 0
    np.testing.assert_array_equal(cache.payload[slots[hit]], payload(keys[hit]))
    return slots


def batch(cache, keys):
    prepare(cache, keys)
    cache.flush_pending(payload)
    return cache.drain()


class HotsetRegression(unittest.TestCase):
    def consistent(self, c):
        live = c.keys[c.keys >= 0]
        self.assertEqual(c.size, len(set(live)))
        self.assertEqual(c.size, len(live))
        self.assertEqual(c.size, np.count_nonzero(c.table >= 0))
        self.assertEqual(c._n_invalid, c._invalid.sum())
        slots = c._probe(live)
        self.assertTrue((slots >= 0).all())
        np.testing.assert_array_equal(c.keys[slots], live)
        valid = ~c._invalid[slots]
        np.testing.assert_array_equal(c.payload[slots[valid]], payload(live[valid]))
        self.assertEqual(c.errors, 0)
        self.assertTrue(c.enabled)

    def test_10000_misses_at_capacity(self):
        c = Cache(16384, 8)
        batch(c, np.arange(c.n))
        incoming = c.n + np.arange(10000)
        prepare(c, incoming)
        reads = []
        c.flush_pending(lambda k: (reads.extend(k), payload(k))[1])
        d = c.drain()
        self.assertEqual((d["evictions"], d["fills"], d["skipped"], d["size"]),
                         (4096, 4096, 5904, c.n))
        np.testing.assert_array_equal(reads, incoming[-4096:])
        self.assertEqual((prepare(c, incoming) >= 0).sum(), 4096)
        self.consistent(c)

    def test_visited_row_survives_selected_victim(self):
        c = Cache(8, 8)
        batch(c, np.arange(8))
        self.assertGreaterEqual(prepare(c, [0])[0], 0)
        d = batch(c, [8])
        self.assertEqual((d["evictions"], d["fills"], d["size"]), (1, 1, 8))
        self.assertEqual((prepare(c, np.arange(9)) < 0).nonzero()[0].tolist(), [1])
        self.consistent(c)

    def test_partly_full_oversize_and_protected_refill(self):
        c = Cache(8, 8)
        c.restore_keys_only(np.arange(6))
        # Two free slots, two permissible victims. One old key refills in
        # place, even though it's at the hand and would otherwise be evicted.
        keys = np.r_[0, 100:115]
        skipped = c._insert_batch(keys, payload(keys), max_evict=2)
        self.assertEqual(skipped, 11)
        self.assertEqual(c.drain()["evictions"], 2)
        self.assertTrue((prepare(c, [0, 111, 112, 113, 114]) >= 0).all())
        self.assertEqual(c._n_invalid, 3)
        k = np.array([0, 1000])
        self.assertEqual(c._insert_batch(k, payload(k), max_evict=0), 1)
        self.assertEqual(c.drain()["evictions"], 0)
        self.consistent(c)

    def test_all_visited_wrap_and_repeated_churn(self):
        c = Cache(32, 8)
        batch(c, np.arange(32))
        rng = np.random.default_rng(613)
        for i in range(80):
            old = set(c.keys)
            # Mark all visited, or a random subset; verify every actual
            # eviction corresponds to exactly one disappeared old key.
            touch = sorted(old) if i % 4 == 0 else rng.choice(sorted(old), 17)
            prepare(c, touch)
            incoming = np.unique(rng.integers(0, 256, 29))
            expected = set(incoming) - old
            d = batch(c, incoming)
            new = set(c.keys)
            self.assertEqual(expected, new - old)
            self.assertEqual(len(old - new), len(expected))
            self.assertEqual((d["evictions"], d["fills"], d["skipped"]),
                             (len(expected), len(expected), 0))
            self.consistent(c)

    def test_probe_exhaustion_never_pays_an_eviction(self):
        c = Cache(128, 8)
        candidates = np.arange(100000)
        home = module.avalanche(candidates) & c.mask64
        collide = candidates[home == 0][:65]
        others = np.array([candidates[home == i][0] for i in range(128, 192)])
        seed = np.r_[collide[:64], others]
        batch(c, seed)
        d = batch(c, collide[64:])
        self.assertEqual((d["evictions"], d["fills"], d["skipped"]), (0, 0, 1))
        c._rehash()
        self.assertTrue((prepare(c, seed) >= 0).all())
        self.consistent(c)
        # Eager and keys-only snapshot restore must also report actual entries.
        for lazy in (False, True):
            r = Cache(128, 8)
            got = r.restore_keys_only(collide) if lazy else r.restore(collide, payload(collide))
            self.assertEqual((got, r.size, r.drain()["fills"]), (64, 64, 0 if lazy else 64))
            self.consistent(r)

    def test_full_size_snapshot_refresh(self):
        c = Cache(8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "table"
            table.write_bytes(b"test table")
            c._table_path = str(table)
            c.snapshot_path = str(Path(tmp) / "keys")
            c._shard_end = 100
            c.save_interval_s = 0
            batch(c, np.arange(8))
            batch(c, [0])
            batch(c, [8])
            c.flush_pending(payload)
            saved = module.load_hotset_snapshot(c.snapshot_path, 0, 100, str(table))
            self.assertEqual(set(saved), {0, 2, 3, 4, 5, 6, 7, 8})

    def test_journal_counts_and_interval_reset(self):
        # Run the exact reporter function without importing the GPU runtime.
        source = SRC.parents[1] / "managers/scheduler_components/metrics_reporter.py"
        fn = next(f for f in ast.parse(source.read_text()).body
                  if isinstance(f, ast.FunctionDef) and f.name == "_ple_stats_msg_suffix")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
        c = Cache(8, 8)
        batch(c, np.arange(8))
        prepare(c, np.arange(8, 28))
        c.flush_pending(payload)
        stats = dict(gathers=1, rows=20, rows_unique=20, pages=1, pages_resident=1, errors=0)
        deps = {
            "sglang.srt.utils.ple_hit_stats": SimpleNamespace(PLE_HIT_STATS=SimpleNamespace(drain=lambda: stats)),
            "sglang.srt.utils.ple_hotset": SimpleNamespace(get_ple_hotset=lambda: c),
        }
        with patch.dict(sys.modules, deps):
            msg = ns["_ple_stats_msg_suffix"]()
            self.assertIn("hotset: 0/20 (8 fills, 8 ev, sz 8, 0 err)", msg)
            self.assertIn("hotset skipped: 12", msg)
            msg = ns["_ple_stats_msg_suffix"]()
            self.assertIn("hotset: 0/0 (0 fills, 0 ev, sz 8, 0 err)", msg)
            self.assertNotIn("skipped", msg)
        self.assertEqual(c.drain(cumulative=True)["skipped"], 12)


if __name__ == "__main__":
    unittest.main()
