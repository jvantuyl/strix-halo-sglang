"""CPU checks for patch 57. In-image, test the installed module."""
import math
import sys
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

if (Path(__file__).resolve().parents[1] / "patches/fsched/runtime.py").is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches/fsched"))
    import runtime as f
else:
    from sglang.srt.managers import fsched as f


class Req:
    def __init__(self, ids, weight=1., cached=0, salt=None):
        self.origin_input_ids = array("q", ids)
        self.output_ids = []
        self.extra_key = None
        self.cache_salt = salt
        self.fsched_weight = weight
        self.num_matched_prefix_tokens = cached
        self.prefix_indices = range(cached)

    def finished(self):
        return False


def batch(p, g):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: bool(p)),
                           reqs=[*p, *g], decoding_reqs=g, prefix_lens=[0]*len(p),
                           extend_lens=[*p.values(), *([1]*len(g))])


class SchedulerTest(unittest.TestCase):
    def test_header(self):
        self.assertEqual(f.header_weight([]), 1.)
        self.assertEqual(f.header_weight([".001"]), .001)
        for bad in (["0"], ["nan"], ["inf"], ["a"], ["2", "1"]):
            with self.assertRaises(ValueError):
                f.header_weight(bad)
        self.assertEqual(sum(f.shares({1:1e308, 2:1e308}).values()), 1.)

    def test_cap_and_cost(self):
        for n in range(13):
            p = f.position_limit(n)
            self.assertLessEqual(p+n, 2048)
            self.assertLessEqual(sum(f.cost(p,n)), 2.7)
            self.assertTrue(p+n == 2048 or sum(f.cost(p+1,n)) > 2.7)
        self.assertAlmostEqual(f.cost(0,4)[1], 3.1219512/12.45013)

    def test_three_modes_and_guard(self):
        p, g = Req(range(244)), Req([777])
        r = f.Round([p], None, [g], 11, 64, 64)
        self.assertEqual([r.choose(d)[0] for d in (-1,0,1)], ["prefill","mixed","generation"])
        f.account(p).skips = 12
        self.assertNotEqual(f.Round([p], None, [g], 11, 64, 64).choose(1)[0], "generation")

    def test_shared_physical_page_once(self):
        p = Req(range(192), 1)
        q = Req([*range(128), *([8]*70)], 2)
        t = Req([*range(64), *([9]*100)], 3)
        private = Req(range(192), 1, salt="different")
        r = f.Round([p,q,t,private], None, [], 1, 64, 64)
        self.assertEqual(list(r.edges), [p])
        self.assertEqual(len(r.rate), 3)
        e = r.edges[p]
        b = e.split(0, 160)
        self.assertAlmostEqual(sum(b.values()), 160)
        self.assertAlmostEqual(b[f.account(t)], 32)
        self.assertAlmostEqual(b[f.account(q)], 64*2/6 + 64*2/3)
        self.assertNotIn(f.account(private), b)

    def test_single_partial_and_lag(self):
        a = Req([1]*3000)
        b = Req([2]*3000)
        c = Req([3]*128)
        f.account(b).lag = 1.
        r = f.Round([a,b,c], None, [], 12, 64, 64)
        self.assertIn(b, r.pack(0))
        self.assertNotIn(a, r.pack(0))
        f.account(c).lag = 5.
        r = f.Round([b,c], a, [], 12, 64, 64)
        self.assertEqual(r.pack(0)[c], 128)
        self.assertNotIn(b, r.pack(0))
        self.assertGreater(r.pack(0)[a], 0)
        self.assertLessEqual(sum(r.pack(0).values()), 2048)

    def test_event_corrections_with_overlap_and_idle(self):
        p, g = Req(range(256)), Req([9])
        s = f.FairScheduler(64,64)
        s.plan([p], None, [g], 1)
        e = s.snapshot(batch({p:128}, []))
        self.assertAlmostEqual(s.debt, e[0]/3)
        s.snapshot(batch({}, [g]))
        s.apply(e, .9-e[0], True)
        self.assertEqual(f.account(g).since, 0)
        self.assertAlmostEqual(f.account(p).lag, -.9/3 + f.cost(0,1)[1]*2/3)
        s.begin()
        s.snapshot(batch({}, [g]))
        s.apply(e, .1, True)
        self.assertEqual(s.debt, 0)
        self.assertTrue(math.isfinite(f.account(p).lag))

    def test_actual_not_planned(self):
        p, g = Req(range(244)), Req([9])
        s = f.FairScheduler(64,64)
        self.assertEqual(s.plan([p], None, [g], 1)[0], "mixed")
        e = s.snapshot(batch({}, [g]))  # native allocator denied prefill
        self.assertEqual(e[-1][:2], ("mixed","generation"))
        self.assertAlmostEqual(s.debt, -2/3*f.cost(0,1)[1])


if __name__ == "__main__":
    unittest.main()
