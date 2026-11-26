"""Three-profile switch and replay regressions. Stdlib only."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fsched_lag import Ready, TimeBudget, TimeLedger
from fsched_profiles import choose_profile
from fsched_replay import Cost, Request, recorded
from fsched_sim import Config, Row, simulate


class Profiles(unittest.TestCase):
    def setUp(self):
        self.cost = Cost(((128, 100.),), ((1, 10.),), 5., 3.)
        self.cfg = Config(budget=128, quantum=16, mixed_width=1)
        self.timing = TimeBudget(2., .5, 2.)
        self.ledger = TimeLedger(self.timing)
        self.requests = [Request("p", 0, 512, 0, 0), Request("g", 0, 0, 120, 120)]
        self.p = [Row("p", (0,), 1, 512)]
        self.g = [Row(1, (1,), 1, 4)]

    def choose(self, debt, factor=2, pre=True, gen=True):
        self.ledger.gen_debt = debt
        ready = Ready(self.p if pre else [], self.g if gen else [], self.requests, 80)
        return choose_profile(ready, self.ledger, self.cfg, self.cost, factor)

    def test_signed_debt_switch_and_forecast(self):
        p, g, d = self.choose(0.)
        self.assertEqual((p, g), ({"p": 112}, {1: 1}))
        # 1.12 prefill + .2 generation; 20% entitlement on the entire round.
        self.assertAlmostEqual(d["projected_debt_s"]["mixed"], .064)
        self.assertAlmostEqual(d["projected_debt_s"]["generation"], -.24)
        self.assertAlmostEqual(d["projected_debt_s"]["prefill"], .256)
        self.assertAlmostEqual(d["band_s"], .48)
        self.assertEqual(d["reason"], "inside_band")
        self.assertEqual(self.choose(1.)[:2], ({}, {1: 4}))
        self.assertEqual(self.choose(-1.)[:2], ({"p": 128}, {}))
        # Inside now, but mixed would exit the band; protect the full verify.
        self.assertEqual(self.choose(.45)[2]["profile"], "generation")
        self.assertEqual(self.choose(.45, 4)[2]["profile"], "mixed")

    def test_no_other_cohort_or_cached_credit(self):
        self.assertEqual(self.choose(20., gen=False)[:2], ({"p": 128}, {}))
        p, g, d = self.choose(-20., pre=False)
        self.assertEqual((p, g, d["debt_before_s"]), ({}, {1: 4}, 0.))
        self.assertEqual(self.choose(5., gen=False, pre=False), ({}, {}, {}))
        ready = Ready([], self.g, self.requests, 80)
        self.ledger.settle(ready, p, g, self.cost, 1, .7, False)
        self.assertEqual(self.ledger.gen_debt, 0.)
        self.assertEqual(self.ledger.contention["seconds"], 0.)

    def test_guard_and_actual_duration(self):
        self.ledger.skips[0] = self.cfg.cap
        p, g, d = self.choose(1.)
        self.assertIn("p", p)
        self.assertNotIn("generation", d["projected_debt_s"])
        self.assertEqual(self.ledger.guard_rounds, 1)
        ready = Ready(self.p, self.g, self.requests, 80)
        dt, _ = ready.charges(p, g, self.cost, 1)
        self.ledger.settle(ready, p, g, self.cost, 1, 2*dt, False)
        self.assertAlmostEqual(self.ledger.gen_debt, 1 + 2*(d["projected_debt_s"][d["profile"]]-1))

    def test_shared_edge_and_generation_packing(self):
        qs = [Request(str(w), 0, 512, 3, 3, w) for w in (1, 2, 3, 1, 1, 1)]
        pre = [Row("edge", (0, 1, 2), 6, 512), Row("private", (3,), 1, 512)]
        gen = [Row(4, (4,), 1, 2), Row(5, (5,), 1, 4)]
        ready = Ready(pre, gen, qs, 80)
        p, g, d = choose_profile(ready, self.ledger, self.cfg, self.cost, 2.)
        self.assertEqual(d["profile"], "mixed")
        self.assertEqual(g, {4: 1, 5: 1})
        self.assertGreater(p["edge"], p["private"])
        self.assertEqual(sum(p.values()), 112)
        _, bill = ready.charges(p, g, self.cost, 1)
        self.assertAlmostEqual(bill[2], 3*bill[0])
        self.ledger.gen_debt = 50
        self.assertEqual(choose_profile(ready, self.ledger, self.cfg, self.cost, 2.)[1], {4: 2, 5: 4})

    def test_infeasible_mixed_and_lag_order_under_tight_cap(self):
        qs = self.requests + [Request("other", 0, 0, 10, 10, .00001)]
        gen = self.g + [Row(2, (2,), .00001, 4)]
        c = Cost(((128, 100.),), ((1, 10.), (2, 3.)), .5, 3.)
        t = TimeLedger(TimeBudget(.5, .4, .5))
        t.gen_debt = 1
        t.ideal[2] = .1
        p, g, d = choose_profile(Ready(self.p, gen, qs, 80), t, self.cfg, c, 2)
        self.assertNotIn("mixed", d["projected_debt_s"])
        self.assertEqual((p, g), ({}, {2: 4}))

    def test_prefill_knee_does_not_limit_full_verifies(self):
        qs = [Request(str(i), 0, 0, 20, 20) for i in range(12)]
        ready = Ready([], [Row(i, (i,), 1, 4) for i in range(12)], qs, 80)
        c = Cost(((16, 100.),), ((1, 10.),), 5., 3.)
        p, g, _ = choose_profile(ready, self.ledger, self.cfg, c, 2.)
        self.assertEqual((p, list(g.values())), ({}, [4]*12))

    def test_lifecycle_and_switch_back_to_pure(self):
        qs = [Request("p", 0, 900, 0, 0), Request("early", .1, 0, 300, 5),
              Request("next", 50, 0, 20, 10)]
        r = simulate(qs, "profile", self.cost, self.cfg, save_rounds=True, timing=self.timing)
        self.assertEqual([q["tokens_produced"] for q in r["requests"]], [0, 5, 10])
        first = r["history"][0]
        self.assertEqual(first["profile"], "prefill")
        self.assertAlmostEqual(first["debt_after_s"], .2*(first["seconds"]-.1))
        last = r["history"][-1]
        self.assertEqual((last["profile"], last["debt_after_s"]), ("generation", 0))
        self.assertTrue(all(h["predicted_s"] <= h["budget_s"] == 2 for h in r["history"]))
        self.assertEqual(r["totals"]["prefill"], 900)

    def test_comparator_matches_previous_experiment(self):
        costs, traces = recorded()
        chunk, _, qs = next(t for t in traces if t[:2] == (2048, "p1-8-mixed"))
        r = simulate(qs, "protected", costs[chunk], Config(mixed_width=1), seed=7, timing=TimeBudget())
        self.assertAlmostEqual(r["wall"], 122.47, places=2)
        self.assertAlmostEqual(r["time"]["contended_gen_s"]/r["time"]["contended_s"], .1906, places=4)
        self.assertEqual(r["batches"], {"prefill": 21, "verify": 143, "mixed": 11})

    def test_cli(self):
        cmd = [sys.executable, str(ROOT / "tools/fsched_sim.py"), "--controller", "profile", "--workload",
               "synthetic", "--arrival-rate", "0", "--count", "6", "--details"]
        r = json.loads(subprocess.check_output(cmd))
        self.assertTrue(r["profile_metrics"]["complete"])
        self.assertFalse(r["hypothetical_mixed_verify"])
        self.assertEqual(r["band_factor"], 2.)
        self.assertTrue(all(h["budget_s"] == 2.7 for h in r["profile"]["history"]))
        for invalid in ("nan", "-1"):
            error = subprocess.run(cmd + ["--band-factor", invalid], capture_output=True)
            self.assertNotEqual(error.returncode, 0)
            self.assertIn(b"--band-factor", error.stderr)


if __name__ == "__main__":
    unittest.main()
