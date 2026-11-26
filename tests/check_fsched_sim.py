#!/usr/bin/env python3
"""Run with python3 tests/check_fsched_sim.py; stdlib, no server or GPU."""
import copy
import json
import subprocess
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fsched_replay import Cost, Request, priority, recorded
from fsched_sim import Config, Ledger, Row, fair_alloc, plan, simulate
from fsched_lag import Controller, Ready, TimeBudget, TimeLedger, compose


COST = Cost(((1, 100.),), ((1, 10.),), 5., 3.)


def run(reqs, budget=64, **kwargs):
    return simulate(reqs, "fsched", COST, Config(budget, **kwargs), seed=9, save_rounds=True)


class Allocation(unittest.TestCase):
    def test_header(self):
        self.assertEqual(priority({}), 1)
        self.assertEqual(priority({"x-SGLANG-priority": "3"}), 3)
        for bad in ("0", "-1", "1.5", "nan", " 1", "١"):
            with self.assertRaises(ValueError):
                priority({"X-SGLang-Priority": bad})

    def test_two_buckets_and_remainders(self):
        pre = [Row("a", (0,), 1, 1000)]
        gen = [Row(i, (i,), 1, 4) for i in range(1, 9)]
        for strict in (False, True):
            cfg = Config(13, strict=strict, quantum=1)
            total = Counter()
            for n in range(5):
                p, g = plan("fsched", cfg, n, pre, gen, defaultdict(Counter))
                total.update(prefill=sum(p.values()), gen=sum(g.values()))
            self.assertEqual(total, {"prefill": 52, "gen": 13})
            p, g = plan("fsched", cfg, 0, pre, gen[:0], defaultdict(Counter))
            self.assertEqual(sum(p.values()), 10 if strict else 13)
            self.assertFalse(g)
        p, g = plan("fsched", Config(20, quantum=1), 0, pre, [Row(1, (1,), 1, 1)], defaultdict(Counter))
        self.assertEqual((sum(p.values()), sum(g.values())), (19, 1))

    def test_priority_under_binding_decode_budget(self):
        # Enough sustained demand to measure service before the high weight
        # exits, not a comparison of totals after every client's finished.
        r = run([Request(str(w), 0, 0, 600, 600, w) for w in (1, 2, 3)], 2)
        counts = Counter()
        for step in r["history"][:300]:
            counts.update({k: accepted for k, (_, accepted) in step["generate"].items()})
        for w in (1, 2, 3):
            self.assertLess(abs(counts[str(w)] - sum(counts.values()) * w/6), 2)
        self.assertLessEqual(r["max_unserved"], 4)

    def test_prefix_weight_and_private_tail(self):
        r = run([Request(str(i), 0, 1088, 1, 1, i, (("p", 1024),)) for i in (1, 2, 3)]
                + [Request("other", 0, 1088, 1, 1)], 448)
        self.assertEqual(r["history"][0]["prefill"], [(["1", "2", "3"], 384), (["other"], 64)])
        self.assertEqual(r["totals"]["prefill"], 1024 + 3*64 + 1088)
        self.assertEqual(sum(c["service"] for c in r["classes"]["prefill"].values()),
                         r["totals"]["prefill"])

    def test_partial_late_hit_and_branch_identity(self):
        r = run([Request("a", 0, 225, 2, 2, prefix=(("x", 128), ("leaf", 96))),
                 Request("b", .1, 225, 1, 1, prefix=(("x", 128), ("leaf", 96))),
                 Request("c", .1, 225, 1, 1, prefix=(("y", 128), ("leaf", 96)))], 64, cap=1)
        self.assertEqual(r["totals"]["prefill"], 2*(128+96) + 3)
        self.assertIn((["a", "b"], 64), r["history"][1]["prefill"])
        self.assertGreater(r["max_queued"], 0)

    def test_acceptance_is_not_a_scheduling_oracle(self):
        reqs = [Request("full", 0, 0, 10, 10), Request("early", 0, 0, 20, 1),
                Request("tail", 0, 0, 2, 2)]
        r = run(reqs)
        batch = r["history"][0]["generate"]
        self.assertEqual(batch["early"], (4, 1))
        self.assertEqual(batch["tail"][0], 2)
        self.assertTrue(all("early" not in b["generate"] for b in r["history"][1:]))
        self.assertEqual(r["totals"]["gen_charged"], 13)
        self.assertGreater(r["totals"]["gen_reserved"], 13)
        # Refund the actual rejected slots, not the entire reservation.
        ledger = Ledger()
        gen = [Row(i, (i,), 1, 4) for i in (0, 1)]
        ledger.settle("generate", gen, {0: 4, 1: 1}, reqs)
        self.assertEqual(ledger.debt["generate"], {0: -1.5, 1: 1.5})
        self.assertEqual(fair_alloc(gen, 2, ledger.debt["generate"]), {1: 2})

    def test_batch_boundaries_and_closed_loop(self):
        reqs = [Request("first", 0, 72, 10, 1), Request("prob", 99, 128, 0, 0, cached=64,
                                                         after_previous=.25)]
        r = run(reqs)
        self.assertEqual(r["requests"][1]["arrival"], r["requests"][0]["completion"] + .25)
        self.assertIsNone(r["requests"][1]["ttft"])
        self.assertEqual(r["totals"]["first_tokens_from_prefill"], 1)
        self.assertEqual(r["totals"]["prefill"], 136)
        self.assertFalse(r["totals"]["gen_reserved"])
        self.assertAlmostEqual(r["requests"][0]["time_running"], .72)

    def test_waiting_and_active_caps(self):
        reqs = [Request(str(i), i*.02, 0, 12, 12) for i in range(10)]
        a, b = run(reqs, 4, cap=3), run(copy.deepcopy(reqs), 4, cap=3)
        self.assertEqual(a, b)
        self.assertGreater(a["max_queued"], 0)
        for h in a["history"]:
            self.assertLessEqual(len(h["generate"]), 3)
            self.assertLessEqual(sum(x[0] for x in h["generate"].values()), 4)
        self.assertEqual(a["totals"]["gen_charged"], 120)

    def test_current_mixed_width_and_hypothetical(self):
        rs = [Request("p", 0, 500, 1, 1), Request("g", 0, 0, 20, 20)]
        for policy, width in (("baseline", 1), ("fsched", 4)):
            r = simulate(rs, policy, COST, Config(64), save_rounds=True)
            self.assertEqual(r["history"][0]["generate"]["g"][0], width)
        self.assertEqual(run(rs, mixed_width=1)["history"][0]["generate"]["g"], (1, 1))


class Replay(unittest.TestCase):
    def test_inputs_and_actual_cli(self):
        costs, traces = recorded()
        self.assertEqual(set(costs), {2048, 4096})
        self.assertEqual(len(traces), 50)
        self.assertEqual(sum(len(rs) for _, _, rs in traces), 398)
        for _, name, rs in traces:
            if name == "quality":
                self.assertEqual(sum(r.requested == 0 for r in rs), 4)
                self.assertEqual(sum(r.requested == 64 for r in rs), 51)
        raw = subprocess.check_output([sys.executable, str(ROOT / "tools/fsched_sim.py"),
                                       "--workload", "synthetic", "--count", "6", "--details",
                                       "--arrival-rate", "0", "--priorities", "1,3"])
        data = json.loads(raw)
        self.assertEqual(len(data["fsched"]["requests"]), 6)
        self.assertEqual(data["fsched"]["totals"]["prefill"], data["baseline"]["totals"]["prefill"])


class ServiceTime(unittest.TestCase):
    def test_measured_charge_once_and_priority_lag(self):
        rs = [Request(str(i), 0, 100, 10, 1, w) for i, w in enumerate((1, 2, 3, .001, 1))]
        rd = Ready([Row("shared", (0, 1, 2), 6, 64)], [Row(i, (i,), rs[i].weight, 4) for i in (3, 4)], rs, 80)
        l = TimeLedger(TimeBudget())
        # Prediction = .64 + .20 seconds; measured forward twice as long.
        l.settle(rd, {"shared": 64}, {4: 1}, COST, 1, 1.68, False)
        for i, w in enumerate((1, 2, 3)):
            self.assertAlmostEqual(l.actual[i], 1.28 * w / 6)
            self.assertAlmostEqual(l.ideal[i], 1.68 * .8 * w / 6)
        self.assertAlmostEqual(l.actual[4], .4)
        self.assertAlmostEqual(sum(l.actual.values()), 1.68)
        self.assertAlmostEqual(l.ideal[3], .001 * l.ideal[4])
        self.assertAlmostEqual(l.since[3], l.ideal[3])
        self.assertEqual(l.since[4], 0)

    def test_arrival_during_nonpreemptible_forward(self):
        rs = [Request("p", 0, 200, 0, 0), Request("g", .25, 0, 50, 20)]
        rd = Ready([Row("p", (0,), 1, 200)], [], rs, 80)
        t = TimeLedger(TimeBudget())
        t.settle(rd, {"p": 100}, {}, COST, 1, 1., False, [(.25, 1, True)])
        self.assertAlmostEqual(t.ideal[1], .75*.2)
        self.assertAlmostEqual(t.actual[0], 1.)
        self.assertAlmostEqual(t.contention["seconds"], .75)
        self.assertEqual(t.skips[1], 0)  # no previously missed dispatch

    def test_pid_limits_antiwindup_and_blend(self):
        cfg = TimeBudget(1., .5, 2.)
        p = Controller(cfg)
        for loss in [1.]*500 + [0.]*500:
            old = p.budget
            p.update(loss, .5)
            self.assertLessEqual(p.budget, min(2., 1.2 * old))
            self.assertGreaterEqual(p.budget, max(.5, .8 * old))
        self.assertAlmostEqual(p.budget, 2., delta=.02)
        self.assertLess(abs(p.integral), 1.)
        p.update(1., .5)
        self.assertLess(p.budget, 1.7)
        p.reset()
        self.assertEqual((p.budget, p.integral, p.previous), (1., 0., None))
        losses, budgets = [], []
        for ratio in (0, 1, 3):
            cfg = TimeBudget(1., .5, 2., ratio)
            losses.append(cfg.blend(.5, .1))
            p = Controller(cfg)
            p.update(losses[-1], .5)
            budgets.append(p.budget)
        for x, y in zip(losses, (.1, .3, .4)):
            self.assertAlmostEqual(x, y)
        self.assertGreater(budgets[0], budgets[1])
        self.assertGreater(budgets[1], budgets[2])

    def test_generation_only_or_mixed_with_same_ready_work(self):
        reqs = [Request("p", 0, 10000, 1, 1), Request("g", 0, 0, 20, 20)]
        rd = Ready([Row("p", (0,), 1, 10000)], [Row(1, (1,), 1, 4)], reqs, 80)
        l = TimeLedger(TimeBudget(1., .7, 2.))
        # Due generation gets a full verify if flipping would be expensive;
        # the same queue can backfill on a cheap mixed path.
        l.gen_debt = .5
        l.ideal[1] = .4
        l.skips[1] = 12  # guard requires this generator in either candidate
        for rate, want_mixed in ((.1, False), (100, True)):
            c = Cost(((1, 100.), (200, 150.)), ((1, 10.),), rate, 3.)
            pa, ga = compose(rd, l, Config(mixed_width=1), c, 1)
            self.assertEqual(bool(pa), want_mixed)
            self.assertEqual(ga, {1: 1 if want_mixed else 4})

    def test_tiny_batch_priority_eventually_wins_floor(self):
        rs = [Request("fg", 0, 0, 400, 400), Request("bg", 0, 0, 16, 16, 1e-20)]
        # No two-row verify fits. Background cannot free-ride on a full batch.
        c = Cost(((64, 1000.),), ((1, 10.), (2, 3.)), 2., 3.)
        r = simulate(rs, "pid", c, Config(cap=2, mixed_width=1), save_rounds=True, timing=TimeBudget(.4, .35, .5))
        self.assertGreater(r["time"]["guard_rounds"], 0)
        self.assertEqual(r["requests"][1]["tokens_produced"], 16)
        self.assertLessEqual(r["max_unserved"], 3)
        self.assertLess(r["requests"][1]["completion"], r["requests"][0]["completion"])
        self.assertTrue(all(h["predicted_s"] <= h["budget_s"] <= .5 for h in r["history"]))

    def test_eos_billed_for_full_verify_and_idle_reset(self):
        rs = [Request("early", 0, 0, 100, 1), Request("next", 30, 0, 100, 20)]
        slow = Cost(COST.prefill_rates, ((1, 4.),), COST.mixed_short_rate, 3.)
        r = simulate(rs, "pid", COST, Config(quantum=16, mixed_width=1), save_rounds=True,
                     timing=TimeBudget(.5, .35, .6), execution_cost=slow)
        h = r["history"]
        self.assertEqual(h[0]["generate"]["early"], (4, 1))
        self.assertEqual(h[1]["budget_s"], .5)
        self.assertNotEqual(h[0]["next_budget_s"], .5)
        self.assertAlmostEqual(h[0]["predicted_s"], .3)
        self.assertAlmostEqual(h[0]["seconds"], .75)  # cap on prediction, not reality
        self.assertAlmostEqual(r["time"]["billed_s"], sum(x["seconds"] for x in h))

    def test_real_comparison_cli(self):
        raw = subprocess.check_output([sys.executable, str(ROOT / "tools/fsched_sim.py"),
                                       "--controller", "pid", "--budget", "2.2", "--ratio", "3",
                                       "--workload", "synthetic", "--count", "6", "--priorities", "1,0.00001",
                                       "--arrival-rate", "0", "--details"])
        r = json.loads(raw)
        self.assertFalse(r["hypothetical_mixed_verify"])
        self.assertTrue(r["time_gates"]["complete"])
        work = r["baseline"]["totals"]["prefill"]
        for k in ("baseline", "fsched", "fixed", "pid"):
            self.assertEqual(r[k]["totals"]["prefill"], work)
            self.assertAlmostEqual(r[k]["time"]["billed_s"], r[k]["totals"]["busy_seconds"])
            self.assertAlmostEqual(r[k]["time"]["entitled_s"], r[k]["totals"]["busy_seconds"])


if __name__ == "__main__":
    unittest.main()
