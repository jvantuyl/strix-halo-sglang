#!/usr/bin/env python3
"""Compare within-batch fair sharing to shortest-prefill-first, offline."""
import argparse
import heapq
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from fsched_replay import BENCH, recorded, synthetic
from fsched_lag import Ready, TimeBudget, TimeLedger, compose
from fsched_profiles import choose_profile, protected


@dataclass
class Row:
    key: object
    members: tuple  # request indices that benefit from this physical row
    weight: int
    capacity: int
    shortest: int = 0


def entitlement(rows, budget):
    """Continuous weighted max-min service, with capped shares donated."""
    left = min(budget, sum(r.capacity for r in rows))
    ordered = sorted(rows, key=lambda r: r.capacity / r.weight)
    # Removing a large weight by subtraction can erase a small batch priority.
    weights = []
    weight = 0
    for r in reversed(ordered):
        weight += r.weight
        weights.append(weight)
    result = {}
    for r, weight in zip(ordered, reversed(weights)):
        share = max(0., left) * r.weight / weight
        result[r.key] = min(share, r.capacity)
        left -= result[r.key]
    return result


def fair_alloc(rows, budget, debt, quantum=1):
    """Integer allocation nearest the earned service, carrying rounding lag."""
    target = entitlement(rows, budget)
    due = [target[r.key] + sum(debt[i] for i in r.members) for r in rows]
    slots = [min(r.capacity, max(0, math.floor(x))) for x, r in zip(due, rows)]
    diff = min(budget, sum(r.capacity for r in rows)) - sum(slots)
    while diff:
        sign = 1 if diff > 0 else -1
        eligible = (i for i, r in enumerate(rows)
                    if (slots[i] < r.capacity if sign == 1 else slots[i] > 0))
        i = max(eligible, key=lambda i: sign * (due[i] - slots[i]) / rows[i].weight)
        slots[i] += sign
        diff -= sign
    if quantum > 1:
        # Publish page-aligned chunks, except for an edge's final tail.
        slots = [n if n == r.capacity else n // quantum * quantum for r, n in zip(rows, slots)]
        left = budget - sum(slots)
        while left:
            eligible = [i for i, r in enumerate(rows)
                        if 0 < min(quantum, r.capacity - slots[i]) <= left]
            if not eligible:
                break
            i = max(eligible, key=lambda i: (due[i] - slots[i]) / rows[i].weight)
            step = min(quantum, rows[i].capacity - slots[i])
            slots[i] += step
            left -= step
    return {r.key: n for r, n in zip(rows, slots) if n}


def greedy(rows, budget):
    result = {}
    for r in rows:
        if budget <= 0:
            break
        result[r.key] = min(budget, r.capacity)
        budget -= result[r.key]
    return result


@dataclass(frozen=True)
class Config:
    budget: int = 2048
    percent: int = 80
    strict: bool = False
    cap: int = 12
    mixed_width: int = 4  # hypothetical fsched; current mixed EAGLE is width one
    window: int = 32
    quantum: int = 64

    def __post_init__(self):
        if (self.budget < 2 or not 0 < self.percent < 100 or self.cap < 1
                or self.mixed_width not in (1, 4) or self.window < 1 or self.quantum < 1):
            raise ValueError("invalid round budget, split, row cap, verify width or window")


def plan(policy, cfg, nround, pre, gen, debt):
    """No EOS, acceptance draws, wall-clock cost or future arrivals visible here."""
    b = cfg.budget
    pc = sum(r.capacity for r in pre)
    if policy == "baseline":
        # The pin's full mixed merge uses one plain decode per running row.
        # Leave room for a prefill if both queues exist at a tiny test budget.
        # In mixed mode every running row needs exactly one slot.
        ga = {r.key: 1 for r in gen[:b - 1]} if pre else greedy(gen, b)
        return greedy(sorted(pre, key=lambda r: r.shortest), b - sum(ga.values())), ga
    if policy != "fsched":
        raise ValueError("unknown scheduling policy")
    # Carry the fractional 0.8 token instead of rounding every round.
    p = (nround + 1) * b * cfg.percent // 100 - nround * b * cfg.percent // 100
    g = b - p
    gc = sum(r.capacity for r in gen)
    if not cfg.strict:
        p, g = min(pc, p + max(0, g - gc)), min(gc, g + max(0, p - pc))
    quantum = cfg.quantum if p >= cfg.quantum else 1
    pa = fair_alloc(pre, p, debt["prefill"], quantum)
    if not cfg.strict:
        g = min(gc, b - sum(pa.values()))
    return pa, fair_alloc(gen, g, debt["generate"])


@dataclass
class State:
    produced: int = 0
    admitted: bool = False
    admitted_at: float | None = None
    first: float | None = None
    finish: float | None = None
    queued_rounds: int = 0
    missed: int = 0
    max_missed: int = 0
    verifies: int = 0


@dataclass
class Ledger:
    debt: dict = field(default_factory=lambda: defaultdict(Counter))
    actual: Counter = field(default_factory=Counter)
    ideal: Counter = field(default_factory=Counter)
    win_actual: Counter = field(default_factory=Counter)
    win_ideal: Counter = field(default_factory=Counter)
    windows: list = field(default_factory=list)

    def settle(self, mode, rows, used, requests):
        # Compare to a fluid server with the same available work and row caps,
        # independent of the policy's reservations. Decompose shared physical
        # work by beneficiary weight; do not bill the whole page to each user.
        target = entitlement(rows, sum(used.values()))
        served = set()
        for r in rows:
            n = used.get(r.key, 0)
            for i in r.members:
                w = requests[i].weight
                actual, ideal = n * w / r.weight, target[r.key] * w / r.weight
                self.debt[mode][i] += ideal - actual
                self.actual[mode, w] += actual
                self.ideal[mode, w] += ideal
                self.win_actual[mode, i] += actual
                self.win_ideal[mode, i] += ideal
                if n:
                    served.add(i)
        return served

    def close_window(self):
        index = {}
        for mode in ("prefill", "generate"):
            x = [self.win_actual[k] / v for k, v in self.win_ideal.items() if k[0] == mode and v > 0]
            if x:
                index[mode] = min(1., sum(x)**2 / (len(x) * sum(a*a for a in x)))
        self.windows.append(index)
        self.win_actual.clear()
        self.win_ideal.clear()

    def classes(self):
        return {mode: {w: {"service": self.actual[m, w], "entitled": v,
                           "share": self.actual[m, w] / total,
                           "relative_error": self.actual[m, w] / v - 1}
                       for (m, w), v in self.ideal.items() if m == mode and v}
                for mode in ("prefill", "generate")
                if (total := sum(n for (m, _), n in self.actual.items() if m == mode))}


def simulate(requests, policy, cost, cfg, seed=1, save_rounds=False, accept_mean=None,
             timing=None, execution_cost=None, band_factor=2., feedback=None):
    if not requests or len({r.name for r in requests}) != len(requests):
        raise ValueError("expected a nonempty workload with unique request names")
    edges = [list(r.edges()) for r in requests]
    sizes = {}
    for path in edges:
        for key, size in path:
            if sizes.setdefault(key, size) != size:
                raise ValueError("same prefix edge has inconsistent lengths")
    state = [State() for r in requests]
    # Independent per-request streams, so reordering requests alone cannot
    # change their proposals or early stops in a paired comparison.
    rng = [random.Random(f"{seed}:{r.name}") for r in requests]
    done, seen = Counter(), Ledger()
    tdm = policy in ("fixed", "pid", "protected", "profile")
    if policy == "profile" and not (math.isfinite(band_factor) and band_factor >= 0):
        raise ValueError("profile band factor must be finite and nonnegative")
    if policy == "profile" and cfg.budget < max(4, cfg.quantum):
        raise ValueError("profile token capacity must fit a prefill quantum and a verify row")
    time = TimeLedger(timing) if timing else None
    if tdm and not time:
        raise ValueError("lag policies require a service-time budget")
    if tdm and timing.floor < max(cost.seconds(cfg.quantum, {}, 0), cost.seconds(0, {0: 4}, cost.accept)):
        raise ValueError("budget floor must fit a prefill quantum and a standalone verify row")
    release = [r.arrival for r in requests]
    arrivals = [(r.arrival, i) for i, r in enumerate(requests) if r.after_previous is None]
    if requests[0].after_previous is not None:
        raise ValueError("first request has no predecessor")
    heapq.heapify(arrivals)
    live = set()
    nround = 0
    now = arrivals[0][0]
    totals, kinds, history = Counter(), Counter(), []
    while arrivals or live:
        if not live:
            if time and arrivals[0][0] > now:
                time.pid.reset()
                time.gen_debt = 0.
            now = max(now, arrivals[0][0])
        while arrivals and arrivals[0][0] <= now:
            live.add(heapq.heappop(arrivals)[1])
        front = {i: next((key for key, n in edges[i] if done[key] < n), None) for i in live}
        remaining = {i: sum(n - done[k] for k, n in edges[i]) for i in live}
        waiting = sorted((i for i in live if not state[i].admitted),
                         key=lambda i: ((remaining[i] if policy == "baseline" else 0),
                                        release[i], i))
        slots = cfg.cap - (len(live) - len(waiting))
        for i in waiting[:slots]:
            state[i].admitted = True
            state[i].admitted_at = now
        for i in waiting[slots:]:
            state[i].queued_rounds += 1

        # A queued follower may contribute priority to an admitted owner's
        # shared prefix without consuming a row/slot. A child cannot run until
        # its parent edge is published at the previous forward's boundary.
        groups = defaultdict(list)
        gen = []
        for i in sorted(live):
            if front[i] is not None:
                groups[front[i]].append(i)
            elif state[i].admitted:
                left = requests[i].requested - state[i].produced
                if left:
                    gen.append(Row(i, (i,), requests[i].weight, min(4, left)))
                else:
                    state[i].finish = now
        pre = [Row(k, tuple(ids), sum(requests[i].weight for i in ids), sizes[k] - done[k],
                   min(remaining[i] for i in ids if state[i].admitted))
               for k, ids in groups.items() if any(state[i].admitted for i in ids)]
        if pre and not tdm:
            width = 1 if policy == "baseline" else cfg.mixed_width
            for r in gen:
                r.capacity = min(width, r.capacity)
        percent, band = cfg.percent, band_factor
        if feedback is not None:
            assert policy == "profile"
            percent, band = feedback.decide(now, pre, gen)
        if time:
            ready = Ready(pre, gen, requests, percent)
        if policy == "profile":
            pa, ga, decision = choose_profile(ready, time, cfg, cost, band)
        elif policy == "protected":
            pa, ga = protected(ready, time, cfg, cost)
        elif tdm:
            pa, ga = compose(ready, time, cfg, cost, cfg.mixed_width)
        else:
            pa, ga = plan(policy, cfg, nround, pre, gen, seen.debt)
        np, ng = sum(pa.values()), sum(ga.values())
        assert tdm or np + ng <= cfg.budget
        width = 1 if policy == "baseline" else cfg.mixed_width
        spec = not np or width == 4
        mean_accept = (cost.accept if accept_mean is None else accept_mean) if spec else 1.
        clock = execution_cost or cost
        duration = clock.seconds(np, ga, clock.accept if spec else 1.)
        late = []
        if time and len(live) < cfg.cap and arrivals and arrivals[0][0] < now + duration:
            # Settlement only, never visible to compose(). Full admission
            # queues do not create fictitious service credits.
            for born, i in sorted(arrivals)[:cfg.cap - len(live)]:
                if born >= now + duration:
                    break
                generating = all(done[k] == n for k, n in edges[i])
                if not generating or requests[i].requested:
                    late.append((born - now, i, generating))
        now += duration
        actual = {}
        for i, reserved in ga.items():
            # Draw only AFTER composing the entire batch. A rejected proposal
            # cannot create another GPU row retroactively. Return its charge
            # at the same round's ledger settlement, before the next dispatch.
            accepted = 1 + sum(rng[i].random() < (mean_accept - 1) / 3 for _ in range(3))
            actual[i] = min(reserved, accepted, requests[i].eos - state[i].produced)
            state[i].produced += actual[i]
            state[i].verifies += 1
            if state[i].first is None:
                state[i].first = now
        for k, n in pa.items():
            done[k] += n
        for i in live:
            if state[i].admitted and front[i] in pa and remaining[i] == pa[front[i]]:
                if requests[i].requested:
                    # Prefill already samples the first output, not a verify.
                    state[i].produced = 1
                    state[i].first = now
                    totals["first_tokens_from_prefill"] += 1
                else:
                    state[i].finish = now
            if state[i].produced and state[i].produced == requests[i].eos:
                state[i].finish = now

        served = seen.settle("prefill", pre, pa, requests) | seen.settle("generate", gen, actual, requests)
        if feedback is not None:
            feedback.delivered(now, served)
        if time:
            step = time.settle(ready, pa, ga, cost, width, duration, policy in ("pid", "protected"), late)
            if save_rounds and policy == "profile":
                step.update(decision, debt_after_s=time.gen_debt)
        for i in live:
            if state[i].admitted:
                state[i].missed = 0 if i in served else state[i].missed + 1
                state[i].max_missed = max(state[i].max_missed, state[i].missed)
        for i in live:
            if state[i].finish is not None and i+1 < len(requests) and requests[i+1].after_previous is not None:
                release[i+1] = state[i].finish + requests[i+1].after_previous
                heapq.heappush(arrivals, (release[i+1], i+1))
        live = {i for i in live if state[i].finish is None}
        kind = "mixed" if np and ng else "prefill" if np else "verify" if ng else "idle"
        kinds[kind] += 1
        totals.update(prefill=np, gen_reserved=ng, gen_charged=sum(actual.values()),
                      prefill_round_seconds=duration if np else 0, busy_seconds=duration,
                      prefill_rows=len(pa), generate_rows=len(ga))
        if save_rounds:
            history.append(dict(start=now-duration, seconds=duration, prefill=[
                ([requests[i].name for i in r.members], pa[r.key]) for r in pre if r.key in pa],
                generate={requests[i].name: (n, actual[i]) for i, n in ga.items()}, **(step if time else {})))
        nround += 1
        if nround % cfg.window == 0 or not (live or arrivals):
            seen.close_window()
    per_req = []
    for r, s, arrival in zip(requests, state, release):
        running = s.finish - arrival
        per_req.append(dict(name=r.name, priority=r.weight, arrival=arrival, tokens_requested=r.requested,
                            tokens_produced=s.produced, time_running=running, completion=s.finish,
                            ttft=None if s.first is None else s.first-arrival,
                            output_tps=s.produced/running if running else None,
                            admission_wait_s=s.admitted_at-arrival, queued_rounds=s.queued_rounds,
                            max_unserved_rounds=s.max_missed, verifies=s.verifies))
        if time:
            i = len(per_req)-1
            per_req[-1].update(entitled_s=time.ideal[i], charged_s=time.actual[i], worst_lag_s=time.peak_lag[i])
    return dict(rounds=nround, batches=dict(kinds), totals=dict(totals), classes=seen.classes(),
                windows=seen.windows, max_unserved=max(s.max_missed for s in state),
                max_queued=max(s.queued_rounds for s in state),
                prefill_tps=totals["prefill"]/totals["prefill_round_seconds"] if totals["prefill"] else None,
                wall=now-min(release), requests=per_req,
                **({"time": time.summary()} if time else {}),
                **({"history": history} if save_rounds else {}))


def compare(base, fair, workload):
    # Predeclared finite-trace checks. Admission and eligibility waits are
    # separate in the output; neither an empty queue nor idle time earns debt.
    worst = lambda r: min((v for w in r["windows"] for v in w.values()), default=1.)
    ratio = fair["prefill_tps"]/base["prefill_tps"] if base["prefill_tps"] else None
    error = max((abs(c["relative_error"]) for cs in fair["classes"].values()
                 for c in cs.values()), default=0.)
    return dict(starvation=all(r["tokens_produced"] == q.eos for r, q in zip(fair["requests"], workload))
                and fair["max_unserved"] < 64 and fair["max_queued"] < 64,
                class_share=error <= .10+1e-12, class_max_error=error,
                jain=worst(fair) >= worst(base)-1e-12, jain_baseline=worst(base), jain_fsched=worst(fair),
                prefill=ratio is None or abs(ratio-1) <= .05+1e-12, prefill_ratio=ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["all", "recorded", "synthetic"], default="all")
    parser.add_argument("--controller", choices=("static", "fixed", "pid", "profile"), default="static",
                        help="profile compares fixed-cap switching with PID and protected backfill")
    parser.add_argument("--budget", help="round seconds (fixed for profile, initial for PID); tokens in static mode")
    parser.add_argument("--band-factor", type=float, help="profile band in generation debt-repayment rounds (default 2)")
    parser.add_argument("--budget-floor", type=float, default=.5, metavar="SECONDS")
    parser.add_argument("--budget-cap", type=float, default=3., metavar="SECONDS")
    parser.add_argument("--ratio", type=float, default=3., help="responsiveness:fairness loss weight")
    parser.add_argument("--prefill-percent", type=int, default=int(os.getenv("FSCHED_PREFILL_PERCENT", "80")))
    parser.add_argument("--strict", action="store_true", help="do not lend unusable reservations across buckets")
    parser.add_argument("--mixed-width", type=int, choices=(1, 4),
                        help="default 1 in time-policy comparisons; 4 in the original static experiment")
    parser.add_argument("--accept-mean", type=float, help="sensitivity run: override tokens/verify (1 to 4)")
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--arrival-rate", type=float, default=2., help="requests/s, zero for one burst")
    parser.add_argument("--priorities", default="1,2,3")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--details", action="store_true", help="include request stats, windows, and batch plans")
    args = parser.parse_args()
    if args.band_factor is not None and (args.controller != "profile"
            or not math.isfinite(args.band_factor) or args.band_factor < 0):
        parser.error("--band-factor requires profile mode and a finite nonnegative value")
    band = 2. if args.band_factor is None else args.band_factor
    if args.count < 1 or not math.isfinite(args.arrival_rate) or args.arrival_rate < 0:
        parser.error("count must be positive and arrival rate finite and nonnegative")
    if args.accept_mean is not None and not 1 <= args.accept_mean <= 4:
        parser.error("accept mean must be between 1 and 4")
    token_budget = os.getenv("FSCHED_ROUND_TOKENS", "2048")
    timing = None
    policies = ["baseline", "fsched"]
    try:
        if args.controller != "static":
            timing = TimeBudget(float(args.budget or 2.7), args.budget_floor, args.budget_cap, args.ratio)
            if args.controller == "profile":
                policies = ["baseline", "pid", "protected", "profile"]
            else:
                policies += ["fixed", "pid"] if args.controller == "pid" else ["fixed"]
        else:
            token_budget = args.budget or token_budget
        cfg = Config(int(token_budget), args.prefill_percent, args.strict,
                     mixed_width=args.mixed_width or (1 if timing else 4))
    except ValueError as e:
        parser.error(str(e))
    if timing and (args.strict or args.prefill_percent != 80):
        parser.error("time comparisons use work-conserving 80/20 entitlement")
    costs, traces = recorded(BENCH)
    if args.workload == "synthetic":
        traces = []
    if args.workload != "recorded":
        traces.append((2048, "synthetic", synthetic(args.seed, args.count, args.arrival_rate,
                                                   args.priorities.split(","), fractional=bool(timing))))
        if timing and args.workload == "all" and args.arrival_rate:
            traces.append((2048, "burst", synthetic(args.seed, args.count, 0, args.priorities.split(","),
                                                     fractional=True)))
    for chunk, name, requests in traces:
        cost = costs[chunk]
        result = {policy: simulate(requests, policy, cost, cfg, args.seed, args.details, args.accept_mean, timing,
                                   band_factor=band)
                  for policy in policies}
        extra = {}
        if "fsched" in result:
            extra["gates"] = compare(result["baseline"], result["fsched"], requests)
        if timing:
            extra.update(time_budget=asdict(timing), hypothetical_mixed_verify=cfg.mixed_width == 4)
            if "pid" in result and "fixed" in result:
                t = result["pid"]["time"]
                ratio = result["fixed"]["wall"] / result["pid"]["wall"] if result["pid"]["wall"] else 1.
                extra["time_gates"] = dict(
                    complete=all(x["tokens_produced"] == q.eos for r in result.values()
                                 for x, q in zip(r["requests"], requests, strict=True)),
                    lag_floor=max(t["worst_lag_s"], t["worst_since_service_s"]) <= timing.floor + 1e-9,
                    bounded=timing.floor <= t["min_budget_s"] <= t["max_budget_s"] <= timing.cap
                    and t["peak_predicted_s"] <= timing.cap,
                    throughput_ratio=ratio, throughput=ratio >= .95,
                    settled=(t["last_variance"] <= .0025 and t["last_variance"] <= t["first_variance"] + .0001)
                    if t["last_variance"] is not None else None)
            if "profile" in result:
                r = result["profile"]
                t = r["time"]
                extra.update(band_factor=band, profile_metrics=dict(
                    complete=all(x["tokens_produced"] == q.eos for x, q in zip(r["requests"], requests, strict=True)),
                    contended_gen_share=t["contended_gen_s"]/t["contended_s"] if t["contended_s"] else None,
                    wall_ratio_vs_protected=r["wall"]/result["protected"]["wall"] if r["wall"] else None))
        if not args.details:
            for r in result.values():
                for k in ("windows", "requests"):
                    del r[k]
        print(json.dumps(dict(chunk_calibration=chunk, trace=name, seed=args.seed, config=asdict(cfg),
                              accept_mean=cost.accept if args.accept_mean is None else args.accept_mean,
                              **extra, **result), allow_nan=False))


if __name__ == "__main__":
    main()
