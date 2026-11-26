"""Time-based round composition and feedback. No access to future arrivals or EOS."""
import math
from collections import Counter
from dataclasses import dataclass
from statistics import pvariance

from fsched_replay import interp


@dataclass(frozen=True)
class TimeBudget:
    initial: float = 2.7
    floor: float = .5
    cap: float = 3.
    ratio: float = 3.

    def __post_init__(self):
        if (not all(map(math.isfinite, (self.initial, self.floor, self.cap, self.ratio)))
                or not 0 < self.floor <= self.initial <= self.cap or self.ratio < 0):
            raise ValueError("need 0 < budget-floor <= budget <= budget-cap and finite ratio >= 0")

    def blend(self, response_loss, fairness_loss):
        return response_loss + (fairness_loss - response_loss) / (1 + self.ratio)


class Controller:
    """Positional PID in seconds. Higher loss reduces the next round budget."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.budget = self.cfg.initial
        self.integral = 0.
        self.previous = None

    def update(self, loss, dt):
        if not dt:
            return
        error = .2 - loss
        slope = 0 if self.previous is None else (error - self.previous) / dt
        self.previous = error
        lo = max(self.cfg.floor, .8 * self.budget)
        hi = min(self.cfg.cap, 1.2 * self.budget)
        base = self.cfg.initial * (1 + error + .025 * max(-2., min(2., slope)))
        candidate = self.integral + .20 * self.cfg.initial * error * dt
        target = base + candidate
        # Conditional integration at both actuator and slew limits.
        if not ((target > hi and error > 0) or (target < lo and error < 0)):
            self.integral = candidate
        self.budget = min(hi, max(lo, base + self.integral))


class Ready:
    """Physical rows and normalized entitlement rates at a dispatch boundary."""
    def __init__(self, pre, gen, requests, percent):
        self.pre = {r.key: r for r in pre}
        self.gen = {r.key: r for r in gen}
        self.requests = requests
        self.percent = percent
        self.parts = {}
        self.rate = {}
        for rows, fraction in ((pre, percent / 100 if gen else 1),
                               (gen, 1 - percent / 100 if pre else 1)):
            weight = sum(r.weight for r in rows)
            for r in rows:
                self.parts[r.key] = [(i, requests[i].weight / r.weight) for i in r.members]
                self.rate.update((i, fraction * (requests[i].weight / weight)) for i in r.members)

    def charges(self, pa, ga, cost, width, measured=None):
        """Price prefill tokens and verify rows, then reconcile to ONE duration."""
        n = sum(pa.values())
        ptime = cost.seconds(n, {}, 0)
        predicted = cost.seconds(n, ga, cost.accept if not pa or width == 4 else 1.)
        scale = measured / predicted if measured is not None and predicted else 1.
        per_tok = scale * ptime / n if n else 0
        per_row = scale * (predicted - ptime) / len(ga) if ga else 0
        bill = {i: count * per_tok * f for k, count in pa.items() for i, f in self.parts[k]}
        bill.update((k, per_row) for k in ga)
        return predicted, bill


class TimeLedger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pid = Controller(cfg)
        self.ideal, self.actual, self.since, self.skips = (Counter() for _ in range(4))
        self.peak_lag = Counter()
        self.worst_lag = self.worst_since = self.peak_pred = self.peak_elapsed = 0.
        self.min_budget, self.max_budget = cfg.cap, 0.
        self.max_skips = self.guard_rounds = 0
        self.contention = Counter()
        self.gen_debt = 0.
        self.scores = []

    def lag(self, i):
        return self.ideal[i] - self.actual[i]

    def guard(self, ready, cap):
        # Time debt protects heavier priorities earlier. The oldest-skipped
        # escape hatch also protects a batch priority arbitrarily close to zero.
        overdue = [i for i in ready.rate
                   if self.since[i] >= self.cfg.floor or self.skips[i] >= cap]
        return max(overdue, key=lambda i: (self.skips[i], self.since[i], -i), default=None)

    def settle(self, ready, pa, ga, cost, width, elapsed, adaptive, newcomers=()):
        predicted, bill = ready.charges(pa, ga, cost, width, elapsed)
        budget = self.pid.budget
        self.min_budget = min(self.min_budget, budget)
        self.max_budget = max(self.max_budget, budget)
        self.peak_pred = max(self.peak_pred, predicted)
        self.peak_elapsed = max(self.peak_elapsed, elapsed)
        if not ready.rate or not elapsed:
            return {}
        earned = Counter()
        gen_paid = sum(bill[i] for i in ga)
        def accrue(dt, rates, gen_ids):
            for i, rate in rates.items():
                earned[i] += dt * rate
            if gen_ids and len(gen_ids) < len(rates):
                charged = gen_paid * dt / elapsed
                self.gen_debt += (1 - ready.percent / 100) * dt - charged
                self.contention.update(seconds=dt, gen_seconds=charged)
            elif dt:
                self.gen_debt = 0.

        # A forward cannot be preempted. New arrivals with a free admission
        # slot earn service only for its remainder. They never change its bill.
        rates, gen_ids, start = ready.rate, set(ready.gen), 0.
        for offset, i, generate in newcomers:
            accrue(offset-start, rates, gen_ids)
            if generate:
                gen_ids.add(i)
            ids = [*rates, i]
            sums = Counter()
            for j in ids:
                sums[j in gen_ids] += ready.requests[j].weight
            rates = {j: ready.requests[j].weight / sums[j in gen_ids] *
                     ((1-ready.percent/100 if j in gen_ids else ready.percent/100) if len(sums) == 2 else 1)
                     for j in ids}
            start = offset
        accrue(elapsed-start, rates, gen_ids)
        attainment = []
        worst = 0.
        for i, due in earned.items():
            paid = bill.get(i, 0.)
            self.ideal[i] += due
            self.actual[i] += paid
            self.worst_lag = max(self.worst_lag, self.lag(i))
            self.peak_lag[i] = max(self.peak_lag[i], self.lag(i))
            # Priority is already in earned. Dividing by the request's own
            # rate here would convert this back into absolute wait time.
            worst = max(worst, self.since[i] + due - paid)
            self.since[i] = 0. if paid else self.since[i] + due
            self.skips[i] = 0 if paid else self.skips[i] + int(i in ready.rate)
            self.max_skips = max(self.max_skips, self.skips[i])
            attainment.append((self.cfg.floor + self.actual[i]) / (self.cfg.floor + self.ideal[i]))
        self.worst_since = max(self.worst_since, worst)
        resp = worst / (self.cfg.floor + worst)
        fair = max(0., 1 - sum(attainment)**2 / (len(attainment) * sum(x*x for x in attainment)))
        loss = self.cfg.blend(resp, fair)
        if ready.pre and ready.gen:
            self.scores.append(loss)
        if adaptive:
            self.pid.update(loss, elapsed)
        return dict(budget_s=budget, next_budget_s=self.pid.budget, predicted_s=predicted,
                    response_loss=resp, fairness_loss=fair, loss=loss, since_service_lag_s=worst)

    def summary(self):
        scores = self.scores
        first = last = None
        if len(scores) >= 32:
            first, last = pvariance(scores[:16]), pvariance(scores[-16:])
        return dict(worst_lag_s=self.worst_lag, worst_since_service_s=self.worst_since,
                    max_skips=self.max_skips, guard_rounds=self.guard_rounds,
                    min_budget_s=self.min_budget, max_budget_s=self.max_budget,
                    peak_predicted_s=self.peak_pred, peak_elapsed_s=self.peak_elapsed,
                    billed_s=sum(self.actual.values()), entitled_s=sum(self.ideal.values()),
                    contended_s=self.contention["seconds"], contended_gen_s=self.contention["gen_seconds"],
                    contended_rounds=len(scores), first_variance=first, last_variance=last)


def compose(ready, ledger, cfg, cost, mixed_width):
    """Allocate virtual slots, then price the discontinuity at the mixed flip."""
    budget = ledger.pid.budget
    forced = ledger.guard(ready, cfg.cap)
    if forced is not None:
        ledger.guard_rounds += 1
    horizon = {i: ledger.lag(i) + budget * rate for i, rate in ready.rate.items()}
    pressure = {k: sum(max(0., horizon[i]) for i, _ in parts) for k, parts in ready.parts.items()}
    knee, peak_rate = max(cost.prefill_rates, key=lambda point: point[1])
    limit = max(knee, cfg.quantum)
    p = (cost.accept - 1) / 3
    # E[min(accepted, reserved)] under the replay's verify model.
    expected = [sum(math.comb(3, j) * p**j * (1-p)**(3-j) * min(1+j, slots) for j in range(4))
                for slots in range(5)]
    solo = interp(cost.decode_rates, 1)
    gen_rate = sum(ready.rate[i] for i in ready.gen)
    best, best_score = None, None

    def consider(pa, ga, dt, paid):
        nonlocal best, best_score
        if not paid or (forced is not None and forced not in paid):
            return
        # An attainable safety bound takes precedence over useful work/s.
        # Retain intermediate plans: a final prefill quantum can put a
        # follower over its limit, and adding a verify row needn't be free.
        projected = max(max(ledger.lag(i), ledger.since[i]) + rate * dt - paid.get(i, 0.)
                        for i, rate in ready.rate.items())
        if ready.pre and ready.gen:
            # Outstanding bucket service survives request completions within
            # a contention episode; per-request credits alone cannot do that.
            projected = max(projected, abs(ledger.gen_debt + gen_rate * dt - sum(paid[k] for k in ga)))
        excess = max(0., projected - ledger.cfg.floor)
        value = sum(pressure[k] * n / peak_rate for k, n in pa.items())
        value += sum(pressure[k] * expected[n] / solo for k, n in ga.items())
        score = (-excess, value / dt, sum(pa.values()) + sum(ga.values()))
        if best_score is None or score > best_score:
            best, best_score = (pa.copy(), ga.copy()), score

    def fill(pre, gen, seed_gen=False):
        pa, ga = {}, {}
        n = 0
        while True:
            dt, paid = ready.charges(pa, ga, cost, mixed_width)
            consider(pa, ga, dt, paid)
            def rank(r):
                members = ready.parts[r.key]
                due = sum(horizon[i] - paid.get(i, 0.) for i, _ in members)
                return (not (pa or ga) and forced in r.members, due)
            candidates = [r for r in pre if pa.get(r.key, 0) < r.capacity and n < limit]
            candidates.extend(r for r in gen if r.key not in ga)
            if seed_gen and not ga:
                candidates = gen
            elif seed_gen and not pa:
                candidates = pre
            for r in sorted(candidates, key=rank, reverse=True):
                is_pre = r.key in ready.pre
                add = min(cfg.quantum, r.capacity - pa.get(r.key, 0), limit - n) if is_pre else 0
                width = mixed_width if n + add else 4
                rows = ga if is_pre else {**ga, r.key: min(width, r.capacity)}
                dt = cost.seconds(n + add, rows, cost.accept if width == 4 else 1.)
                if dt > budget:
                    continue
                if is_pre:
                    pa[r.key] = pa.get(r.key, 0) + add
                    ga = {k: min(v, width) for k, v in ga.items()}
                    n += add
                else:
                    ga = rows
                break
            else:
                return

    pre, gen = list(ready.pre.values()), list(ready.gen.values())
    fill(pre, gen)
    if pre and gen:
        # Compare the filled mixed plan to *full verifies*, not its narrowed
        # decode rows. Also consider whether adding any decode fits at all.
        fill([], gen)
        fill(pre, [])
        fill(pre, gen, seed_gen=True)
    if best is None:
        if pre or gen:
            raise ValueError("time budget cannot fit one eligible prefill quantum or verify row")
        return {}, {}

    return best
