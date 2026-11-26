"""Three fixed-cap forward profiles, selected by projected cohort service debt."""
from fsched_lag import compose


def _limit(cfg, cost):
    knee, _ = max(cost.prefill_rates, key=lambda point: point[1])
    return min(cfg.budget, max(cfg.quantum, knee))


def _backfill(ready, ledger, cfg, cost, pa, ga, limit, forced=None, whole_budget=False):
    """Fill eligible physical edges; a shared edge uses one row, not one per follower."""
    pa = pa.copy()
    n = sum(pa.values())
    room = limit - sum(ga.values())
    budget = ledger.pid.budget
    mean_accept = cost.accept if cfg.mixed_width == 4 else 1.
    while n < room:
        _, paid = ready.charges(pa, ga, cost, cfg.mixed_width)
        def rank(r):
            add = min(cfg.quantum, r.capacity - pa.get(r.key, 0))
            # Do not allocate a whole round's projected credit for the first
            # quantum: a high-weight row could take a short batch unopposed.
            horizon = budget if whole_budget else cost.seconds(n + add, ga, mean_accept)
            return (not pa and forced in r.members, sum(ledger.lag(i) + horizon * ready.rate[i]
                                                       - paid.get(i, 0.) for i in r.members))
        for row in sorted(ready.pre.values(), key=rank, reverse=True):
            old = pa.get(row.key, 0)
            add = min(cfg.quantum, row.capacity - old)
            if add and n + add <= room and cost.seconds(n + add, ga, mean_accept) <= budget:
                pa[row.key] = old + add
                n += add
                break
        else:
            break
    return pa


def protected(ready, ledger, cfg, cost):
    """Comparator: previous PID decision, top up only if it already contains prefill."""
    pa, ga = compose(ready, ledger, cfg, cost, cfg.mixed_width)
    if pa:
        pa = _backfill(ready, ledger, cfg, cost, pa, ga, _limit(cfg, cost), whole_budget=True)
    return pa, ga


def choose_profile(ready, ledger, cfg, cost, band_factor):
    """Compare one packed candidate per mode. No token, accept or EOS oracle."""
    budget = ledger.pid.budget  # constant for this policy
    limit = _limit(cfg, cost)
    forced = ledger.guard(ready, cfg.cap)
    if forced is not None:
        ledger.guard_rounds += 1
    rank_gen = sorted(ready.gen.values(), key=lambda r: (
        forced in r.members, ledger.lag(r.key) + budget * ready.rate[r.key]), reverse=True)

    def generate(width, min_prefill):
        ga = {}
        used = min_prefill
        capacity = limit if min_prefill else cfg.budget
        for row in rank_gen:
            n = min(row.capacity, width)
            if used + n > capacity:
                continue
            trial = {**ga, row.key: n}
            if cost.seconds(min_prefill, trial, cost.accept if width == 4 else 1.) <= budget:
                ga = trial
                used += n
        return ga

    # A mixed candidate must fit at least one eligible quantum *and* one
    # generation row. Reserve the forced edge first when the guard names it.
    plans = {}
    if ready.pre:
        pa = _backfill(ready, ledger, cfg, cost, {}, {}, limit, forced)
        if pa:
            plans["prefill"] = pa, {}
    if ready.gen:
        ga = generate(4, 0)
        if ga:
            plans["generation"] = {}, ga
    both = bool(ready.pre and ready.gen)
    if both:
        required = [r for r in ready.pre.values() if forced in r.members]
        smallest = min(min(cfg.quantum, r.capacity) for r in required or ready.pre.values())
        ga = generate(cfg.mixed_width, smallest)
        if ga:
            pa = _backfill(ready, ledger, cfg, cost, {}, ga, limit, forced)
            if pa:
                plans["mixed"] = pa, ga

    owed = ledger.gen_debt if both else 0.
    g = 1 - ready.percent / 100
    # The reference mass is the amount a full feasible generation round
    # would repay at this concurrency, not the duration of a prefill chunk.
    band = (band_factor * (1-g) * cost.seconds(0, plans["generation"][1], cost.accept)
            if both and "generation" in plans else 0.)
    projected = {}
    for name, (pa, ga) in plans.items():
        dt, charge = ready.charges(pa, ga, cost, cfg.mixed_width)
        if forced is None or forced in charge:
            projected[name] = owed + g*dt - sum(charge[k] for k in ga) if both else 0.
    if not projected:
        if ready.rate:
            raise ValueError("cannot fit eligible protected work in a profile round")
        return {}, {}, {}
    if ("mixed" in projected and abs(owed) <= band and abs(projected["mixed"]) <= band):
        name, reason = "mixed", "inside_band"
    else:
        name = min(projected, key=lambda k: (abs(projected[k]), k != "mixed"))
        reason = "closest_debt" if both else "one_cohort"
    return *plans[name], dict(profile=name, reason=reason, debt_before_s=owed, band_s=band,
                              projected_debt_s=projected, guarded_request=forced)
