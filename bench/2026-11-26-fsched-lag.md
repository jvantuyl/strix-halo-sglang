# Does adapting the round budget buy better service?

**The lag ledger made a difference; the PID did not consistently beat a fixed
budget.** In the recorded 18-short + 2-long burst (2k arm, pass 1), the
adaptive scheduler reduced the worst outstanding service debt from 13.27 s
to 0.23 s and raised generation's estimated share under contention from
6.3% to 19.4%. That cost wall time: 142.11 s instead of 120.79 s. Fixed-budget
lag scheduling made the same dispatches on that trace.

These are **CPU simulations**, not GPU speedups or measured per-request GPU
times. All primary results use the pin's actual **one-token mixed row**;
generation-only rounds can use NEXTN. Nothing here changes the server.

## Comparison

Same workloads, per-request RNG, admission cap of 12 and cost calibration
for the four policies. The baseline approximates shortest-prefill-first and
full mixed merge, not the 1k/mixed-off production config. The token policy
is the preceding simulator with mixed width explicitly set to one. Fixed
and PID use the same TDM allocator, starting at 2.7 seconds. Ratio 3 weights
responsiveness 75%, fairness 25%; it does **not** change the 80/20 entitlement.

| Policy | 18+2 wall | 18+2 generation time share | 36-request simultaneous burst wall | Burst worst outstanding lag | Burst worst lag since service |
|---|---:|---:|---:|---:|---:|
| Baseline | 120.79 s | 6.3% | 181.93 s | 6.85 s | 6.85 s |
| Static 80/20 tokens | 121.07 s | 6.5% | 181.86 s | 2.89 s | 1.59 s |
| Fixed time, lag-TDM | 142.11 s | 19.4% | 213.29 s | 0.37 s | 0.499 s |
| PID time, lag-TDM | 142.11 s | 19.4% | 214.57 s | 0.26 s | 0.493 s |

In that synthetic burst PID dispatched **61 prefill, 119 mixed and 233
verify-only forwards**. Generation received 16.1% of contended time, not an
exact 20%: a contention episode can end with an unpaid balance, and the
required safety slot can displace other work. A virtual slot is neither a
fixed token count nor a guaranteed time slice.

## Gates

50 recorded replays (398 requests), one 36-request trace with staggered
arrivals and one 36-request simultaneous burst. We evaluate extremes over
individual workloads, not the pooled average.

| Gate | Result with ratio 3 |
|---|---|
| Every request completes at its observed/synthetic EOS | **52/52** traces, 470 requests per policy |
| Worst positive outstanding lag **and** lag since service ≤ 0.5 s of priority-weighted entitlement | **51/52**; staggered synthetic arrivals reached **0.689 s** |
| Budget always in [0.5, 3.0] s; every predicted forward fits | **52/52**, including cold dispatch; budgets observed 1.574–3.000 s, largest predicted forward 2.716 s |
| End-to-end throughput ≥95% of fixed lag policy on each paired workload | **52/52**; worst ratio 0.994, on the simultaneous burst. On recorded traces PID/fixed differed by at most 0.46% |
| Synthetic burst score variance settles | **Pass**: 0.00630 → 0.00164. Compare first/last 16 rounds starting with both buckets ready, not the uncongested tail. Require ≥32 such rounds, final variance ≤0.0025 (standard deviation 0.05) and no rise above initial variance +0.0001 |

**The lag miss is a non-preemption limit.** At 0.470 s only `s0` was ready.
Its 2,048-token prefill ran for 2.700 s. Six requests arrived during that
forward; priority-3 `s2` earned 0.689 s before the scheduler could dispatch
again. Increasing its priority or changing the PID after the forward cannot
recover that opportunity. The staggered trace also missed the burst variance
criterion (final 0.00320). A bound on **predicted duration** is not a hard
guarantee on responsiveness to future arrivals.

Lag is not TTFT. None of the policies changes admission at cap 12 into a
preemptive queue. On the staggered synthetic workload the longest admission
wait was **201.34 s** with PID and 165.57 s with the baseline. Returning
weighted lag to zero among admitted requests doesn't fix that queue.

## Dispatch and settlement

- At a dispatch boundary, split entitlement **in time**, 80/20 only while
  both buckets are eligible; otherwise the ready bucket gets 100%. Divide
  each bucket by its active priorities. Positive fractional priorities are
  allowed in time-policy runs. Use a common seconds scale, **not** division
  by each request's weight: the latter would turn responsiveness into raw
  wait. The first charge to a late arrival is zero; with admission headroom
  it earns only from its arrival, even during a non-preemptible forward.
  Queued followers can participate in a shared edge at a dispatch boundary.
- Project entitlement to the budget horizon, assign one 64-token prefill
  quantum (or edge tail) or one generation row, then recompute residual lag.
  Coalesce consecutive quanta on one edge. Consider intermediate plans,
  not just a full batch. Compare pure, mixed, and generation-seeded mixed
  plans within the budget; backfill no further than the measured peak-rate
  prefill shape (2,048 here). Prefer plans that avoid exceeding the lag
  allowance, including outstanding **aggregate** bucket debt; among them,
  choose lag-weighted useful work per predicted second. Converting a full
  verify to a mixed row loses proposed output and changes duration. Neither
  EOS nor actual accepted length is visible at this point.
- The floor has **two** mechanisms. The minimum budget must fit an indivisible
  prefill quantum and a single verify. Separately, reserve a slot for the
  oldest overdue eligible request when its earned lag since service reaches
  the floor or it has missed 12 dispatches. The latter protects even a tiny
  priority. Among overdue requests, choose most skipped first. This ensures
  service opportunities, not an absolute latency or admission deadline.
- Charge each elapsed forward **once**. Estimate the prefill fraction from
  shape and the generation fraction from row count/mode, then scale both
  to the observed duration. Divide prefill cost by physical tokens, a
  shared edge's charge by beneficiary weights, and generation cost by
  scheduled rows. A rejected proposal still consumed compute; return unused
  *token* reservations, not elapsed seconds. Total billed and entitled time
  matched total busy time on every primary replay (absolute error <100 ns).
- Outstanding lag is cumulative earned minus billed seconds. For response,
  track earned service since last service, subtract the current charge, and
  take the largest positive shortfall before resetting served rows. Map
  that shortfall `x` to `x/(floor+x)`. Fairness loss is `1 − Jain` over ready
  requests' `(floor+billed)/(floor+earned)`; the floor regularizes arrivals.
  High blended loss shrinks the next budget. The PID uses elapsed seconds
  for integral/derivative terms, a 0.20 loss target, conditional integration
  at saturation, and a ±20% change limit per forward. Reset on **actual idle
  time**, not on every arrival. No PID output changes priority or bucket split.

## Checks, sensitivity and limits

The prior [timing calibration and fixture](2026-11-26-fsched-sim.md) are
unchanged: train on timing passes 1–2 plus the separate decode probe. Replay
pass 3 at each arm's actual budget for an independent wall-clock check. Across
eight scenarios per arm, median absolute error was 4.58% at both sizes;
worst error was 19.1% at 2k and 18.6% at 4k. The held-out mixed bursts were
120.79 vs measured 121.74 s, and 158.97 vs measured 162.55 s. This validates
the reference replay approximately, **not** causal predictions for a new
batch composition. Scheduler Python overhead is not a GPU timing measurement.

With burst seeds 8 and 9 the PID also passed all five gates. At acceptance
means 2.44 and 3.49 the seed-7 burst still passed. With batch priorities
`0.000001,1,3` all requests finished, but peak outstanding debt reached
1.33 s and score variance did not settle. Ratio 0 and 1 both produced
lower final burst score variance than ratio 3, but those are *different
objective functions*, not evidence for changing the user's preference.

**Hypothetical four-slot mixed verify only:** at `--mixed-width 4` the
seed-7 simultaneous burst finished in 197.54 s instead of 214.57 s, but
failed the variance gate (0.01429 final). This alternative has no measured
four-slot mixed implementation or timing anchor. Do not use it as a
deployment forecast.

## Reproduce

```sh
python3 tests/check_fsched_sim.py
python3 tools/fsched_sim.py --controller pid --ratio 3
python3 tools/fsched_sim.py --controller fixed --budget 2.7
python3 tools/fsched_sim.py --controller pid --workload synthetic --arrival-rate 0 --details
python3 tools/fsched_sim.py --controller pid --workload synthetic --priorities 0.000001,1,3
```

`--budget` is **seconds** in fixed/PID mode, tokens for the legacy default.
`--details` includes actual vs predicted durations, budgets, selected work,
round losses, and per-request charged/entitled seconds, lag and TTFT.
[Per-trace four-policy results](2026-11-26-fsched-lag-results.csv).

**Decision:** keep Phase 2 on hold. The next decision is whether to accept
one in-flight forward of lag overshoot on new arrivals, or impose a shorter
prefill cap when the system is lightly loaded. The current data does not
justify trading throughput for PID rather than using the simpler fixed
budget with the same lag allocator.
