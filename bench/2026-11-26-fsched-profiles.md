# Three-profile service-debt scheduler (CPU replay)

**The selector met the cohort target without giving back the throughput gain
from protected backfill. It did not meet the old per-request lag target.**
Across all six recorded 18-short + 2-long bursts, it spent 19.84–20.03% of
contended time on generation and finished in 122.03–122.24 s. These are
simulator results using the one-token mixed forward, not measured server
performance.

## Same requests, same work

2k arm, pass 1. Every policy consumed 52,964 physical prefill tokens and
returned 3,728 output tokens. Generation share is estimated service time
while both cohorts have work, **not** the fraction of generated tokens or
of all wall time.

| Policy | Wall (s) | Generation share | Median short completion (s) | Worst outstanding lag (s) | Worst lag since service (s) |
|---|---:|---:|---:|---:|---:|
| Original shortest-prefill-first mixed reference | 120.79 | 6.32% | 36.97 | 13.27 | 13.27 |
| Previous per-request time-lag PID | 142.11 | 19.37% | 30.71 | 0.23 | 0.21 |
| PID with protected prefill backfill | 122.47 | 19.06% | 33.62 | 0.19 | 0.30 |
| **Three profiles, fixed cap** | **122.03** | **19.84%** | **32.89** | **0.39** | **0.88** |

The 0.44 s wall-time difference versus protected backfill is too small to
call a speedup. Compared with the original PID's 109 mixed forwards, the
selector dispatched 31 mixed, 2 pure prefill and 122 full generation
forwards. At the baseline's finish time (120.79 s), it had delivered 3,660
tokens and finished 14 requests. Protected backfill had delivered 3,648
and finished 17. Delivering more tokens early need not finish more requests.

## Decision rule

1. While both cohorts have work, keep a signed balance `D`: 20% of elapsed
   time minus time charged to generation. Positive means generation is
   behind, negative means prefill is behind. Requests don't inherit debt
   because their prompt is long. When only one cohort can run, serve it
   without banking idle-cohort credit.
2. Construct one feasible candidate per profile. Generation-only packs full
   verify rows; mixed packs one position per eligible generator, then fills
   prefill; pure prefill uses the block for prompt work. Limit prefill-bearing
   blocks to the measured throughput knee (2,048 positions here, including
   decode rows), or the fixed duration cap (2.7 s), whichever binds first.
   Page-sized quanta, edge tails, actual ready work and the request cap of 12
   also constrain packing. Do **not** turn a selected full verify into a mixed
   forward after the decision.
3. Inside each candidate rank requests by projected owed service less billed
   service. Bill shared prefix tokens once, splitting cost by beneficiary
   priority. Keep the existing guard: when a request is overdue, only consider
   plans that serve it. These are *opportunities* for service, not a bound on
   arrival or delivery latency during a non-preemptible forward.
4. Project each candidate's ending balance as `D + 0.2 * duration - gen_charge`.
   Band half-width is `factor * 0.8 * predicted_generation_only_duration`,
   default factor 2. This is the debt two feasible generation rounds could
   repay **at the current concurrency**. Prefer mixed if the balance both
   starts and ends inside that band. Otherwise select the eligible candidate
   ending closest to zero. Use actual elapsed time to settle; only the split
   of a mixed forward is estimated. Neither PID nor future arrivals influence
   this profile decision.

## Checks and limitations

- 50 recorded and two synthetic traces; all 470 requests completed under
  each policy. All 4,244 selector dispatches fit their individual budgets.
  All six primary bursts passed both comparison checks: generation share
  within two percentage points of 20%, wall no more than 5% above protected
  backfill. Across all 52 traces the largest wall regression versus
  protected backfill was 2.28% (2k four-stream concurrent workload, pass 3).
- On the 36-request simultaneous priority burst, profile selection finished
  in 183.44 s vs 196.50 s for protected backfill, with generation getting
  18.65% vs 15.78%. But peak per-request outstanding lag rose from 0.85 s
  to **2.02 s**. Overall, the selector missed the prior 0.5 s individual
  lag gate on eight traces: the six recorded mixed bursts and both
  synthetics. The cohort band is not a per-request deadline.
- **Band sensitivity:** on the primary burst, factors 0/1/2/4 gave
  122.36/122.53/122.03/122.02 s respectively. Factor 0 removes the preference
  for mixed near balance; it does not prohibit mixed. Factor 1 brought peak
  lag since service down from 0.88 s to 0.34 s there, but not on the 36-request
  burst (1.74 s). A narrower band alone isn't a general individual-lag fix.
  Reducing the fixed cap to 1.5 s at factor 2 gave 128.21 s and 0.30 s worst
  lag since service on the primary burst.
- **Model limitation:** the historical calibration prices the generation part
  of a mixed forward at a fixed time independent of decode row count. This
  rewards packing, but it is not a GPU measurement of that marginal cost.
  The [previous hold-out check](2026-11-26-fsched-lag.md) tested the
  reference's wall times, not these new compositions. None of the primary
  results assume four-token mixed verifies.

## Reproduce

```sh
python3 -m unittest discover -s tests -p 'check_fsched*.py'
python3 tools/fsched_sim.py --controller profile
python3 tools/fsched_sim.py --controller profile --band-factor 1 --details
python3 tools/fsched_sim.py --controller profile --budget 1.5 --workload synthetic --arrival-rate 0
```

`--budget` is the selector's fixed duration cap in seconds; the PID and
protected-backfill comparators start at that budget and then adapt within
their usual limits. `--details` includes the candidate ending debts, actual
settlement, band, guard and switch reason for each dispatched profile.
[Per-trace comparisons and useful progress at equal wall time](2026-11-26-fsched-profiles.csv).

**Decision:** retain this as a simulator candidate. Before a server trial,
test whether a *projected per-request lag veto* on long blocks can protect
individual streams without losing the cohort and wall-time results. No
server change or deployment was made.
