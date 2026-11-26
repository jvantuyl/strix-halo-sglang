# Fair-share mixed batches: CPU simulation, 2026-11-26

**Do not proceed to server integration on these results.** Sharing prefill
work by priority looks useful: three users with priorities 1, 2 and 3 get
one prefix computation with weight 6. The problem is the unit of the split.
At 2,048 tokens, 20% reserves about 410 generation tokens. Twelve running
requests can consume **at most 48** in one four-slot verify, or **12** in
the pin's actual mixed forward. Arrival timing, prompt lengths and EOS can
only reduce those bounds. A work-conserving scheduler lends almost all
that reservation back to prefill. A strict split leaves most of it idle.
Neither gives generation 20% of GPU time.

## Default paired experiment

Same arrivals, requests, row cap (12), per-request acceptance RNG seed (7)
and cost model for both policies. Both use 2,048-token rounds, including
when arrivals were recorded in the 4k arm. The reference approximates
shortest-prefill-first with full mixed merge, **not** the production
1k/mixed-off configuration. The candidate mixes four-slot speculative
rows with prefill, which is a hypothesis, not a capability of the pin.

| Gate | 50 recorded replays | Synthetic, 36 requests, priorities 1:2:3 |
|---|---:|---:|
| All complete, no wait reaching the 64-round alarm | 44/50 | Fail (74 rounds queued) |
| Each priority class within 10% of weighted entitlement in each bucket | 50/50 (only default priority) | Fail: prefill worst error 0.42%, generation 90.3% |
| Worst-window Jain index at least the reference's on each trace | 50/50 | Pass: 0.818 vs 0.792 |
| Physical prefill throughput within 5% of reference | 44/50 | Fail: −12.1% |

All **398 recorded and 36 synthetic requests completed** under both policies.
The six recorded failures were the six 18-short + 2-long bursts. The candidate's
worst prefill loss was 13.2%. In those bursts the admission queue waited up
to 66 rounds. No admitted candidate request missed even one eligible round at
the default budget. The 64-round alarm is a conservative diagnostic (two
measurement windows), not proof of infinite starvation. A finite replay can
detect starvation symptoms, not prove liveness for all future workloads.

On the synthetic trace, priority-1 generation got 697 tokens vs an ideal
366; priority-3 got 1,572 vs 1,963. Even if every row runs, rejected proposals
cannot be transferred to a heavy request whose verify already ran. This is
why passing the single-class replays alone says nothing about priority.
With a genuinely binding two-token budget, the unit test checks accepted
service against 1:2:3 over 300 rounds while all three requests are backlogged.

## What a round means

- **Eligibility:** Dispatch at arrivals or the previous forward's completion.
  Admit at most 12 requests, FCFS for the candidate. An admitted running
  request keeps its slot. A queued user can benefit from an admitted user's
  shared prefix without occupying a compute row. Report admission waits
  separately from rounds missed after admission. A new arrival does not
  accrue credit for earlier rounds.
- **Composition:** Start with 80/20, carrying fractional split tokens into
  later rounds. Water-fill capped shares in each bucket, then allocate integer
  work by earned credit less actual service. The default lends unusable quota;
  `--strict` disables cross-bucket lending. Prefill takes 64-token chunks or
  the final edge tail; sub-page test budgets use token granularity. No round
  spends more physical tokens than its budget. A full prompt prefill also
  samples the first output; it is not billed again as a verify.
- **Sharing:** Only the next unfinished radix edge is eligible. Its weight
  is the sum of the priorities of current beneficiaries. A token advances
  all of them but is billed once, divided by weight. A child edge cannot run
  until its parent has been published. Prefix identities include ancestors;
  equal token counts are not cache keys. No eviction or KV pressure modeled.
- **Settlement:** Cap reservations by remaining **requested** length and one
  verify per row, not by future EOS. Before dispatch, redistribute known
  unused capacity. After dispatch, charge accepted tokens only: EOS releases
  the request and the current round's ledger redistributes entitlement over
  actual service. Unknown rejected slots cannot be reissued by the *same*
  forward. Deficits persist per request and bucket. There is no credit for
  time before arrival or after completion.
- **Metrics:** Per round, the independent fluid reference divides **actual
  physical service** by active weights, capped by each row's known capacity.
  It does not know future acceptance either. Sum by class and bucket for
  the ±10% check. For Jain, use per-request actual/entitled service in
  nonoverlapping 32-round windows, including zero service to entitled users;
  compare the minimum over windows and buckets, including the last partial
  window. Prefill tok/s is physical prefill tokens divided by time for all
  forwards containing prefill, **including mixed work**. Wall time and client
  metrics are separate outputs. `--details` includes every batch, wait, TTFT,
  completion time and output tok/s.

## Inputs and limits on interpretation

[The 2k/4k measurements](2026-11-25-prefill-mixed.jsonl) supply 48 scenarios.
[The numerical fixture](2026-11-26-fsched-arrivals.jsonl) joins each scenario
to `req_received` / `req_finished` in its arm's `events.jsonl` and pass
window, in the benchmark driver's scenario order. The four fields per request
are relative arrival seconds, input length, actual output length, and cached
length. No text or request IDs. Bursts with 64-token hits model one shared
chat page; the recorder does not contain token IDs to prove radix identity.
For the two [quality gauntlet](2026-11-24-qsa-native-q8-quality.jsonl)
replays, use the observed cache hits and closed-loop think times: issue the
next request only after the previous one finishes. Replay all 51 generation
limits (64) and four prefill-only logprob requests (0). This tests the
request shapes and early stops, **not** model quality.

The cost model interpolates physical prefill rate by dispatched size and
per-stream decode rate by batch width. Anchors from timing passes 1–2:
758 tok/s at 2k, 476 at 4k. Smaller shapes approach the short-prompt rate
of 672 tok/s. The mixed decode component uses observed mixed short-stream
rates of 6.01 and 5.91 tok/s. Verify outcomes range from 1 to 4, using
one guaranteed token plus three Bernoulli draws fitted to the separate
decode-only probe (mean 3.122). The duration is charged for executing the
verify, not discounted if it stops early. These client measurements include
scheduling delays; adding prefill and generation time is an approximation,
especially for wide mixed verifies that **no hardware arm measured**.

**Hold-out check:** run the reference policy at each arm's *actual* budget
on pass 3. Median absolute scenario wall-time error was 4.58% at 2k and
4.58% at 4k; worst errors were 19.1% and 18.6% (shortest single client).
The mixed burst took 120.79 vs measured 121.74 s at 2k, and 158.97 vs
162.55 s at 4k. These checks support comparisons, not a 5%-accurate
prediction for a particular request. Neither prompt-dependent verify
acceptance nor memory, kernel launches or graph changes are modeled.

**Sensitivity:** keeping the candidate on today's one-token mixed path
cleared the throughput check on all 51 workloads (worst ratio 0.995),
but synthetic generation still missed the priority gate by 60.7% in the
worst class. At accept means 2.44 and 3.49, the wide-mixed candidate's
worst throughput ratios were 0.871 and 0.872. Disabling donation lowered
the worst ratio to 0.824. Increasing the round budget to 4,096 made the
relative throughput gate pass on all workloads, but class error remained
88.9% on synthetic; a relative gate is not evidence that 4k is faster.

## Reproduction

From the repo root, no server, model, dependencies or GPU required:

```sh
python3 tests/check_fsched_sim.py
python3 tools/fsched_sim.py                         # one JSON object per paired trace
python3 tools/fsched_sim.py --mixed-width 1         # current mixed hardware semantics
python3 tools/fsched_sim.py --budget 4096 --strict
python3 tools/fsched_sim.py --accept-mean 2.44
python3 tools/fsched_sim.py --workload synthetic --arrival-rate 0 --priorities 1,2,3 --details
```

[Default paired results](2026-11-26-fsched-results.jsonl). Keep the
prefix-sharing allocator for the next experiment. Before patching the
server, compare the token split to a **forward-time split** using the
existing one-token mixed path, with a separately chosen maximum decode
wait. That would change the resource being promised and needs a decision,
not a silent change to the 80/20 rule.
