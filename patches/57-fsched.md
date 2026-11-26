# 57. Choose the forward profile by cohort debt, fill by request lag

Experimental, **off by default**. Select in the trial's `QWEN38_EXTRA_ARGS`
with `--schedule-policy fsched --chunked-prefill-size 2048 --enable-mixed-chunk
--prefill-decode-interval 0`. Keep the 12-row cap and the existing TP1, PP1,
NEXTN 3/1/4, overlap, prefix cache and QSA/Mamba settings. No change to
context, pool or KV dtype. No new priority-preemption mode.

`X-SGLang-Priority` is a finite, positive service weight (default 1),
including fractions for background jobs. It is not the native integer
queue priority. Batched prompts inherit the header. The ordinary server
ignores this header. `SGLANG_FSCHED_TRACE=1` in the **container** logs the
selected and actual profile, physical input positions, generation rows,
predicted time, measured GPU stream time and attribution, not prompt data.

* When both cohorts can run, earn 1/3 of GPU service time for generation
  and 2/3 for prefill. Price a packed candidate for each feasible profile.
  Prefer mixed when both the current and projected balances fit within half
  the debt a full feasible verify round repays. Otherwise minimize absolute
  projected debt. A single eligible cohort takes all service, without banking.
* Native cache matches precede the decision. A FIFO window sets eligibility,
  not time spent waiting for an unallocatable slot. Within each cohort,
  positive request lag is earned time minus charged time. A 0.5 service-second
  or 12-dispatch skip guard takes priority when feasible. Pack page quanta
  by projected lag, not prompt age. A complete small row may fit alongside
  a partial. The adder retains final authority over KV, rows and page padding.
* Pending requests can pool weight only on the same pending checkpoint page:
  exact token prefix, cache namespace, salt and uncached frontier. Charge
  common positions once, in proportion to weights; stop at the last shared
  grid boundary and bill the private suffix to its owner. This is cost
  attribution to *potential* cache beneficiaries, not a promise that later
  cache hits survive cancellation, checkpoint placement or eviction.
* Pure generation uses the native speculative path. Mixed takes one position
  per running generator, not four. The calibrated 2.7 s bound is a **prediction**
  using the [recorded cost curves](../bench/2026-11-26-fsched-sim.md);
  it is neither a deadline nor a wall-clock streaming SLO.
* Snapshot the **actual** forward and book an estimated charge. Queue paired
  HIP timing events around the worker on its forward stream. At later decision
  boundaries, query completed events and correct the ledger by the difference.
  The measured quantity is elapsed stream time across the forward, including
  host launch stalls between its kernels, not the sum of kernel busy times.
  Mixed prefill/generation attribution within that measurement is estimated.
  No device-wide or per-round blocking synchronization. A late event cannot
  revive a cleared contention balance or charge a reused client request ID.

## Limits of the pin

The native scheduler owns **one** unfinished prefill. It can't publish a dozen
independent partials from one dispatch. Keep a continuation's KV lifecycle,
schedule complete lag-ranked companions that fit, and admit at most one new
partial. Running generation rows stay together with their speculative relay
tensors; all fit within the fixed cap. Thus a generation-heavy profile does
not reorder decode rows by weight. When all are runnable, each receives one
row. A request arriving inside a non-preemptible forward first enters the
server ledger at the next eligibility decision. These differences from the
idealized simulator matter when interpreting fairness.

Logprobs, beams, embeddings/overrides and sessions do not use mixed packing.
They fall back to native pure prefill or decode. Native allocator rejection
can also turn a proposed profile into a different actual forward.

Implementation: [anchored patch](patch_fsched_scheduler.py),
[installed runtime module](fsched/runtime.py), [CPU checks](../tests/check_fsched_server.py).
See the [isolated GPU measurements](../bench/2026-11-26-fsched-server.md).
The [cross-workload feedback test](../bench/2026-11-26-fsched-feedback.md)
is offline only; the server does not adjust the user-selected split or band.
