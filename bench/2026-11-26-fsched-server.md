# Patch 57: isolated server validation

**Keep this policy opt-in.** It achieved the chosen split in the contention
probe and preserved Q8 quality. The real three-pass mixed workload did **not**
reproduce the simulator's short-latency improvement. The longest long-prompt
TTFT rose by 12.75 s. Production remained inactive.

## GPU smoke: all three profiles ran

The opt-in server ran under its own transient unit on loopback HTTPS,
port 18038. Production port 8000 stayed closed. Boot checks required the
native Q8 sidecar on 12 target projections and the presharded BF16-fused
draft `TP-1-v2-mb5f16b15`. Neither the public unit nor its environment was
changed. This was the 2048-position, cap-12 arm, with the old forced
prefill/decode interval disabled.

* Four short streams returned 800/800 tokens, 35.4 aggregate tok/s,
  maximum TTFT 2.82 s, minimum free VRAM 4,901 MiB. This was a traced
  smoke test, not an untraced throughput comparison.
* The separate header/overlap probe rejected `nan` with HTTP 400. A
  `0.125` weight reached the scheduler. A running 256-token stream and
  three shared-prefix requests completed on the real HTTPS endpoint.
* Across the smoke run, 218 completed event pairs recorded **5 pure prefill,
  7 mixed and 206 pure generation** forwards. During 31 contended forwards
  (15.471 s), generation was charged **33.30%**. The denominator is measured
  GPU stream time; the division of a mixed forward between cohorts is
  estimated from shape. This is not the generation fraction of whole-run
  wall time, or evidence of a universal latency bound.
* Maximum predicted forward: 2.640 s. Maximum observed: 2.245 s. Largest
  prefill-bearing shape: 1,857 positions including decode tails. No GPU
  exception or scheduler crash was observed. The isolated smoke unit exited.

Local artifacts: `~/scratch/prefill-isolation-20261125/fsched57-smoke1/`
(`concurrent-4.jsonl`, `journal.log`, `events.jsonl`, `config.json`).
The clean-pin replay passed 113/113 checks. Host scheduler/simulator tests:
33 passed; installed server module: 7 passed.

## What the full comparison measures

The matched `mixed-2k-fixed-full` control uses the same model, cache, draft,
2048-position mixed setting and cap. It uses `shortest-prefill-first` and
the earlier one-round decode interval; the comparison is between complete
scheduling policies, not an ablation of just the debt formula. Across its
three prior 18-short/2-long bursts, the 54 short requests had a median
server lifecycle of 33.48 s; median burst wall time was 122.68 s. Its
separate forced-overlap stream saw a maximum inter-delivery gap of 2.75 s.
These are recorded control results, not simulator predictions.

### Untraced results

| Metric | Prior 2k mixed control | fsched 2:1, 0.5× |
|---|---:|---:|
| Short TTFT, median of 54 server lifecycles | 1.48 s | 1.68 s |
| Short TTFT, p95 of 54 server lifecycles | 33.90 s | 31.31 s |
| Short TTFT, worst at the client | 33.92 s | 34.23 s |
| Short completion, median of 54 server lifecycles | 33.48 s | 33.52 s |
| Long TTFT, worst of six requests | 110.42 s | 123.17 s |
| 18+2 burst wall time, median of three | 122.68 s | 124.18 s |
| Short decode rate, median of 54 streams | 6.220 tok/s | 6.224 tok/s |
| 32k single-request prefill, median of three | 758.09 tok/s | 762.70 tok/s |
| Forced-overlap max client delivery gap (one probe per arm) | 2.75 s | 2.54 s |
| Decode-only accepted tokens per verification, median of three probes | 3.122 | 3.122 |
| Lowest sampled free VRAM | 4,117 MiB | 4,177 MiB |

Short TTFT uses the receive-to-first-token interval, not time to completion.
The fsched median was slower in each pass. The p95 is the 52nd of the
54 sorted observations in each arm. The worst-client row uses the HTTPS
benchmark's clock rather than scheduler event timestamps.

All 144 timed requests in the fsched arm had complete receive/first-token/finish
records, with no request errors. There were no scheduler crashes or GPU
exceptions. The candidate matched the prior Q8 correct/incorrect result on
every one of 51 quality tasks (40 correct). Across 195 scored tokens its
NLL was +0.002522 nats/token relative to BF16, within the 0.004 gate. This
tests the logprob fallback as well as generation; it is not an assertion
that all generated texts are bit-identical. Acceptance comes from the
independent decode-only probes, not the benchmark's unsampled status lines.

The slightly smaller forced-overlap gap is one observation, not a new
streaming guarantee. The untraced run did not capture per-round charges;
33.30% is the separate smoke result, not a measurement of the 18+2 bursts.
The one-continuation native constraint and the cost model are reasons not
to extrapolate the simulator's ranking, not a measured attribution of the
long-TTFT difference. No production flags or deployment changed.

[Per-pass measurements, per-request latencies and paired quality
results](2026-11-26-fsched-server.json). Full local artifacts:
`~/scratch/prefill-isolation-20261125/fsched57-full1/`.
