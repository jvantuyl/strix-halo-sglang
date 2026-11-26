# Should the server control both the time split and the switch band?

**Keep 2:1 and 0.5× fixed.** A coupled two-input PI controller improved its
training objective by 1.34%, but lost on 11 of 12 held-out workloads. The
equal-weight geometric mean of short latency, longest long-prompt TTFT,
worst streaming gap and wall time got **4.01% worse** out of sample. The
worst short-latency regression was **71.3%**. These are CPU replays, not
GPU measurements, and they do not rule out every possible adaptive policy.

Each cell below is the median of three held-out workload seeds, fixed →
adaptive, seconds. Each workload contains 24 arrivals with shared prefixes,
long and short prompts, unequal priorities, and early outputs. “Short” is
median short request completion; “long” is maximum long TTFT.

| Load | Short | Long | Max inter-delivery gap | Wall |
|---|---:|---:|---:|---:|
| Interactive, occasional long prompts | 42.66 → 58.21 | 63.98 → 54.22 | 2.68 → 2.68 | 77.01 → 76.74 |
| Document-heavy | 11.65 → 11.65 | 218.30 → 218.21 | 2.68 → 2.68 | 222.02 → 221.92 |
| Slower arrivals, wider priority range | 20.35 → 34.87 | 86.55 → 81.24 | 2.68 → 2.68 | 91.57 → 90.57 |
| Within-run change from streams to documents | 34.33 → 50.30 | 142.66 → 142.61 | 2.68 → 2.68 | 145.15 → 145.10 |

The controller observes the longest **absolute time since service** for each
eligible cohort over a 2.7 s window. Differential pressure moves the prefill
share (50–80%); common pressure moves the band (0.25–2×). Both outputs have
slew limits, conditional integration and reset outside contention. Four
gain pairs, including both signs for the band, were fit on six *training*
workloads. The winning pair was `(0.15, -1.5)`. There was no fitting on
the three new seeds or the two new arrival regimes. No derivative term was
used: the actuator is discrete and often flat, and differentiating the peaks
would amplify dispatch timing noise. This is a test of a joint optimization
strategy, not evidence that an untuned 2×2 PID would improve on it.

The median contended generation share across held-out runs fell from 30.9%
to 23.4%. Median worst lag since service rose from 1.58 to 1.90 service
seconds. Median minimum window Jain index improved from 0.37 to 0.44;
that does not pay for worse interactive latency. The score uses absolute
client metrics, not a controller's ability to reduce its own promises. All
54 runs checked completion, predicted cap and service conservation. Model
assumptions are the [one-token mixed path and measured calibration](2026-11-26-fsched-profiles.md);
the simulator permits more simultaneous partial prefills than the server.

[Raw per-run CSV](2026-11-26-fsched-feedback.csv). Reproduce on a new path:

```sh
python3 tools/fsched_feedback_trial.py ~/scratch/fsched-feedback-rerun.csv
```
