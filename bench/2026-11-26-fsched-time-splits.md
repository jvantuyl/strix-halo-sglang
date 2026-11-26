# Profile scheduler: prefill:generation time-split sweep

Columns are target shares of **GPU service time during contention**, not
the ratio of requests or the number of tokens. The tables use the original
2k arm, pass 1 (18 short + 2 long). Each of the 54 settings also ran on
the other five recordings of this burst. Seed 7, fixed 2.7 s cap,
one-token mixed row, 2,048-token prefill target, same arrivals and EOS
within each paired replay. All settings did the same useful work.

### Median short-request completion (wall seconds)

| Band / time split | 16:1 | 8:1 | 4:1 | 2:1 | 1:1 | 1:2 | 1:4 | 1:8 | 1:16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25× | 30.70 | 30.70 | 31.93 | 30.61 | 30.29 | 30.29 | 30.29 | 30.07 | 30.07 |
| 0.5× | 36.45 | 32.89 | 31.49 | 30.29 | 30.29 | 30.29 | 30.29 | 30.07 | 30.07 |
| 1× | 36.45 | 35.32 | 31.49 | 30.29 | 30.29 | 30.29 | 30.29 | 30.07 | 30.07 |
| 2× | 36.45 | 36.45 | 32.89 | 30.29 | 30.29 | 30.29 | 30.29 | 30.29 | 30.07 |
| 3× | 36.45 | 36.45 | 35.32 | 31.49 | 30.29 | 30.29 | 30.29 | 30.29 | 30.29 |
| 4× | 36.45 | 36.45 | 36.45 | 32.89 | 30.29 | 30.29 | 30.29 | 30.29 | 30.29 |

### Worst lag since service (owed service seconds)

| Band / time split | 16:1 | 8:1 | 4:1 | 2:1 | 1:1 | 1:2 | 1:4 | 1:8 | 1:16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25× | 0.058 | 0.162 | 0.299 | 0.429 | 0.583 | 0.805 | 0.982 | 2.190 | 2.328 |
| 0.5× | 0.901 | 0.922 | 0.337 | 0.396 | 0.583 | 0.805 | 0.982 | 2.190 | 2.328 |
| 1× | 0.901 | 0.832 | 0.337 | 0.396 | 0.583 | 0.805 | 0.982 | 2.190 | 2.328 |
| 2× | 0.901 | 0.832 | 0.876 | 0.396 | 0.583 | 0.805 | 0.982 | 1.101 | 2.328 |
| 3× | 0.901 | 0.832 | 0.715 | 0.714 | 0.583 | 0.805 | 0.982 | 1.101 | 1.170 |
| 4× | 0.901 | 0.832 | 0.715 | 0.904 | 0.583 | 0.805 | 0.982 | 1.101 | 1.170 |

## Sweet spots across all six recordings

- **Interactive, keeping the 0.5 s lag target:** 2:1 and 0.5× (1× and 2×
  tie on both requested outputs). Median short completion 30.29–30.44 s,
  worst lag 0.396–0.429 s. The longest short-stream inter-delivery gap was
  2.66–2.67 s. This is the fastest short median among settings that met
  the lag target on *every* recording.
- **Prefill-heavy:** 16:1 and 0.25× gave the lowest worst lag, 0.058–0.060 s.
  Median short completion was 30.70–30.86 s. The slower long request got its
  first token in 104.21–104.48 s, versus 119.63–120.03 s for 2:1/0.5×.
  The low lag is relative to a small 5.88% generation entitlement. Short
  streams actually went up to 5.27 s between deliveries. Widening the band
  just to 0.5× at 16:1 increased median short completion to 36.45–36.61 s.
- **Generation-heavy has diminishing returns:** 1:8 and 0.5× cut only about
  0.22 s off the short median compared with 2:1/0.5×. Lag jumped to 2.19 s.
  Moving to 1:16 at the same band did not improve completion at all. Both
  spent about 59% of contended time on generation rather than the requested
  88.9% or 94.1%. In the reference replay both selected the same 76 pure
  generation and seven mixed rounds starting under contention; six mixed
  decisions were subject to the overdue-request guard. A target is not a
  guaranteed or measured hardware share.

Lag is owed service, **not an absolute time between tokens**. The extra
gap metric is the longest time between output deliveries after first token
on any short stream; for this simulation multiple accepted tokens arrive
together at forward completion. Long TTFT is the worse of the two long
requests, including its admission wait. The six recordings are repeats of
one workload shape, not evidence of a universal optimum.

[324 per-run results](2026-11-26-fsched-time-splits.csv), including actual
generation share, maximum long TTFT and maximum short inter-delivery gap.
All 144 overlapping runs reproduce the four archived metrics from the
[earlier grid](2026-11-26-fsched-profile-grid.md) exactly. Reproduce with
`python3 tools/fsched_profile_grid.py --time-splits` (optional
`--csv /path/to/new.csv`).
