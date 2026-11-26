# Profile scheduler grid: 2k arm, pass 1 (18 short + 2 long)

48 paired settings; one-token mixed path, fixed 2.7 s cap, cap 12, seed 7.
Each column sets a prefill:generation **service-time entitlement** during
contention, not a token ratio. The per-round switching band is the indicated
multiple of the generation debt a feasible full verify could repay. Other
settings are unchanged from the [profile comparison](2026-11-26-fsched-profiles.md).
Both tables are for the same reference replay, not averages over traces.
Short completion is arrival to finish (queueing included); lag is the maximum
earned-but-unserved time since service over every request and round.

### Median short-request completion (wall seconds)

| Band / discount | 1:1 | 2:1 | 3:1 | 4:1 | 5:1 | 6:1 | 7:1 | 8:1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25× | 30.29 | 30.61 | 30.61 | 31.93 | 30.70 | 30.70 | 30.70 | 30.70 |
| 0.5× | 30.29 | 30.29 | 31.49 | 31.49 | 31.49 | 32.89 | 32.89 | 32.89 |
| 1× | 30.29 | 30.29 | 31.49 | 31.49 | 31.49 | 32.89 | 35.32 | 35.32 |
| 2× | 30.29 | 30.29 | 31.49 | 32.89 | 35.32 | 36.45 | 36.45 | 36.45 |
| 3× | 30.29 | 31.49 | 32.89 | 35.32 | 36.45 | 36.45 | 36.45 | 36.45 |
| 4× | 30.29 | 32.89 | 35.32 | 36.45 | 36.45 | 36.45 | 36.45 | 36.45 |

### Worst lag since service (owed service seconds)

| Band / discount | 1:1 | 2:1 | 3:1 | 4:1 | 5:1 | 6:1 | 7:1 | 8:1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25× | 0.583 | 0.429 | 0.340 | 0.299 | 0.248 | 0.156 | 0.159 | 0.162 |
| 0.5× | 0.583 | 0.396 | 0.494 | 0.337 | 0.354 | 0.963 | 0.990 | 0.922 |
| 1× | 0.583 | 0.396 | 0.494 | 0.337 | 0.354 | 0.963 | 0.814 | 0.832 |
| 2× | 0.583 | 0.396 | 0.494 | 0.876 | 0.759 | 0.790 | 0.814 | 0.832 |
| 3× | 0.583 | 0.714 | 0.800 | 0.715 | 0.759 | 0.790 | 0.814 | 0.832 |
| 4× | 0.583 | 0.904 | 0.649 | 0.715 | 0.759 | 0.790 | 0.814 | 0.832 |

**At the existing 80/20 entitlement:** keep 4:1 and try 0.5×. Its two outputs
were identical to 1× across these six replays. Against 2× it improved both
metrics on the reference: 32.89 → 31.49 s completion and 0.876 → 0.337 s lag.
Whole-workload wall time rose by 0.50 s, from 122.03 to 122.53 s.

**If changing the entitlement is acceptable:** 2:1 at 0.5× tied for fastest
short completion and kept lag below 0.5 s on all six bursts (completion
30.29–30.44 s; lag 0.396–0.429 s). At 6:1 and 0.25×, lag was smallest
(0.155–0.156 s), with completion 30.70–30.86 s. Lag is relative to the
configured entitlement: 6:1 promises generation 14.3% of contended time,
not the 33.3% promised at 2:1. Lower service lag need not mean a shorter
absolute wait.

[All 288 runs, including wall time and actual generation share](2026-11-26-fsched-profile-grid.csv).
The six 4:1, 2× control cells reproduce the previous CSV's completion, lag,
wall time, and generation share to its stored precision. All runs processed
52,964 physical prefill tokens and delivered the same 3,728 output tokens.
The six bursts use the same workload shape; they are not six independent
workload types or hardware measurements of these settings.

Reproduce from the repo root with `python3 tools/fsched_profile_grid.py`
(prints the two tables). Add `--csv /path/to/new.csv` to save all runs.
