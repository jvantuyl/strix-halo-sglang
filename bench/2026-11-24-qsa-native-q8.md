# Native Q8 QSA QKV serving trial

[Patch 54](../patches/54-qsa-native-q8.md) replaced only the 12 text target
QSA QKV projections. The trial image added this patch to the verified
dev53 serving image. Both runs used the same checkpoint, FP8 KV, context
262,144, request cap 12, and three-step NEXTN. The draft was unchanged.
The BF16 service ran first; Q8 ran on a fresh boot on port 18038.

All packed weights dequantized bit-for-bit to the BF16 weights from the
[unweighted Q8 quality trial](2026-11-23-qsa-quality.md). The GPU check
passed exact dequant for all 65,536 code/scale pairs, compared M=0–17
against BF16 linear on the same dequantized weights, and verified graph
replay at M=1, 4, 16, and 17. The isolated test emitted a TunableOp
cache-path warning because the service's cache mount was not used there.
The patch stack replay passed 107/107 checks. Live startup logged 12
replacements and both runs completed all requests without errors.

## Serving results

Medians of three passes of `tools/bench_qwen38.py` at the listed target
prompt sizes. Single-stream rates exclude time to first token. Concurrent
rates are aggregate and include prefill.

| Workload | BF16 tok/s | Q8 tok/s | Change |
| --- | ---: | ---: | ---: |
| Single, 128 | 23.35 | 24.69 | +5.7% |
| Single, 2,048 | 17.85 | 21.89 | +22.6% |
| Single, 8,192 | 22.99 | 22.25 | −3.2% |
| Single, 32,000 | 20.95 | 21.52 | +2.7% |
| 4 concurrent | 43.64 | 46.68 | +7.0% |
| 8 concurrent | 61.60 | 62.66 | +1.7% |
| 12 concurrent | 75.79 | 76.18 | +0.5% |

Single-stream median time to first token changed by less than 1% at every
length. Mean NEXTN acceptance was 2.55 BF16 and 2.57 Q8 (30 scheduler
samples per arm). GPU free memory was 4,886 vs. 5,028 MiB at idle and
4,044 vs. 4,144 MiB at the measured minimum. Host available memory had
similar minima (5,297 MiB BF16, 5,395 MiB Q8) but different end readings
(6,308 MiB BF16, 5,486 MiB Q8). Different boot and cache states make
host-memory attribution unreliable.

These are sequential runs, not interleaved boots. The 2,048-token BF16
decode rate ranged from 17.54 to 20.98 tok/s, so the large median gain
should not be treated as a general improvement. At cap 12 the aggregate
gain was smaller than run-to-run variation. The 8k regression was present
in all three passes of this run.

## Quality and disposition

Both scored 40/51. On the 12 added Python traces both scored 9/12, with
one gain (odd-number sum) and one loss (shared list through
`dict.fromkeys`) for Q8. Both passed all seven retrieval cases, including
the new two-answer case at 64,124 input tokens. Q8 matched the earlier
BF16-compute Q8 logprob result: +0.002522 nats/token vs. BF16 and
186/195 top-1 matches. The original eight Python tasks still have a floor
effect. The sample is not evidence of unchanged general coding quality.

Native Q8 is a usable opt-in candidate, with a small measured VRAM gain,
roughly unchanged prefill, and workload-dependent decode changes. It is
not a blanket speedup. The trial restored and independently verified the
original BF16 QSA service on port 8000 with three-step NEXTN.

Raw data: [BF16 speed](2026-11-24-qsa-native-bf16-speed.jsonl),
[Q8 speed](2026-11-24-qsa-native-q8-speed.jsonl),
[BF16 quality](2026-11-24-qsa-native-bf16-quality.jsonl),
[Q8 quality](2026-11-24-qsa-native-q8-quality.jsonl).
Recompute the paired quality stats with
`tools/qsa_quality_gauntlet.py compare --extended BASELINE CANDIDATE`.
