# Larger prefill blocks on Strix Halo

**Keep 1,024 tokens per block.** The first larger arm crossed the 2 GiB
free-VRAM guard, so the block-size ladder stopped. No serving or tuning
settings changed.

The verified baseline and candidate used loopback HTTPS on port 18038
with production port 8000 closed during measurement. Other services stayed
running. Both used the deployed Q8 QSA sidecar, the corrected INT4 draft,
three-step NEXTN, the 2M-row PLE hotset, FP8 KV, cap 12, 262,144 context,
and the 524,288-token pool cap. The existing rocBLAS 5.6 cache was left in use.

| Measurement | 1,024 tokens | 16,384 tokens |
| --- | ---: | ---: |
| Isolated 31,992-token prompt, TTFT | 49.00 s | 69.41 s |
| Isolated prefill rate (prompt tokens / TTFT) | 653 tok/s | 461 tok/s |
| Scheduler prefill reports in that request | 32 | 2 |
| Gap between consecutive prefill reports, p50 / max | 1.44 / 3.10 s | 32.14 / 32.14 s (one gap) |
| GPU busy %, sampled median during that prefill | 95% | 100% |
| Minimum free VRAM sampled across arm | 4,118 MiB | 989 MiB |

The larger arm's first cold long prompt was enough to fail the memory
guard. Its throughput and progress cadence were also worse in that
probe; there are no steady-state or mixed-load results for that size.
The 8,192-token and mixed-chunk arms were not run after the guard fired.
The current source allowlists EAGLE for mixed chunk, but actual
NEXTN mixed-forward compatibility on this deployment remains untested.

The 1,024-token baseline completed three timing passes without request
errors: median 713 tok/s at the 32k point and 5.79 tok/s across the 54
short streams in the mixed workload. In a separate forced-overlap probe,
the longest observed client delivery gap was 1.72 s. The quality
gauntlet reproduced the deployed Q8 profile (40/51; +0.002522 nats/token
versus BF16 over 195 tokens).

## Where the time went

In the separate four-step profile of the 1,024-token baseline, the union
of five CPU `scheduler.run_batch` spans occupied 6.062 of 6.186 seconds
between the first span's start and the last span's end. The 0.124 s
outside those scopes was small compared with the work inside. On the
device timeline, kernels, copies, and sets covered 5.243 of 5.460 seconds
between first and last observed device activity. These are different
trace spans, not an additive account of the full request. GPU busy
indicates engine activity, not compute occupancy or useful throughput.

The largest summed kernel times in the trace were routed AWQ MoE
(`fused_moe_kernel_gptq_awq`, 2.575 s) and sparse QSA prefill
(`_sparse_gqa_chunk_prefill`, 0.414 s), out of 5.236 s of summed kernel
time. That points to work inside the forwards rather than primarily to
chunk boundaries. This capture does not attribute the idle gaps to PLE
hotset churn, so no hotset change was made. No larger chunk qualified for
additional TunableOp or MoE tuning. Context, pool, and KV dtype were kept.

The first attempt accidentally omitted the corrected draft-model path
override in the trial environment and measured acceptance of 1.00; its
timings are excluded. The runner then verified the effective draft path
and exact draft dump at boot, and required an acceptance smoke test
before running the timed workload. A run interrupted after its timing
passes by a stale idle snapshot is also excluded.

[Raw timings and isolated-request telemetry](2026-11-25-prefill-blocks.jsonl).

## Q8 prefill projection

The separate [`bench_qsa_q8_prefill.py`](../tools/bench_qsa_q8_prefill.py)
benchmark uses a hash-verified deployed QSA sidecar layer and compares the
current unpack-to-BF16 plus `F.linear` path with tiled Triton Q8 matmul.
Run it in the patch-54 ROCm image with the weights mounted read-only,
TunableOp tuning off, and exclusive GPU access:

```sh
python3 /bench-tools/bench_qsa_q8_prefill.py /qsa-q8 \
  --m 1024 4096 16384 --out /results/qsa-q8-prefill.jsonl
```

Eager timing is the primary comparison because production prefill graphs
are disabled. The rocBLAS 5.6 cache was mounted read-only. TunableOp was
enabled, but the benchmark did not record an active results filename or
confirm cache hits. Median eager times for the best of 12 tiles at each size:

| Token rows | Unpack + `F.linear` | Fused Q8 | Fused latency |
| ---: | ---: | ---: | ---: |
| 1,024 | 2.10 ms | 3.42 ms | +63% |
| 4,096 | 7.02 ms | 12.88 ms | +84% |
| 16,384 | 26.95 ms | 51.05 ms | +89% |

The fused path saved 65 MiB of peak allocated scratch per call, but
missed the 5% speedup gate at every size. Exact dequantization passed
for all 65,536 code/scale pairs and 34,078,720 weights in the selected
layer; maximum relative L2 across matmul outputs was 0.000028 and
graph replay was exact. No serving gauntlet was run for the fused path
because it failed the speed gate. Keep the current sidecar and BF16
prefill compute. [Raw microbenchmark results](2026-11-25-prefill-q8.jsonl).

## Follow-up: 2,048 and 4,096 with mixed chunk and NEXTN

The original 2,048-token arm failed with an HSA queue exception at the
first four-stream workload. A bounded replay reached the target's first
MIXED forward (three 52-token prefill rows and one decode token) but
never returned from it. The pinned compressed QSA planner assumes all
extend prefixes are ratio-aligned, which need not hold for a decode row
in a mixed batch. [Patch 55](../patches/55-qsa-cross-prefix.md) handles
groups that cross that boundary. [Patch 56](../patches/56-mixed-mamba-checkpoint.md)
separately preserves non-lazy Mamba checkpoint writes across the merge.
The instrumented four-stream replay on the candidate image finished
all 800 generated tokens and confirmed actual MIXED execution.

The candidate image then completed three full HTTPS passes at each size
without request errors. NEXTN, context, pool, KV dtype, and cap were
unchanged. These are serving comparisons to the original 1,024 baseline,
not an isolated estimate of the effect of either patch.

| Metric | 1,024, mixed off | 2,048, mixed on | 4,096, mixed on |
| --- | ---: | ---: | ---: |
| 32k prefill, median tok/s (3 runs) | 713.32 | 758.09 (+6.3%) | 478.11 (−33.0%) |
| Mixed short decode, pooled p50 tok/s (54 streams) | 5.79 | 6.22 (+7.4%) | 5.34 (−7.8%) |
| Longest client delivery gap in forced overlap | 1.72 s | 2.75 s | 8.90 s |
| Minimum sampled free VRAM | 4,118 MiB | 4,117 MiB | 3,693 MiB |
| Quality score / NLL delta vs BF16 | 40/51 / +0.002522 | 40/51 / +0.002522 | 40/51 / +0.002522 |

Decode-only acceptance matched the baseline on all three probes in both
arms. The longest 32k scheduler gap ending in prefill was 2.80 s at 2k
and 9.14 s at 4k; these post-step gaps are proxies, not kernel timings.
Both arms passed the headroom and quality gates, but neither met the
15% throughput gate. The 4k arm also missed the mixed-short throughput
target. No serving settings or tuning changed, and production was left
offline as requested. The quality gauntlet is not a dedicated mixed
prefix-cache state equivalence test.

[Raw follow-up timing and gate records](2026-11-25-prefill-mixed.jsonl).
