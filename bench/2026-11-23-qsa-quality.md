# Qwen3.8-Flash-Next QSA QKV quality gauntlet

This is a paired smoke test, not a general quality benchmark. The baseline
is the serving AWQ int4 checkpoint with BF16 QSA QKV. The variants change
only the 12 fused `model.layers.{3,7,…,47}.qkv_proj.weight` tensors:

- **BF16:** original weights.
- **Q8:** `Q8_0_ROCMFPX` reference encoding, decoded back to BF16.
- **FP4:** `Q4_0_ROCMFP4_FAST` reference encoding, decoded back to BF16.

Both variants use the pinned [ROCmFPX C reference](https://github.com/charlie12345/ROCmFPX/tree/fb08d7cdb67082ddda15a25e64dc7ec413afc41c)
without an importance matrix. The [overlay builder](../tools/prepare_qsa_quality.py)
verifies all bytes outside the 12 intended tensor ranges against the
original presharded files. Vision, the draft, QSA output and indexer, GDN,
the LM head, and all other weights are unchanged. All variants execute
the same BF16 QSA linear path, so observed differences are not attributable
to new quantized GPU kernels. This test cannot measure quantized throughput.

The [gauntlet](../tools/qsa_quality_gauntlet.py) uses greedy generation with
thinking disabled, exact answer scoring, and fixed prompts. It includes
eight arithmetic, eight Python tracing, eight JSON, eight reading, and
four retrieval tasks (two at 7,463 tokens and two at 29,543). Separately,
it teacher-forces 195 continuation tokens across four short passages
and pairs logprob and top-1 comparisons on identical token IDs.

## Results

| Metric | BF16 | Q8 | FP4 |
| --- | ---: | ---: | ---: |
| Correct | 28/36 | 28/36 | 27/36 |
| Arithmetic | 8/8 | 8/8 | 7/8 |
| Python tracing | 0/8 | 0/8 | 0/8 |
| JSON / reading / retrieval | 8/8 / 8/8 / 4/4 | 8/8 / 8/8 / 4/4 | 8/8 / 8/8 / 4/4 |
| Same generated text as BF16 | 36/36 | 29/36 | 28/36 |
| Mean NLL (nats/token) | 1.395590 | 1.398113 | 1.399143 |
| Paired NLL change | 0 | +0.002522 | +0.003552 |
| Top-1 agreement with BF16 | 195/195 | 186/195 | 185/195 |

FP4 answered `3692` rather than `3656` on `arithmetic-2`. No other
correctness flips occurred. The Python tasks have a floor effect: BF16
genuinely got all eight wrong, so the tie is not evidence of preserved
coding quality. Four successful retrieval probes do not establish quality
across long-context workloads. The 195 scored tokens are too few to treat
the small Q8–FP4 NLL gap as a general ranking.

Raw results: [BF16](2026-11-23-qsa-quality-bf16.jsonl),
[Q8](2026-11-23-qsa-quality-q8.jsonl), and
[FP4](2026-11-23-qsa-quality-fp4.jsonl). Recompute the paired statistics
using `tools/qsa_quality_gauntlet.py compare BASELINE CANDIDATE`.

Q8 is the better first candidate for a native QSA-only implementation:
no observed correctness losses here, plus a decode-shape latency gain in
the [dense microbenchmark](2026-11-23-rocmfpx-q8-dense.md). Neither result
establishes end-to-end performance or quality. Test importance-weighted FP4
as a separate variant, after confirming calibration coverage and the fused
Q/K/V mapping. Leave Q8 unweighted unless a broader test shows meaningful
drift. The original BF16 QSA service was restored, with HTTP 200 on port
8000 and NEXTN reported as three-step EAGLE.

## Importance-weighted FP4 follow-up

Use the [Unsloth Qwen3.8-Flash-Next matrix](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/c8b5954a88c2775c546b92593eda40ea041d3176/imatrix_unsloth.gguf_file),
SHA-256 `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`.
The GGUF metadata names `unsloth_calibration_Qwen3.8-Flash-Next.txt`.
All 36 Q/K/V input vectors have 2,560 finite positive values after dividing
the stored squared activation sums by their counts, as the pinned quantizer
loader does. Each vector reports a count of 845,728.

The builder's `--imatrix` option uses the `gguf` Python package from the
pinned reference checkout (`PYTHONPATH=<checkout>/gguf-py`). It applies the
three per-layer vectors to the TP=1 runtime's consecutive gated-Q, K, and V
row ranges (12,288, 512, and 512 rows). The source is the same original BF16
QKV as in the first run. All 12 weighted spans differ from unweighted FP4,
and every other byte matches the original dump. This calibration came from
the GGUF publisher, not from our serving AWQ checkpoint. The comparison
still uses the same BF16 compute and does not measure speed.

| Metric | BF16 | Q8 | FP4 | Weighted FP4 |
| --- | ---: | ---: | ---: | ---: |
| Correct | 28/36 | 28/36 | 27/36 | 28/36 |
| Arithmetic | 8/8 | 8/8 | 7/8 | 8/8 |
| Same generated text as BF16 | 36/36 | 29/36 | 28/36 | 30/36 |
| Paired NLL change vs. BF16 (nats/token) | 0 | +0.002522 | +0.003552 | +0.015205 |
| Top-1 agreement with BF16 | 195/195 | 186/195 | 185/195 | 186/195 |

Weighted FP4 recovered `arithmetic-2` (`3656`) without any new correctness
losses. The mean NLL increase over unweighted FP4 was +0.011653
nats/token. This was not uniform: weighted FP4 had lower NLL than unweighted
FP4 on prose and code, but higher NLL on technical and structured text.
The 49-token structured passage drove the net increase (+0.062261 vs.
BF16, compared with +0.003316 for unweighted FP4). The JSON generation
tasks still passed 8/8. The coding floor and small logprob sample still apply.

[Weighted FP4 raw results](2026-11-24-qsa-quality-fp4-weighted.jsonl).
The original service was verified again on port 8000 (HTTP 200,
three-step EAGLE, trial port closed). Keep Q8 as the first native QSA
candidate. If pursuing FP4, compare both encodings on a larger held-out
set before choosing; recovering one arithmetic answer does not establish
an overall quality win.
