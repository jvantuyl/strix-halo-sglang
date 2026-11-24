# ROCmFPX Q8 dense smoke test on gfx1151

The [ROCmFPX reference](https://github.com/charlie12345/ROCmFPX/tree/fb08d7cdb67082ddda15a25e64dc7ec413afc41c/ggml/rocmfpx)
stores 32 signed int8 codes and one unsigned E4M3 scale (8.25 bits per
weight). The benchmark separates the codes and scale bytes for aligned
loads and dequantizes to BF16 inside the Triton GEMM.

Measured in the verified `rocpro10`/dev53 image on the idle Strix Halo GPU,
with real stock BF16 weights, synthetic activations, and the production
TunableOp cache mounted read-only (tuning off). Each row uses the median
of five alternating graph-replay timings, after screening 27 tiles. The
dequant check covered all 65,536 code/scale pairs exactly. Each tile was
checked against BF16 linear on dequantized weights at sampled output rows.

| Projection (N × K) | M=1 BF16 → Q8 (µs) | M=4 BF16 → Q8 (µs) | M=16 BF16 → Q8 (µs) |
|---|---:|---:|---:|
| GDN input, 16384 × 2560 | 367.4 → 317.0 | 355.1 → 322.4 | 359.7 → 351.5 |
| QSA QKV, 13312 × 2560 | 291.2 → 256.6 | 291.3 → 254.6 | 295.2 → 260.9 |
| LM head, 248320 × 2560 | 5233.9 → 4574.1 | 5261.5 → 4614.0 | 5346.6 → 4711.1 |
| GDN output, 2560 × 6144 | 39.6 → 121.8 | 38.4 → 120.7 | 40.8 → 124.5 |
| QSA output, 2560 × 6144 | 39.3 → 121.4 | 39.1 → 121.3 | 40.9 → 123.8 |
| Shared expert gate/up, 1280 × 2560 | 11.2 → 27.6 | 10.8 → 28.0 | 11.9 → 28.0 |
| Shared expert down, 2560 × 640 | 8.0 → 14.6 | 7.9 → 14.5 | 7.1 → 14.7 |

This is a **go for further selective Q8 investigation**, not for blanket
dense conversion. The largest input projections and the LM head win; the
remaining measured shapes lose. The harness measures isolated linear calls,
not the model's mixed working set. Small matrices cycle through four copies,
but cache behavior can still differ from a full forward pass. The selected
M values cover single-token decode and small NEXTN verifies, not the maximum
verify batch or prefill. No serving throughput, FP4 expert speed, or model
quality was measured. In particular, LM-head quantization needs a quality
check before deployment; matching the dequant reference does not establish
parity with the original BF16 checkpoint.

[Raw timings](2026-11-23-rocmfpx-q8-dense.jsonl) ·
[Benchmark harness](../tools/bench_rocmfpx_q8_dense.py)
