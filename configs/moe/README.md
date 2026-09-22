# Tuned MoE Triton configs for gfx1151

Upstream ships no fused-MoE tile configs for `Radeon_8060S_Graphics`, so the
kernel falls back to a generic shape (`BLOCK_SIZE_N=128, num_warps=4` for
large M; `16/32/64` for small M). The files here are baked into the image by
the Dockerfile, under the config directory of whatever Triton version the
image installs, so they apply without a mount. To try a config without
rebuilding, mount this directory over that path instead:

```
-v $(pwd)/configs/moe:/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_0
```

Keys must be integers (batch sizes); the loader does `int(key)` on every
key, so no comment fields.

## `E=256,N=256` (Qwen3.5-35B-A3B)

Hand swept. At batch size 1 the fused MoE GEMM is bound by how much of the
GPU it can keep busy, not by tile efficiency. `BLOCK_SIZE_M` is irrelevant
(only one row tile exists at decode). `BLOCK_SIZE_N` and `num_warps` together
decide workgroup count, and the upstream defaults launch roughly 64
workgroups across a 40-CU part.

Measured end-to-end on Qwen3.5-35B-A3B, single stream:

| BLOCK_SIZE_N | num_warps | tps |
|---:|---:|---:|
| 128 (upstream default) | 4 | 16.3 |
| 32 | 4 | 34.5 |
| 16 | 4 | 38.0 |
| **16** | **2** | **39.7** |
| 16 | 1 | 35.7 |
| 16 | 8 | 29.7 |

`num_warps=1` is too few to cover memory latency; 8 oversubscribes. 2 is the
optimum here. The `_down` file is a copy: both kernels share one
`moe_align_block_size` sort, so the down config must use the same
`BLOCK_SIZE_M`.

## `E=512,N=320` (Qwen3.8-Flash-Next, AWQ int4 group 32)

Produced by [`tools/tune_moe_gfx1151.py`](../../tools/tune_moe_gfx1151.py),
which drives upstream's `benchmark/kernels/fused_moe_triton` tuner with a
gfx1151 search space (1,560 configs: `BLOCK_M` 16–256, `BLOCK_N` 16–256,
`BLOCK_K` 16–128, 1–8 warps, `GROUP_SIZE_M` 1/8/32, `waves_per_eu` 0/2,
pruned to 64 KB LDS and at most 128 accumulators per lane) and an early
bail-out: one graph replay is timed right after capture and the config is
dropped if it is already 3× slower than the best so far (a single eager run
rejects the grossly slow ones before capture). 35–60% of configs bail at
every M, which brings the full 18-size sweep to about 7 GPU-hours; each M
runs in its own process with its result saved as it finishes, and compiled
kernels stay in the Triton cache, so an interrupted run resumes at replay
speed.

Reproduce inside the image (GPU attached, server may stay up):

```bash
docker cp tools/tune_moe_gfx1151.py sglang-qwen38:/tmp/
docker exec -w /tmp sglang-qwen38 python3 /tmp/tune_moe_gfx1151.py \
    --model /models/qwen38 --dtype int4_w4a16 --disable-shared-experts-fusion --tune
docker cp "sglang-qwen38:/tmp/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" configs/moe/
```

Results are in [`docs/RUNNING_QWEN38.md`](../../docs/RUNNING_QWEN38.md)
(MoE tile tuning section).
