# Tuned MoE Triton configs for gfx1151

Upstream ships no fused-MoE tile configs for `Radeon_8060S_Graphics`, so the
kernel falls back to a generic shape (`BLOCK_SIZE_N=128, num_warps=4` for
large M; `16/32/64` for small M). The configs live here, one directory per
checkpoint they were tuned on, and are **mounted at run time**, not baked
into the image: the launchers and `compose.yaml` bind the chosen profile at
`/moe-configs` and set `SGLANG_MOE_CONFIG_DIR=/moe-configs`, which
[patch 17](../../patches/17-moe-config-dir.md) turns into a search path
checked before upstream's builtin tree (flat files or the
`configs/triton_x_y_z/` layout; several directories may be joined with `:`).

| Profile | Tuned on | Launcher default for |
|---|---|---|
| [`qwen3.5-35b-a3b/`](qwen3.5-35b-a3b/) | `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (`E=256,N=256`, int4 g32) | `start-sglang.sh`, compose `sglang` (`MOE_CONFIG_DIR`) |
| [`qwen38-flash-next/`](qwen38-flash-next/) | `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (`E=512,N=320`, int4 g32) | `start-qwen38.sh`, compose `qwen38` (`QWEN38_MOE_CONFIG_DIR`); also the recommended profile for the DERISKED g128 checkpoint, see below |
| [`qwen38-flash-next-derisked/`](qwen38-flash-next-derisked/) | `davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ` (`E=512,N=320`, int4 g128) | none: wins the tuner's microbenchmark but loses 7–9% end to end, kept as data (see below) |

The file name is keyed on expert count, `N` and dtype only (`block_shape`
`[0, group]` is dropped because of the leading zero), so a config tuned for
another group size or checkpoint of the same shape gets the **same name**;
keeping one directory per checkpoint is what stops them overwriting each
other and keeps each model's tuning data separate. `MOE_CONFIG_DIR=` (empty)
runs on upstream's generic tiles, e.g. for a before/after measurement.

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
docker cp "sglang-qwen38:/tmp/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" \
    configs/moe/qwen38-flash-next/
```

**`--tp-size` and the shape the tuner measures.** The runtime keys the file
on the int4-*packed* width: the real intermediate is 640, stored as 320
int32 columns, hence `N=320`. The tuner instead computes `N` as
`intermediate / tp_size`, so its default `--tp-size 2` produces the right
file *name* (`N=320`) but benchmarks a 320-wide shard, half the work the
runtime kernel does. `--tp-size 1` measures the true shape and writes an
`N=640` file that nothing reads; rename (or copy) it to `N=320` to use it.
The tile choices below were all swept at the default `--tp-size 2` (half
shape); a true-shape sweep has not been run, so it is unknown whether it
would pick different tiles. For another checkpoint of the same shape (e.g.
a g128 requant) write into a new profile directory and point
`MOE_CONFIG_DIR` / `QWEN38_MOE_CONFIG_DIR` at it. The tuner's benchmark
mode (without `--tune`) times whichever config the runtime would pick, so
`SGLANG_MOE_CONFIG_DIR=/moe-configs` vs unset gives a direct tuned vs
generic comparison; add `--tp-size 1` (with the configs copied to an
`N=640` name) for true-shape numbers.

Results are in [`docs/RUNNING_QWEN38.md`](../../docs/RUNNING_QWEN38.md)
(MoE tile tuning section).

## `E=512,N=320` DERISKED (AWQ int4 group 128)

Same tuner, same 1,560-config space, run on the
`davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ` conversion (group size 128
instead of 32). Full 18-size sweep took 8 h 3 min wall (M=1 alone 34 min;
M=8 37 min because of a compile burst on `num_warps=1` large-`BLOCK_M`
configs; the later sizes ~55 min each). The sweep ran in a standalone
container with no server up; the Triton cache is host-mounted so recompiles
are avoided across runs:

```bash
docker run --name tune-derisked --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt seccomp=unconfined \
  -v /opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8:/models/qwen38:ro \
  -v "$HOME/.cache/strix-halo-sglang-cache:/root/.cache/sglang" \
  -v "$PWD/tools/tune_moe_gfx1151.py:/tune_moe_gfx1151.py:ro" \
  -v "$HOME/tune-derisked:/work" -w /work \
  -e SGLANG_USE_AITER=0 -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
  -e PYTORCH_TUNABLEOP_TUNING=0 -e MOE_TUNE_PARTIAL_DIR=/work/partial \
  strix-halo-sglang:dev \
  python3 /tune_moe_gfx1151.py --model /models/qwen38 --dtype int4_w4a16 \
      --disable-shared-experts-fusion --tune
cp "$HOME/tune-derisked/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" \
   configs/moe/qwen38-flash-next-derisked/
```

Winning tiles (`M x N x K`, `GROUP_SIZE_M`, warps, `waves_per_eu`; all
`num_stages=2`):

| M | g32 profile | g128 profile |
|---:|---|---|
| 1 | `16x32x64 g8 w1 wpe2` | `16x32x32 g32 w1 wpe2` |
| 2–512 | `16x16x64 w1 wpe2` | `16x32x64 w1 wpe2` |
| 1024, 1536 | `32x16x64 g1 w1 wpe2` | same |
| 2048 | `64x16x32 g1 w1 wpe2` | same |
| 3072 | `64x32x32 g1 w2 wpe2` | `64x128x32 g1 w4 wpe0` |
| 4096 | `128x64x32 g1 w4 wpe0` | same |

The g128 checkpoint prefers a `BLOCK_N` of 32 rather than 16 at small M
(fewer scale loads per tile with the wider group), and the two agree from
1024 up except at 3072. Best times track the g32 sweep (M=1 97 µs vs 103,
M=2 171 vs 200).

**A/B, tuner benchmark mode on the g128 checkpoint** (µs per fused-MoE
call, batch 1 / 8 / 16 / 20 / 32):

| Profile | `--tp-size 2` (half shape) | `--tp-size 1` (true shape) |
|---|---|---|
| upstream generic | | 438 / 3066 / 5685 / 6829 / 9794 |
| g32 profile | 98.2 / 688 / 1243 / 1505 / 2124 | 179 / 1390 / 2544 / 3037 / 4340 |
| g128 profile | 100.7 / 646 / 1174 / 1379 / 1937 | 191 / 1285 / 2339 / 2780 / 3942 |

Microbenchmark: g128 tiles 6–9% faster at batch ≥ 8, 2–7% slower at
batch 1.

**A/B, end to end** (derisked server, patches ≤ 21 image, runs back to
back, same prompts as the runbook):

| | g32 profile | g128 profile (two runs) |
|---|---|---|
| decode tok/s, 183 / 2634 / 10998-token prompts | 14.80 / 13.94 / 14.00 | 14.73 / 13.79 / 13.70 and 14.63 / 11.59 / – |
| prefill tok/s, same prompts | 318 / 769 / 712 | 300 / 693 / 712 |
| aggregate decode, 8 / 16 / 20 streams | 46.5 / 69.0 / 70.2 | 41.8 / 62.4 / 65.4 and 43.3 / 62.9 / 64.3 |

End to end the g128 tiles are **7–9% slower** at 8–20 streams and no better
single stream, the opposite of the microbenchmark. The tuner routes tokens
uniformly across experts and (see above) times half the real GEMM width;
real routing is skewed, so per-expert row counts at a given batch size are
not what the tuner's M stands for. That is the working hypothesis, not a
verified cause. Until a true-shape or traced-routing sweep says otherwise
the DERISKED launcher keeps the g32 profile (`qwen38-flash-next/`); this
directory is kept so the result is reproducible and the data is not lost.
Determinism was checked with each profile (96-token and 8.6k-token greedy
repeats bit-identical, 3/3 each); output between the two profiles was not
compared, and a different `BLOCK_K` can change the reduction order, so it
need not match.
