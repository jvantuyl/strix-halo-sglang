# Running Qwen3.8-Flash-Next on Strix Halo

Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) is a 48-layer hybrid
GDN + sparse-attention (QSA) MoE with a vision tower and a 51B-parameter
n-gram "PLE" embedding table. The resident part of the AWQ-INT4 checkpoint is
~75 GiB; the PLE table alone is 95 GiB in bf16. It fits a 128 GB Strix Halo
only because SGLang can keep the PLE table **off the GPU and out of RAM**,
reading rows on demand from a file. This page is the runbook for that setup
on the fork image (`strix-halo-sglang:dev`).

Tested checkpoint: `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (compressed-tensors,
group 32, asymmetric; vision tower unquantized). MTP is not used.

## What the image adds

| Patch | What | Why on gfx1151 |
|---|---|---|
| [11](../patches/11-qwen4-exp-rocm.md) | CPU-side PLE gather, Triton QSA decode route, safe top-k fallback, QSA smem schedule | GPU must not touch host memory; SM121-only kernel gates; JIT top-k reads OOB; 64 KB workgroup smem cap |
| [12](../patches/12-wna16-triton-zp.md) | Pass zero points to the Triton WNA16 MoE kernel; load-time zp transpose; GC between MoE layers in post-load | Asymmetric AWQ; without it every expert weight is off by `(8 - zp) * scale`. Without the GC the old expert weights of every converted layer stay allocated (~1.4 GiB each) and the 48-layer load OOMs |
| [13](../patches/13-ple-table-reuse.md) | Reuse the file-backed PLE table across boots | Upstream rewrites the 48 GiB table from the checkpoint on every start; a fingerprinted marker lets later boots skip the PLE shards |
| [14](../patches/14-cuda-graph-ple.md) | Decode CUDA graphs with the CPU-side PLE gather | The host gather cannot be captured; fill the static PLE prefetch buffer from the host before each replay |
| [15](../patches/15-qsa-graph-scratch.md) | Dedicated QSA packed-KV scratch for captured graphs | Upstream shares one growable scratch between graphs and eager decode; an eager step above the graph range re-allocates it and the graphs write into freed memory (GPU page fault after the next `empty_cache`). Not gfx1151-specific |
| [16](../patches/16-wna16-rocm-dense.md) | Dense compressed-tensors int4 Linear on ROCm: dequantize to bf16 at load, serve with `F.linear` | The dense WNA16 scheme is Marlin-only and Marlin is CUDA-only (`NameError: gptq_marlin_repack`); needed by checkpoints that also quantize attention `q/k/v/o`, e.g. the [abliterated variant](#running-the-abliterated-variant-derisked) |
| [17](../patches/17-moe-config-dir.md) | `SGLANG_MOE_CONFIG_DIR` searched before the builtin MoE tile tree (flat or tree layout, missing dirs skipped) | Upstream's knob replaces the tree and crashes on a missing dir; tuned tiles are now mounted per checkpoint instead of baked into the image |
| [18](../patches/18-hc-mix-rocm.md) | Atomics-free two-launch HyperConnection mix for decode batches, used on HIP | The sm_100 JIT mix is unavailable, so every decode step ran the persistent kernel whose split-K `atomic_add` made greedy decode differ run to run (and whose software grid barrier assumes co-resident CTAs). Same speed, bit-identical |
| [10](../patches/10-sleep-on-idle-default.md) | Idle scheduler sleeps | unchanged, re-anchored to the new `arg_groups` layout |
| [configs/moe](../configs/moe/) | Tuned fused-MoE Triton tiles for `E=512,N=320,int4_w4a16`, mounted at `/moe-configs` by the launchers | Upstream has no `Radeon_8060S_Graphics` configs; the generic tile is 2.2× slower at decode. See MoE tile tuning below |

Everything else (GDN, QSA prefill/indexer, HyperConnections, fused sigmoid-mul,
n-gram hashing) is pure Triton upstream and runs unmodified.

## Prerequisites

1. Image: `docker build -t strix-halo-sglang:dev .` (see
   [BUILDING.md](BUILDING.md); the Dockerfile pins upstream
   `70b5b03e78612c94f86ac98eb4d2d8d19ceda738`).
2. Kernel parity check on the box (no model needed, ~1 min):
   ```bash
   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
       --security-opt seccomp=unconfined \
       -v $PWD/tools/test_qwen38_rocm.py:/test.py:ro strix-halo-sglang:dev python3 /test.py
   ```
   Expect `ALL PARITY TESTS PASSED` (PLE gather bf16/fp8, QSA decode, top-k chain,
   MoE zero points incl. a negative control, dense WNA16 dequant, MoE config
   search path, deterministic HC mix).
3. Convert the PLE table to fp8 (halves the table to ~48 GiB and is the format
   the file backend expects to keep resident-free):
   ```bash
   docker run --rm -v /path/to/Qwen3.8-Flash-Next-AWQ-INT4:/src:ro \
       -v ~/models/Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8:/dst \
       -v $PWD/tools/convert_ple_fp8.py:/convert.py:ro \
       strix-halo-sglang:dev python3 -u /convert.py /src /dst
   ```
   Two streaming passes (per-table amax, then rewrite). Non-PLE files are
   hardlinked when source and destination share a filesystem, copied otherwise.
   Safe to rerun: finished files are skipped. Output ≈ 128 GiB.

## Launch

```bash
./start-qwen38.sh                                # port 30001, container sglang-qwen38
docker compose --profile qwen38 up -d qwen38     # same thing, detached
```

Both run:

```
python3 -m sglang.launch_server \
    --model-path /models/qwen38 \
    --ple-offload-embedding \
    --ple-offload-backend file --ple-offload-dir /ple \
    --mem-fraction-static 0.85 --context-length 131072 \
    --kv-cache-dtype fp8_e4m3 --max-total-tokens 262144 \
    --attention-backend triton \
    --cuda-graph-max-bs-decode 20 --max-running-requests 20 \
    --max-mamba-cache-size 100 --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3-thinking --tool-call-parser qwen3_coder
```

with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment.

with `SGLANG_FORCE_NATIVE_LAYERNORM=1 SGLANG_USE_AITER=0
SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 PYTORCH_TUNABLEOP_TUNING=0` in the
environment, `~/.cache/strix-halo-sglang-cache` mounted at
`/root/.cache/sglang` (SGLang's Triton / JIT kernel cache) so restarts do not
recompile every kernel, and [`configs/moe/qwen38-flash-next`](../configs/moe/)
mounted at `/moe-configs` with `SGLANG_MOE_CONFIG_DIR=/moe-configs` (the
tuned MoE tiles; `MOE_CONFIG_DIR` / `QWEN38_MOE_CONFIG_DIR` pick another
profile, empty disables it).

The first boot writes the 48 GiB PLE table and takes ~10 min; later boots
reuse it (patch 13) and skip the 128 PLE shards. First-request kernel
compiles add a few more minutes on a cold kernel cache. Keep the PLE directory
on local NVMe: it is random-read during decode.

### Why these flags

| Flag / env | Reason |
|---|---|
| `--ple-offload-embedding` | Must be explicit. Upstream defaults it on only for `is_cuda and dtype==bf16`; on ROCm it resolves to `False`, and `--ple-offload-backend file` refuses to start without it. |
| `--ple-offload-backend file --ple-offload-dir /ple` | The table becomes a sparse file-backed `mmap`; rows are read through the page cache on demand and the resident set is trimmed (8 GiB cap by default). The first boot writes the table (~48 GiB) and arms a completion marker (`<table>.complete.json`, fingerprinted by the checkpoint index and PLE shard sizes); later boots skip the PLE shards while the marker matches. Delete the marker to force a rewrite. |
| `SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1` | Upstream gates the file backend on `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (GB10). Irrelevant here: patch 11 gathers on the CPU, the GPU never dereferences the mapping. |
| `--attention-backend triton` | Same as every other model on this box; aiter's CK paths are CDNA-only. |
| `--context-length 131072` | The model is trained to 262,144 positions (no RoPE scaling needed). 131k costs nothing here: the pools are preallocated and the 8192-token prefill chunks bound activation size, so peak VRAM at a 120k prefill equals idle (88.9 GiB). Measured below: prefill flat at 450–480 tok/s to 125k tokens, decode 14.0 tok/s at 125k vs 14.5 short, exact needle recall at 120k. `QWEN38_CONTEXT` / `SGLANG_CONTEXT` override. |
| `--kv-cache-dtype fp8_e4m3 --max-total-tokens 262144` | fp8 KV halves the pool; the token cap keeps the saving as headroom instead of a larger pool (see Memory). `QWEN38_KV_DTYPE` / `QWEN38_MAX_TOTAL_TOKENS` override. |
| `--cuda-graph-max-bs-decode 20 --max-running-requests 20` | Decode graphs for bs 1, 2, 4, 8, 12, 16, 20; patch 14 fills the PLE prefetch buffer from the host before each replay. The two numbers are independent (`QWEN38_CUDA_GRAPH_MAX_BS`, `QWEN38_MAX_RUNNING_REQUESTS`); the default keeps them equal because graphs above bs 8 cost 0.3 GB and nothing else. They used to be tied because eager decode above the graph range faulted the next replay; patch 15 fixed that. Note this upstream split the flag: a bare `--cuda-graph-max-bs` is rejected as ambiguous. |
| `--max-mamba-cache-size 100` | The GDN layers keep a fixed-size recurrent state (conv window + SSM matrix) per request instead of per-token KV, in a pool counted in slots. With the radix cache on SGLang reserves 5 slots per request (3 for the live state, prefix-cache branch points and the prefill→decode handoff, plus 2 for the overlap scheduler's ping-pong buffer), so `max_running_requests = slots // 5`. The ratio-sized pool came out at 99 slots and silently capped the server at 19 requests (and the graph list at `[..., 16, 19]`); 100 makes the advertised 20 real. ~54 MB per slot in bf16, so ~270 MB per extra request. `QWEN38_MAMBA_CACHE_SIZE` overrides; keep it at 5 × `QWEN38_MAX_RUNNING_REQUESTS`. |
| `--mamba-ssm-dtype bfloat16` | The GDN recurrent state is fp32 by default (~108 MB per slot); bf16 halves it, so 100 slots cost 5.4 GB instead of 10.8. Upstream's own suggestion in the startup log. |
| `--reasoning-parser qwen3-thinking --tool-call-parser qwen3_coder` | The chat template opens `<think>` in the generation prompt and asks for `<tool_call><function=...><parameter=...>` tool calls. Without these the thinking and the XML come back as plain `content`. |
| `PYTORCH_TUNABLEOP_TUNING=0` | The image enables PyTorch TunableOp, which benchmarks every GEMM solution for each *new* M (= tokens in the prefill chunk). That is 14–20 s of TTFT for every novel prompt length (measured; the recorded results persist in `~/.cache/strix-halo-sglang-tunableop` so a repeated length is fast). With tuning off the recorded solutions are still used and untuned shapes take hipBLASLt's heuristic pick. `SGLANG_TUNABLEOP_TUNING=1 ./start-qwen38.sh` to deliberately record more. |
| `SGLANG_USE_AITER=0` | Set in the image. |

### What to look for in the log

```
Using CompressedTensorsWNA16TritonMoE (ROCm)
Using MoE kernel config from /moe-configs/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json.
PLE table: file-backed mmap /ple/ple_table_<rows>x160_float8_e4m3fn_..._rows0-<rows>.bin (47.x GiB, torch.float8_e4m3fn)
PLE table: WILLNEED prefetch on for gathers of >= 2048 rows (row = 160 B)
PLE table: resident set capped at 8.0 GiB, checked every 30 s
Using QSA for sparse full-attention layers.
QSA decode on HIP: using Triton qwen38_qsa kernel (heads=(12,1) head_dim=256)
```

If you see `QSA decode on HIP: using flash_attn varlen fallback`, the QSA
kernel's shape contract was not met; the fallback is correct but slow.

## Dummy-weights smoke test

To exercise the whole stack without the 175 GiB checkpoint (useful after
rebuilding the image), generate a shrunken config that keeps the production
kernel geometry and launch with `--load-format dummy`:

```bash
python3 tools/make_mini_qwen38_config.py /path/to/Qwen3.8-Flash-Next-AWQ-INT4 /tmp/qwen38-mini
docker run -d --name mini --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --ipc=host --network=host --security-opt seccomp=unconfined \
    -v /tmp/qwen38-mini:/models/mini:ro -v /tmp/ple-mini:/ple \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 -e SGLANG_USE_AITER=0 -e SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 \
    strix-halo-sglang:dev python3 -m sglang.launch_server --model-path /models/mini --load-format dummy \
    --host 0.0.0.0 --port 30002 --ple-offload-embedding --ple-offload-backend file --ple-offload-dir /ple \
    --mem-fraction-static 0.12 --context-length 4096 --attention-backend triton --cuda-graph-max-bs-decode 8
```

Output is noise (random weights) but every path runs: fp8 PLE file table,
WNA16 int4 MoE with zero points, QSA prefill + Triton decode, GDN, vision
tower (send a `data:image/png;base64,...` message), concurrent requests.

## Memory

96 GiB of the 128 GB is carved out as VRAM on this box (the rest is host RAM,
which the PLE table must *not* fill; the file backend's RSS trimmer keeps it
near the 8 GiB cap). Measure with `amdgpu_top --json --dump`
(`VRAM.Total VRAM Usage`).

Measured: weights 75.9 GiB after `load_weights` (the vision tower is ~0.9
GiB of that), KV cache 3.0 GiB (262,144 tokens fp8; only the 12
full-attention layers hold KV, ~12 KB/token in fp8, so one full 262k-token
request fits), GDN state 5.5 GB in bf16 (100 slots → 20 requests).
Scheduler RSS ≈ 2.9 GB; host `buff/cache` holds the PLE pages.

The `avail mem` figure in the log is not headroom on this ROCm build:
PyTorch sits at ~95 of 96 GiB once serving (a 1,650-token prefill under MTP
OOMed with `avail mem=13 GB` printed). Lowering `--mem-fraction-static`
does not create headroom either, the budget just moves into the KV/mamba
pools. What does:

| Change | Saves | Cost | Default |
|---|---:|---|---|
| `--kv-cache-dtype fp8_e4m3` | half the KV pool | none measured: identical greedy answers, exact needle recall in a 3,858-token prompt, decode 12.9 tok/s | on |
| `--max-total-tokens 262144` | ~3.2 GiB vs the fraction-sized pool | two full 131k-context requests or 20 × 13k fit at once; beyond that SGLang queues or retracts | on |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fragmentation | none | on |
| vision tower off (`QWEN38_VISION=0` / `QWEN38_MODEL_OVERRIDE='{"language_model_only": true}'`) | ~0.9 GiB (333 tensors skipped) | no image input | off |
| `--max-mamba-cache-size N` | ~54 MB per slot (bf16) | 5 slots per request with the radix cache on, 1 with `--disable-radix-cache`; under MTP an extra `(requests+1) × draft_tokens` intermediate states | 100 (= 20 requests) |

`--language-model-only` itself is whitelisted to three unrelated
architectures in this upstream; the model code supports it, so the override
sets `language_model_only` on the HF config instead.

Not worth doing: pruning East Asian tokens from the vocabulary. 65,932 of
248,077 tokens (26.6%) contain CJK, kana or hangul, but `embed_tokens` and
`lm_head` are 1.18 GiB each in bf16, so the whole prune saves ~0.63 GiB.
It also needs a checkpoint rewrite, BPE merge surgery, and, because the PLE
n-gram table is indexed by hashes of token ids, an id-remap in front of the
hash (or the 51B table is wrong for every renumbered token).

## Measured performance

Single box, no other GPU tenant, PLE table on local NVMe, radix cache
flushed before every prefill measurement (streaming client, `max_tokens`
128, temperature 0):

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 0.7 s | 264 tok/s (fixed overhead dominates) | 14.5 tok/s |
| 1,650 | 3.1 s | 535 tok/s | 14.5 tok/s |
| 6,693 | 12.1 s | 554 tok/s | 14.5 tok/s |
| 26,983 | 56.8 s | 475 tok/s | 14.7 tok/s |
| 32,935 | 71.7 s | 459 tok/s | 13.9 tok/s |
| 66,043 | 138.5 s | 477 tok/s | 14.0 tok/s |
| 99,151 | 215.2 s | 461 tok/s | 14.0 tok/s |
| 125,576 | 280.9 s | 447 tok/s | 14.0 tok/s |

Prefill is flat to the context limit (sparse attention and GDN, not dense
attention) and decode loses ~3% between short and 125k contexts. A needle at
60% depth of a 119,911-token prompt was recalled exactly. Two concurrent
59.5k-token requests with distinct prefixes: 119k KV tokens in use (46% of
the pool), 19.3 tok/s aggregate decode; the second request's prefill queued
behind the first (TTFT 151 s / 248 s) while the first decoded at 1.2 tok/s
between prefill chunks, which is chunked prefill working as designed.

| Concurrency (short prompts, 200 tokens each) | Aggregate | Per stream | TTFT (max) |
|---:|---:|---:|---:|
| 4 | 33.3 tok/s | 8.8 tok/s | 1.3 s |
| 8 | 57.9 tok/s | 7.5 tok/s | 1.2 s |
| 12 | 77.5 tok/s | 6.9 tok/s | 2.3 s |
| 16 | 99.4 tok/s | 6.7 tok/s | 2.6 s |
| 20 | 88–97 tok/s | 4.6–5.1 tok/s | 2.3 s |
| 24 | 74–76 tok/s | 5.8 tok/s | 42.6 s (4 queued) |

How it got here, single stream / 8 streams: eager 11.2 / 42.9 tok/s; decode
graphs (patch 14) 12.7 / 43.0; tuned MoE tiles (below) 14.5 / 57.9. Graphs
for bs 12–20 do not change throughput measurably against eager at those
sizes (server-side peak +4% at bs 16, within run-to-run noise end to end);
they were originally captured to keep eager decode out of the picture (see
patch 15) and stay on because they are cheap. Run-to-run spread at 8 streams
is about ±7%. Throughput
plateaus at 16 streams; the 20-request cap is queueing capacity, not speed
(each extra request costs ~270 MB of GDN state and one more graph).

The PLE gather costs about 1 s per 2048 cold tokens (32k rows faulted from
NVMe, ~35 µs each); rows already in the page cache shave that off (440–520
tok/s cold vs 550–585 warm). Decode is GPU-bound: with graphs on, ~98% of
the scheduler's host time is the D2H copy of the n-gram ids waiting for the
previous replay to finish. What is left in the ~69 ms bs=1 step is mostly
the bf16 dense projections (~8.6 GB of weight traffic per token, ~38 ms at
the measured ~225 GB/s) plus a long tail of small kernels; the MoE is now
~5 ms of it.

### MoE tile tuning

Upstream ships no fused-MoE Triton configs for `Radeon_8060S_Graphics`, so
the int4 expert GEMMs ran on a generic tile. [`tools/tune_moe_gfx1151.py`](../tools/tune_moe_gfx1151.py)
drives upstream's `benchmark/kernels/fused_moe_triton` tuner with a gfx1151
search space (1,560 configs) and an early bail-out: a single eager run and
then the first graph replay are timed, and a config is dropped as soon as it
is 3× slower than the best so far. 35–60% of configs bail at each M, which
took the sweep from an estimated day to about 7 GPU-hours on one 8060S with
the server left running (it needs ~1.5 GB of VRAM). Each batch size runs in
its own process (the Ray worker keeps every compiled kernel resident and the
kernel OOM-killed a single-process sweep at M=64 on this 31 GB-host-RAM
box), results are saved per M, and compiled kernels persist in the Triton
cache, so an interrupted run resumes at replay speed.

Kernel time for one MoE layer (M = tokens in the step; ×48 layers per step):

| M | Default tile | Tuned | Best config |
|---:|---:|---:|---|
| 1 | 223 µs | 103 µs | 16×32×64, 1 warp, `waves_per_eu` 2 |
| 2 | 740 µs | 200 µs | 16×16×64, 1 warp, `waves_per_eu` 2 |
| 4 | 1,184 µs | 403 µs | same |
| 8 | 1,634 µs | 739 µs | same |
| 16 | 2,960 µs | 1,342 µs | same |
| 32 | 5,012 µs | 2,070 µs | same |
| 64 | 7,653 µs | 3,471 µs | same |
| 128 | 9,876 µs | 4,475 µs | same |
| 512 | 11,462 µs | 5,223 µs | same |
| 1,024 | 12,671 µs | 5,814 µs | 32×16×64, 1 warp |
| 2,048 | 14,314 µs | 9,145 µs | 64×16×32, 1 warp |
| 4,096 | 22,483 µs | 14,761 µs | 128×64×32, 4 warps |

The same lesson as the Qwen3.5 hand sweep: at decode sizes the kernel is
bound by how many workgroups it can put on 40 CUs, so the smallest tiles
with one wave32 wavefront each win; `waves_per_eu=2` (not in upstream's
space) is worth another few percent; `BLOCK_K` 64 beats the group size 32
(the kernel indexes scales per element, so `BLOCK_K` may be any multiple of
the group). Only past M≈1k does tile efficiency start to matter and
`BLOCK_M` grow. End to end: bs=1 decode 12.6 → 14.5 tok/s (+15%), 4 streams
27.4 → 33.3 (+22%), 8 streams 43.0 → 57.9 (+35%); prefill unchanged within
noise (dominated by GDN/QSA and the PLE gather). Greedy answers, needle
recall in a 7,790-token prompt and generated code were re-checked after the
change; the tuner itself does not verify numerics.

The config lives in `configs/moe/qwen38-flash-next/` and is mounted at
`/moe-configs` by the launcher (patch 17 makes `SGLANG_MOE_CONFIG_DIR` a
search path in front of upstream's tree); nothing is baked into the image.
Re-tune after a Triton or kernel change:

```bash
docker cp tools/tune_moe_gfx1151.py sglang-qwen38:/tmp/
docker exec -w /tmp sglang-qwen38 python3 /tmp/tune_moe_gfx1151.py \
    --model /models/qwen38 --dtype int4_w4a16 --disable-shared-experts-fusion --tune
docker cp "sglang-qwen38:/tmp/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" \
    configs/moe/qwen38-flash-next/
```

Keep the tuner's `--tp-size` at its default 2 (that is what yields the
`N=320` key the runtime looks up). Another checkpoint gets its own profile
directory (`QWEN38_MOE_CONFIG_DIR=…`) so the two sets of tuning data never
overwrite each other; see [`configs/moe/README.md`](../configs/moe/README.md).

### Speculative decoding (MTP)

The checkpoint ships one MTP layer; `--speculative-algorithm NEXTN` loads it
as the draft model (+70 s load, 0.2 GB) and captures target-verify and draft
graphs. It needs real headroom: the mamba pool grows a 2.3 GB
`intermediate_ssm_state_cache`, and before the fp8 KV cache and the 262144
token cap became the defaults the eager GDN prefill OOMed on prompts over
~1k tokens (lowering `--mem-fraction-static` does not help, the budget just
moves into the pools; `--max-total-tokens 131072 --max-mamba-cache-size 30`
was the workaround). It still needs the request cap halved: at the default
20 requests the mamba pool is 5.5 GB plus a 4.4 GB intermediate cache
(`(requests+1) × 4 draft tokens`), and a 26k-token prefill OOMs. At 10
requests (50 slots + 2.3 GB intermediate) everything fits:

```bash
QWEN38_CUDA_GRAPH_MAX_BS=10 ./start-qwen38.sh --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 0.8 s | 222 tok/s | 21.0 tok/s |
| 1,650 | 2.9 s | 574 tok/s | 21.2 tok/s |
| 6,693 | 12.8 s | 522 tok/s | 21.8 tok/s |
| 26,363 | 54.6 s | 483 tok/s | 22.4 tok/s |

Mean accept length 2.6 of 4 draft tokens (accept rate 0.53); +45–55%
single-stream decode over plain graphs with the tuned MoE tiles. 4
concurrent: 39.5–40.4 tok/s aggregate (vs 33.3); 8 concurrent: 57.9–58.9
(vs 57.9). More than 10 streams queue (12 streams: 51.6 aggregate, 31 s
worst TTFT). `--speculative-num-steps 2
--speculative-num-draft-tokens 3` was tried: accept length 2.2, bs=1 18.4–19.7
tok/s, 8 streams 60.0, 12 requests allowed; not better. Not on by default: it
trades concurrency for single-stream speed.

Verified end to end: chat with thinking (`reasoning_content` split out),
structured tool calls (`finish_reason: tool_calls`), vision (exact OCR of
rendered text plus shape/color identification, 128 image tokens), 8
concurrent streams, 6302-token prompt with radix-cache reuse.

## Running the abliterated variant (DERISKED)

`davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ` is a refusal-ablated requant
of the same architecture: compressed-tensors W4A16, **symmetric, group 128**,
experts *and* the 12 full-attention layers' `q/k/v/o` quantized; indexer,
GDN, PLE, MTP, gates, norms, `lm_head` and the vision tower bf16 (the 333
vision tensors are byte-identical in name and shape to cyankiwi's). Kept
separate from the stock checkpoint end to end: own model directory, own PLE
directory (the table file name is the same for every checkpoint), own
container and served name, and its own benchmark and tuning data.

What had to change to run it, and where:

| Difference | Handling |
|---|---|
| Quantized dense attention projections | [Patch 16](../patches/16-wna16-rocm-dense.md): dequantized to bf16 at load, served with `F.linear` (the dense WNA16 scheme is Marlin-only; the MoE path already had a ROCm Triton kernel). Costs no extra bandwidth over the stock checkpoint's bf16 attention. |
| `model-mtp-merged.safetensors` is 100 GiB (all 128 PLE shards + MTP in one file) | `tools/convert_ple_fp8.py --part-bytes 2GiB` splits any oversized file into PLE-only `-pleNNN` and `-restNNN` parts and rewrites the index, so patch 13's PLE-shard skip still applies. The converter also stopped using `safe_open`: safetensors maps the file `PROT_WRITE|MAP_PRIVATE`, which overcommit mode 0 refuses for 100 GiB on a 30 GB host; it now reads headers and tensors with plain seek/`readinto`. |
| Chat template prepends a "Qwentium" obedience persona to every system block | Renamed to `chat_template.derisked.jinja`; the stock template is used. It is a template, not weights: the model's behaviour was probed without it. |
| No `preprocessor_config.json` / `video_preprocessor_config.json` | Copied from the stock checkpoint (identical processor). |
| Symmetric g128 experts | Same Triton kernel; the tuned `E=512,N=320` tiles measured the same on g128 as on g32 in the tuner's benchmark mode (bs 1/8/16/20/32: 103/734/1418/1569/2242 µs vs 105/750/1347/1611/2305), so the stock profile is mounted until a g128-tuned profile exists. |

Conversion (the 100 GiB file needs a memory cap only to keep the page cache
honest; RSS stays under 2 GiB):

```bash
docker run --rm --memory 14g \
    -v /scratch/hf-staging/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ:/src:ro \
    -v /opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8:/dst \
    -v $PWD/tools/convert_ple_fp8.py:/convert.py:ro \
    strix-halo-sglang:dev python3 -u /convert.py /src /dst --part-bytes 2GiB
cd /opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8
mv chat_template.jinja chat_template.derisked.jinja
cp /path/to/stock/{chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} .
```

Launch beside (not with: the GPU holds one of these) the stock server:

```bash
SGLANG_CONTAINER=sglang-qwen38-derisked QWEN38_SERVED_NAME=qwen38-flash-next-derisked \
MODEL_DIR=/opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8 \
PLE_DIR=/opt/llm/ple-cache-derisked ./start-qwen38.sh
```

Loads in ~6 min (first boot writes its own 47.7 GiB table), ~1.5 GB more
free VRAM after load than stock, same graph list and request cap. Verified: 12 greedy probes answered (the stock model
refused none of them either; the difference is in tone, not in refusals, for
that set), identity unchanged, vision exact on the synthetic test image
(160 image tokens). Greedy decode is bit-identical across cold runs since
patch 18; before it, this checkpoint (and stock) drifted from the first
decode token on, see below.

| | Stock (cyankiwi g32) | DERISKED (g128) |
|---|---:|---:|
| Prefill 1.6k / 6.7k / 27k tokens | 535 / 554 / 475 tok/s | 535 / 538 / 504 tok/s |
| Decode bs=1 | 14.5 tok/s | 14.3–15.0 tok/s |
| 8 streams | 57.9 (sanity re-run 49.6) | 47.1 / 51.8 |
| 16 streams | 99.4 (sanity re-run 89.5) | 70.6 / 73.9 / 72.7 |
| 20 streams | 88–97 | 72.9 |

Single-stream and prefill match; at 16–20 streams the abliterated build is
~20% behind. The MoE tiles are ruled out (same kernel time on both group
sizes, see above); the runs were not back to back with stock, and the
stock 16-stream figure itself moved 99 → 90 between sessions, so treat the
gap as partly run variance and partly open. Its logs show every decode step
in a captured graph. The int4 attention projections are served as bf16
(patch 16), so they cannot be slower than stock's bf16 ones.

## Known limitations

- The "avail mem" log figure is not headroom; see Memory before adding
  anything that allocates.
- Prefill runs eager (upstream disables prefill graphs for this model).
- First boot writes the 48 GiB PLE table; the `pinned` backend is not an
  option here (host RAM is the same pool).
- Fixed, kept for the record: greedy decode was not repeatable. The same
  prompt at `temperature=0` gave different top-1 logprobs from the first
  decode token on (a handful of recurring values), and long completions
  diverged on near-tie tokens; prefill was bit-identical. Graphs, overlap
  scheduling, the radix cache and the PLE fusion were ruled out one by one,
  as were the QSA, GDN, MoE and GEMM kernels by repeat-and-compare tests.
  The cause was the HyperConnection mix: on ROCm every batch of ≤ 16 rows
  ran upstream's persistent Triton kernel, which accumulates its split-K
  down projection with device-scope atomics. Patch 18 replaces it on HIP
  with a two-launch variant (per-split partials, fixed-order reduction, no
  grid barrier) at the same speed; 96- and 512-token completions are now
  bit-identical across cold runs. Sending the mix to the torch path instead
  would have cost ~35% decode speed (the GEMM shapes are untuned under
  TunableOp).
- Fixed, kept for the record: eager decode above the largest captured graph
  used to fault the next replay (`Memory access fault by GPU node-1 ... Page
  not present`) after a `/flush_cache`. Two factors: upstream's QSA backend
  shares one growable packed-KV scratch between captured graphs and eager
  decode, so an eager step larger than any graph re-allocated it and the
  graphs kept writing into the freed block; and `empty_cache()` then unmapped
  that block. Patch 15 gives the graphs their own scratch. Verified with
  graphs for bs ≤ 8 and 20 streams, flush between rounds, 3 rounds clean,
  greedy probe identical. Without the flush the stale write was silent.
