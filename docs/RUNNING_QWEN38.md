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
| [10](../patches/10-sleep-on-idle-default.md) | Idle scheduler sleeps | unchanged, re-anchored to the new `arg_groups` layout |
| [configs/moe](../configs/moe/) | Tuned fused-MoE Triton tiles for `E=512,N=320,int4_w4a16` | Upstream has no `Radeon_8060S_Graphics` configs; the generic tile is 2.2× slower at decode. See MoE tile tuning below |

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
   MoE zero points incl. a negative control).
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
    --mem-fraction-static 0.85 --context-length 32768 \
    --kv-cache-dtype fp8_e4m3 --max-total-tokens 262144 \
    --attention-backend triton --cuda-graph-max-bs-decode 8 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3-thinking --tool-call-parser qwen3_coder
```

with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment.

with `SGLANG_FORCE_NATIVE_LAYERNORM=1 SGLANG_USE_AITER=0
SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 PYTORCH_TUNABLEOP_TUNING=0` in the
environment, and `~/.cache/strix-halo-sglang-cache` mounted at
`/root/.cache/sglang` (SGLang's Triton / JIT kernel cache) so restarts do not
recompile every kernel.

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
| `--kv-cache-dtype fp8_e4m3 --max-total-tokens 262144` | fp8 KV halves the pool; the token cap keeps the saving as headroom instead of a larger pool (see Memory). `QWEN38_KV_DTYPE` / `QWEN38_MAX_TOTAL_TOKENS` override. |
| `--cuda-graph-max-bs-decode 8` | Decode graphs for bs 1–8 (0.39 GB). Patch 14 fills the PLE prefetch buffer from the host before each replay. Note this upstream split the flag: a bare `--cuda-graph-max-bs` is rejected as ambiguous. `QWEN38_CUDA_GRAPH_MAX_BS` overrides. |
| `--mamba-ssm-dtype bfloat16` | The GDN recurrent state is fp32 by default: 5.4 GB for 50 slots, which caps `max_running_requests` at 10 (5 slots per request). bf16 halves it to 20 requests in the same memory. Upstream's own suggestion in the startup log. |
| `--reasoning-parser qwen3-thinking --tool-call-parser qwen3_coder` | The chat template opens `<think>` in the generation prompt and asks for `<tool_call><function=...><parameter=...>` tool calls. Without these the thinking and the XML come back as plain `content`. |
| `PYTORCH_TUNABLEOP_TUNING=0` | The image enables PyTorch TunableOp, which benchmarks every GEMM solution for each *new* M (= tokens in the prefill chunk). That is 14–20 s of TTFT for every novel prompt length (measured; the recorded results persist in `~/.cache/strix-halo-sglang-tunableop` so a repeated length is fast). With tuning off the recorded solutions are still used and untuned shapes take hipBLASLt's heuristic pick. `SGLANG_TUNABLEOP_TUNING=1 ./start-qwen38.sh` to deliberately record more. |
| `SGLANG_USE_AITER=0` | Set in the image. |

### What to look for in the log

```
Using CompressedTensorsWNA16TritonMoE (ROCm)
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
request fits), GDN state 2.7 GiB in bf16 (~100 slots → 19–20 requests).
Scheduler RSS ≈ 2.9 GB; host `buff/cache` holds the PLE pages.

The `avail mem` figure in the log is not headroom on this ROCm build:
PyTorch sits at ~95 of 96 GiB once serving (a 1,650-token prefill under MTP
OOMed with `avail mem=13 GB` printed). Lowering `--mem-fraction-static`
does not create headroom either, the budget just moves into the KV/mamba
pools. What does:

| Change | Saves | Cost | Default |
|---|---:|---|---|
| `--kv-cache-dtype fp8_e4m3` | half the KV pool | none measured: identical greedy answers, exact needle recall in a 3,858-token prompt, decode 12.9 tok/s | on |
| `--max-total-tokens 262144` | ~3.2 GiB vs the fraction-sized pool | one full-context request or 8 × 32k still fit | on |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fragmentation | none | on |
| vision tower off (`QWEN38_VISION=0` / `QWEN38_MODEL_OVERRIDE='{"language_model_only": true}'`) | ~0.9 GiB (333 tensors skipped) | no image input | off |
| `--max-mamba-cache-size N` | 27 MB per slot (bf16) | 5 slots per request under MTP, 1 otherwise | fraction-sized |

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

| Concurrency (short prompts, 200 tokens each) | Aggregate | Per stream | TTFT (max) |
|---:|---:|---:|---:|
| 4 | 33.3 tok/s | 8.8 tok/s | 1.3 s |
| 8 | 57.9 tok/s | 7.5 tok/s | 1.2 s |

How it got here, single stream / 8 streams: eager 11.2 / 42.9 tok/s; decode
graphs (patch 14) 12.7 / 43.0; tuned MoE tiles (below) 14.5 / 57.9.

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

The config is baked into the image (Dockerfile copies `configs/moe/*.json`
into the installed Triton version's config directory). Re-tune after a
Triton or kernel change:

```bash
docker cp tools/tune_moe_gfx1151.py sglang-qwen38:/tmp/
docker exec -w /tmp sglang-qwen38 python3 /tmp/tune_moe_gfx1151.py \
    --model /models/qwen38 --dtype int4_w4a16 --disable-shared-experts-fusion --tune
docker cp "sglang-qwen38:/tmp/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" configs/moe/
```

### Speculative decoding (MTP)

The checkpoint ships one MTP layer; `--speculative-algorithm NEXTN` loads it
as the draft model (+70 s load, 0.2 GB) and captures target-verify and draft
graphs. It needs real headroom: the mamba pool grows a 2.3 GB
`intermediate_ssm_state_cache` and the eager GDN prefill then OOMs on
prompts over ~1k tokens at the default pool sizes (lowering
`--mem-fraction-static` does not help, the budget just moves into the pools).
Cap the pools instead:

```bash
./start-qwen38.sh --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --max-total-tokens 131072 --max-mamba-cache-size 30
```

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 1.2 s | 150 tok/s | 18.7 tok/s |
| 1,650 | 3.3 s | 499 tok/s | 17.2 tok/s |
| 6,693 | 13.9 s | 481 tok/s | 18.5 tok/s |
| 26,983 | 59.8 s | 451 tok/s | 18.8 tok/s |

Mean accept length 2.7 of 4 draft tokens (accept rate 0.55) on the summary
prompts above; +45% single-stream decode over plain graphs. (Measured before
the MoE tile tuning; the draft and verify steps use the same MoE kernel, so
expect both columns to move.) 4 concurrent:
29.5 tok/s aggregate (vs 27.4). `--max-mamba-cache-size 30` allows 6
concurrent requests (5 slots each), so 8 streams queue (31.5 tok/s
aggregate, 32 s worst TTFT vs 43.0 tok/s without MTP). Not on by default:
it trades concurrency for single-stream speed.

Verified end to end: chat with thinking (`reasoning_content` split out),
structured tool calls (`finish_reason: tool_calls`), vision (exact OCR of
rendered text plus shape/color identification, 128 image tokens), 8
concurrent streams, 6302-token prompt with radix-cache reuse.

## Known limitations

- The "avail mem" log figure is not headroom; see Memory before adding
  anything that allocates.
- Prefill runs eager (upstream disables prefill graphs for this model).
- First boot writes the 48 GiB PLE table; the `pinned` backend is not an
  option here (host RAM is the same pool).
- MTP needs the pool caps above; without them prefill OOMs past ~1k tokens.
