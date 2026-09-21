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
    --attention-backend triton --cuda-graph-max-bs-decode 8 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3-thinking --tool-call-parser qwen3_coder
```

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

Measured with the default flags: weights 75.9 GiB after `load_weights`,
KV cache 6.1 GiB (267k tokens bf16; only the 12 full-attention layers hold
KV, ~24 KB/token, so one full 262k-token request fits), GDN state 5.4 GiB in
fp32 (50 slots → `max_running_requests` 10; 2.7 GiB / 20 requests with
`--mamba-ssm-dtype bfloat16`), 93.8 GiB VRAM in use once serving.
Scheduler RSS ≈ 2.9 GB; host `buff/cache` holds the PLE pages.

Note that torch reports a bogus total capacity on this ROCm build
(`avail mem=11.9 GB` in the log while 20 GiB of VRAM is actually free); the
KV/mamba pools were sized sensibly anyway with `--mem-fraction-static 0.85`.
If you need more concurrent requests, raise `--max-mamba-cache-size` rather
than `--mem-fraction-static`.

## Measured performance

Single box, no other GPU tenant, PLE table on local NVMe, radix cache
flushed before every prefill measurement (streaming client, `max_tokens`
128, temperature 0):

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 1.0 s | 192 tok/s (fixed overhead dominates) | 12.7 tok/s |
| 1,650 | 3.1 s | 539 tok/s | 12.6 tok/s |
| 6,693 | 13.3 s | 505 tok/s | 12.8 tok/s |
| 26,983 (eager run) | 58.9 s | 458 tok/s | 10.8 tok/s |

| Concurrency (short prompts, 200 tokens each) | Aggregate | Per stream | TTFT (max) |
|---:|---:|---:|---:|
| 4 | 25.9 tok/s | 6.7 tok/s | 1.4 s |
| 8 | 42.9 tok/s | 5.6 tok/s | 1.9 s |

Decode graphs (patch 14) took bs=1 from 11.2 to 12.7 tok/s; concurrency
figures are unchanged from the eager run.

The PLE gather costs about 1 s per 2048 cold tokens (32k rows faulted from
NVMe, ~35 µs each); rows already in the page cache shave that off (440–520
tok/s cold vs 550–585 warm). Decode is GPU-bound at ~75 ms per token:
with graphs on, ~98% of the scheduler's host time is the D2H copy of the
n-gram ids waiting for the previous replay to finish. The next lever is
kernel tuning, starting with the MoE Triton config (see below); nothing in
this table is memory-bandwidth bound yet.

Verified end to end: chat with thinking (`reasoning_content` split out),
structured tool calls (`finish_reason: tool_calls`), vision (exact OCR of
rendered text plus shape/color identification, 128 image tokens), 8
concurrent streams, 6302-token prompt with radix-cache reuse.

## Known limitations

- Prefill runs eager (upstream disables prefill graphs for this model).
- First boot writes the 48 GiB PLE table; the `pinned` backend is not an
  option here (host RAM is the same pool).
- MoE Triton configs for `E=512, N=640, int4_w4a16` are not tuned yet
  (`Using default MoE kernel config` warning).
- MTP / speculative decoding (`--speculative-algorithm NEXTN`) is only
  smoke-tested with dummy weights (graphs captured, requests complete); not
  measured on the real checkpoint.
