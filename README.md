# strix-halo-sglang

[SGLang](https://github.com/sgl-project/sglang) Docker image for **AMD Ryzen AI Max+ 395 / Radeon 8060S** (gfx1151, "Strix Halo").

AMD's official `rocm/sgl-dev` images only target MI300/MI350 data-center GPUs. This image fills the gap for consumer Strix Halo hardware.

## Status

| Capability | Status |
|---|---|
| Server starts, loads model | ✅ |
| Chat completion, tool calling | ✅ |
| RadixAttention prefix caching | ✅ |
| Continuous batching | ✅ — **4.76× Ollama at 8 concurrent streams** (Qwen3.5-35B-A3B, patches 7+8 + tuned MoE config) |
| Single-stream decode | ✅ **Parity with Ollama** (1.06×) on the 35B MoE with [patches 7+8](patches/07-wna16-rocm-linear.md) + [tuned MoE config](configs/moe/) — was 0.62×. See the quantization caveat in [`bench/results.md`](bench/results.md). |
| Quantized dense layers | ✅ — int4 attention/projection layers **and `lm_head`** on RDNA via [patch 7](patches/07-wna16-rocm-linear.md) + [patch 8](patches/08-lmhead-compressed-tensors.md); upstream is Marlin-only (CUDA) |
| AWQ-MoE inference | ✅ — Qwen3.5-35B-A3B-AWQ-4bit works end-to-end after [patch 4](patches/04-warp-size-wave32.md) |
| MXFP4 (Quark) inference | ✅ — Quark/MXFP4 checkpoints load and serve after [patch 9](patches/09-aiter-gfx1151-mxfp4.md); e.g. `Qwen3.5-27B-Quark-AWQ-MXFP4`. ⚠️ The generated gfx1151 GEMM configs are clamped to fit 64 KB of LDS, not tuned for it, and one model has been tried — treat it as working, not as fast. |

Tested on Fedora 43 host, ROCm 7.13 nightly, PyTorch 2.13.

## Tested models

| Model | Loads | Inference | GPU mem | Notes |
|---|:-:|:-:|---:|---|
| `Qwen/Qwen3-0.6B` | ✅ | ✅ | ~2 GB | Smoke test. Loads in seconds. |
| `Qwen/Qwen3.5-4B` | ✅ | ✅ | ~10 GB | Reference benchmark. 23.1 tps single-stream, 159.9 tps at 8 concurrent (warm TunableOp). |
| `Qwen/Qwen3.6-27B` | ✅ | ✅ | ~52 GB | Dense Mamba+attention hybrid, BF16. Comfortable on 96 GB+ GTT; on a 64 GB box squeeze it in with `--mem-fraction-static 0.96 --max-mamba-cache-size 16 --max-total-tokens 8192 --disable-cuda-graph`. ~1.7 tps single-stream — BF16 27B is GTT-bound on the iGPU. |
| `cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4` | ❌ | — | — | 4-bit (compressed-tensors wNa16) calls `gptq_marlin_repack`, which is NVIDIA-only — same wall as the GPTQ row below. Run the BF16 model above instead. |
| `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | ✅ | ✅ | ~23 GB | Mamba+MoE hybrid; needs `--max-total-tokens N --max-mamba-cache-size M` to fit. See [docs/RUNNING_AWQ_MOE.md](docs/RUNNING_AWQ_MOE.md). |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | ❌ | — | — | GPTQ-on-MoE needs the `gptq_marlin` backend, which is NVIDIA-only today. Use the AWQ variant above instead. |
| `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` | ✅ | ✅ | ~91 GB + PLE on disk | Qwen4-Exp hybrid GDN + sparse attention MoE with a 51B n-gram (PLE) table. The table is converted to fp8 and read from a file on demand, never resident. Needs a 128 GB box with 96 GB VRAM carved out. 14.5 tps single-stream, 58 tps at 8 concurrent, ~500 tps prefill flat to the 131k context limit (decode CUDA graphs and [tuned MoE tiles](configs/moe/) on). Chat, thinking, tool calls, vision and 120k-token needle recall verified. See [docs/RUNNING_QWEN38.md](docs/RUNNING_QWEN38.md). |

Hardware tested: AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 61.7 GB GTT. If you've run this on a different Strix Halo SKU please open an issue with results.

## Quickstart

Build the image:

```bash
docker build -t strix-halo-sglang:dev .
```

Run it with the helper script:

```bash
./start-sglang.sh                          # defaults to Qwen3.5-4B
./start-sglang.sh Qwen/Qwen3.5-4B
```

Or with Docker Compose:

```bash
docker compose up -d
```

Or the long form if you prefer:

```bash
docker run -d --name sglang \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/strix-halo-sglang-tunableop:/root/.tunableop \
    -e HF_TOKEN=$HF_TOKEN \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
    strix-halo-sglang:dev \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-4B \
        --host 0.0.0.0 --port 30000 \
        --mem-fraction-static 0.5 \
        --context-length 8192 \
        --attention-backend triton \
        --disable-cuda-graph
```

**Mount `~/.cache/strix-halo-sglang-tunableop:/root/.tunableop` — without it, the TunableOp cache is wiped on every restart and single-stream throughput collapses by ~50%.** `start-sglang.sh` and `compose.yaml` do this automatically; the long-form command above is the only invocation that needs the explicit volume.

Then:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hi"}]}'
```

Build takes ~10 min on first run; details in [docs/BUILDING.md](docs/BUILDING.md).

## What's in the image

- Base: [`kyuz0/vllm-therock-gfx1151:stable`](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151)
- PyTorch 2.13.0a0 + ROCm 7.13 (TheRock nightly)
- AITER pre-built for gfx1151
- SGLang pinned to a verified commit (see `SGL_BRANCH` in the [Dockerfile](Dockerfile)), built from source
- `sgl-kernel` compiled with `--amdgpu-target=gfx1151`

## Patches applied

Four small patches let SGLang run on gfx1151:

1. **`sgl-kernel/setup_rocm.py`** — allow `gfx1151` in the arch list (upstream guards against anything but `gfx942`/`gfx950`).
2. **`sglang/srt/layers/layernorm.py`** — `SGLANG_FORCE_NATIVE_LAYERNORM=1` skips the aiter (CDNA-only inline asm) and vLLM (older 4-arg signature) RMSNorm paths.
3. **`sglang/srt/layers/quantization/awq/schemes/awq_moe.py`** — `SGLANG_AWQ_MOE_TRITON_ROCM=1` routes AWQ MoE through SGLang's existing Triton path on ROCm.
4. **`sgl-kernel/csrc/moe/moe_topk_{softmax,sigmoid}_kernels.cu`** — fix host/device `WARP_SIZE` mismatch that caused `hipErrorLaunchFailure` → GPU page fault on the first MoE forward pass for any wave32 (RDNA 3.5) target. **This is the patch that unlocks AWQ-MoE inference on consumer Strix Halo.** A root-cause version of this fix, bundled with the gfx1151 arch guard, is upstream in [sgl-project/sglang#28097](https://github.com/sgl-project/sglang/pull/28097) (open); this repo carries the narrower downstream form so the image works today.

All four are baked into the Dockerfile by default. See [`patches/`](patches/) for the diffs.

Plus **patches 7 and 8**, which unlock quantized *dense* layers and `lm_head` on RDNA:

7. **[`compressed_tensors_wNa16.py`](patches/07-wna16-rocm-linear.md)** — SGLang's int4 Linear path is Marlin-only and Marlin is CUDA-only, so on ROCm you can quantize a MoE's experts but not its attention/projection layers. Adds a ROCm branch using `gptq_gemm`. Apply with [`patches/patch_wna16_rocm.py`](patches/patch_wna16_rocm.py).
8. **[`compressed_tensors.py`](patches/08-lmhead-compressed-tensors.md)** — `get_quant_method` returns `None` for `ParallelLMHead`, so a quantized `lm_head` silently falls back to an unquantized parameter the checkpoint never fills, producing uninitialized logits. Adds the dispatch. Apply with [`patches/patch_lmhead_rocm.py`](patches/patch_lmhead_rocm.py).

Plus **patch 9**, which unlocks Quark/MXFP4 checkpoints on gfx1151:

9. **[aiter gfx1151 MXFP4 fix](patches/09-aiter-gfx1151-mxfp4.md)** — aiter's `is_fp4_avail()` only whitelists `gfx950`, and no gfx1151 GEMM configs are shipped (the `gfx950` ones need 100 KB shared memory, RDNA 3.5 only has 64 KB). Baked into the Dockerfile: allows `gfx1151` and generates clamped `gfx1151-*.json` configs from the `gfx950` ones. Verified with `Qwen3.5-27B-Quark-AWQ-MXFP4` on a Ryzen AI Max+ 395.

Together with [`tools/quantize_nonexpert.py`](tools/quantize_nonexpert.py) they take Qwen3.5-35B-A3B from **3.70 GB to 1.69 GB streamed per decode token** — just under Ollama's ~1.8 GB — and, with the [tuned MoE config](configs/moe/), single-stream from **23.4 → 39.6 tps (+69%)** and 8-stream from 127.0 → **199.3 tps (+52%)**. That brings single-stream to **parity with Ollama (1.06×)** and **4.76× at 8 concurrent**. Both engines re-measured in one session, one at a time. ⚠️ The single-stream margin is small and my checkpoint is quantized more aggressively than Ollama's Q4_K_M (21 GB vs 26 GB resident) with **no quality evaluation done** — treat it as parity, not a win.

Plus **patches 11 to 15**, which bring Qwen3.8-Flash-Next (Qwen4-Exp) up on gfx1151:

11. **[Qwen4-Exp on ROCm](patches/11-qwen4-exp-rocm.md)** — the PLE n-gram table gather is moved to the CPU (the GPU must not dereference host memory here), QSA decode is routed to upstream's pure-Triton kernel (registry-gated to SM121) with a 64 KB shared-memory schedule, and the JIT top-k kernel (which reads out of bounds on RDNA 3.5 and poisons the HIP queue) is bypassed. Apply with [`patches/patch_qwen4_exp_rocm.py`](patches/patch_qwen4_exp_rocm.py).
12. **[WNA16 Triton MoE zero points](patches/12-wna16-triton-zp.md)** — the ROCm compressed-tensors MoE path dropped the asymmetric zero points, so every expert weight was off by `(8 - zp) * scale`; the loader also left them untransposed, and post-load weight conversion leaked ~1.4 GiB of dead expert weights per MoE layer (OOM at layer ~14 of 48) until a GC is run between layers. Apply with [`patches/patch_wna16_zp.py`](patches/patch_wna16_zp.py).
13. **[PLE table reuse](patches/13-ple-table-reuse.md)** — upstream rewrites the 48 GiB file-backed PLE table from the checkpoint on every start. A fingerprinted completion marker lets later boots skip the PLE shards entirely. Apply with [`patches/patch_ple_table_reuse.py`](patches/patch_ple_table_reuse.py).
14. **[CUDA graphs with the CPU PLE gather](patches/14-cuda-graph-ple.md)** — the host-side gather cannot be captured, so the decode graph runner fills the static PLE prefetch buffer from the host before each replay. Apply with [`patches/patch_cuda_graph_ple.py`](patches/patch_cuda_graph_ple.py).
15. **[Dedicated QSA scratch for captured graphs](patches/15-qsa-graph-scratch.md)** — upstream's QSA attention backend shares one growable packed-KV scratch between captured decode graphs and eager decode; an eager step larger than any graph re-allocates it and the graphs keep writing into freed memory, which becomes a GPU page fault at the next `empty_cache`. Not gfx1151-specific. Apply with [`patches/patch_qsa_graph_scratch.py`](patches/patch_qsa_graph_scratch.py).

On top of those, [`configs/moe/`](configs/moe/) ships fused-MoE Triton tiles for `E=512,N=320,int4_w4a16` found by [`tools/tune_moe_gfx1151.py`](tools/tune_moe_gfx1151.py) (upstream's tuner with a gfx1151 search space and an early bail-out that cuts the sweep by ~4×): the generic tile was 2.2× slower at decode sizes; end to end +15% single-stream and +35% at 8 streams. The Dockerfile bakes every file in that directory into the image, so the Qwen3.5 config above now applies without a mount too.

Note: with the current upstream pin, patch 1 (gfx1151 arch guard) and patch 4 (wave32 `WARP_SIZE`) are applied by upstream's own [`docker/patches/sgl-kernel-gfx1151.sh`](patches/sgl-kernel-gfx1151.sh) (vendored here), patch 3 is no longer needed (upstream auto-routes ROCm compressed-tensors MoE to the Triton path), and patch 5 was fixed upstream.

Two more are documented but not part of the serving path:

Plus **patch 10**, which stops an idle server from spinning a CPU core:

10. **[`server_args.py`](patches/10-sleep-on-idle-default.md)** — upstream's scheduler busy-polls for requests, pinning one core at 100% while the server is idle (Tctl 38 → 71 °C on a Ryzen AI Max+ 395). The image defaults `--sleep-on-idle` to on: idle CPU drops to 1%, with no measurable latency or throughput cost. `--no-sleep-on-idle` restores upstream behaviour.

5. **[`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`](patches/05-moe-tuner-n-mismatch.md)** — the MoE tuner halves `N` a second time for int4, so it writes config files the runtime never opens. Tuning an AWQ/GPTQ MoE silently no-ops. Fixed in the Dockerfile.
6. **[GPTQ MoE on ROCm](patches/06-gptq-moe-rocm.md)** — `moe_wna16` is denied by a blanket ROCm list, and `gptq_gemm`/`gptq_shuffle` are imported only under `if _is_cuda`. Patching both makes a GPTQ MoE checkpoint load and serve on gfx1151, but generation is still numerically wrong, so this is **not** enabled by default.

## Benchmarks

Reference result — the 35B MoE, same family on both engines (`qwen3.5:35b-a3b` GGUF vs `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`):

| Concurrent streams | Ollama tps | SGLang, experts-only int4 | SGLang, **patches 7+8** | advantage |
|---:|---:|---:|---:|---:|
| 1 | 37.6 | 23.4 | **39.7** | 1.06× |
| 4 | 38.7 | 73.2 | **126.5** | 3.27× |
| 8 | 38.9 | 131.1 | **185.1** | **4.76×** |

The middle column is the stock experts-only checkpoint every public release ships; the right
column adds int4 dense layers and `lm_head`, which upstream cannot load on AMD at all. Bytes
streamed per decode token drop 3.70 GB → 1.69 GB. Details: [`bench/results.md`](bench/results.md).

Qwen3.5-4B, with the caveats below:

| Concurrent streams | Ollama (llama.cpp) tps | strix-halo-sglang tps | SGLang advantage |
|---:|---:|---:|---:|
| 1 | 43.3 | 23.1 | 0.53× |
| 4 | 44.6 | 85.8 | 1.92× |
| 8 | 45.1 | 159.9 | **3.55×** |


Ollama serializes; SGLang's continuous batching keeps per-stream throughput nearly flat (4B: 23.1 → 20.0 tps, 35B-A3B: 23.4 → 16.4 tps from 1 → 8 streams) while aggregate scales near-linearly. Full numbers + what-didn't-work table: [`bench/results.md`](bench/results.md). Script: [`bench/concurrent_throughput.py`](bench/concurrent_throughput.py). **Benchmark one engine at a time** — the script needs both endpoints live, which makes cross-contamination easy.

The image enables TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`) by default — first request to each unique GEMM shape autotunes; results persist to a mounted volume at `/root/.tunableop`. Subsequent runs use the cached tunings (warm cache = the numbers above). **You must mount this path** — `start-sglang.sh` and `compose.yaml` do it automatically.

## Known limitations

- **AWQ MoE page fault — fixed.** Early builds crashed on the first MoE forward pass; the cause was a host/device `WARP_SIZE` mismatch in the topk gating kernels, fixed by [patch 4](patches/04-warp-size-wave32.md) (baked into the default build). Qwen3.5-35B-A3B-AWQ-4bit now runs end-to-end — see [docs/RUNNING_AWQ_MOE.md](docs/RUNNING_AWQ_MOE.md). The debugging record lives in [docs/AWQ_MOE_DEBUG.md](docs/AWQ_MOE_DEBUG.md).
- **Idle CPU spin — fixed for SGLang.** An idle server used to hold one core at 100%, fixed by [patch 10](patches/10-sleep-on-idle-default.md). A separate ROCm runtime spin (`AsyncEventsLoop`) can still pin a core on some kernels — see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md#an-idle-server-pins-a-cpu-core).
- **No aiter Flash Attention on gfx1151.** aiter's MHA kernels use Composable Kernel templates that assume wave64; gfx1151 is wave32. Falls back to Triton attention (slower).
- **No aiter RMSNorm.** `rmsnorm_quant_kernels.cu` uses CDNA-only `v_pk_mul_f32` inline asm.
- **CUDA graphs engage but the bottleneck is in-kernel.** Measured on the 35B: 23.1 tps with graphs vs 23.4 without — noise. Capture costs 8 s and 0.60 GB, so they are cheap, just not useful here.

## Why this exists

The `strix-halo-toolboxes` ecosystem (notably [kyuz0's vLLM toolbox](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes)) covers vLLM, PyTorch, ComfyUI, fine-tuning, and voice. SGLang has been the missing piece. SGLang's RadixAttention prefix caching and continuous batching are exactly what consumer homelab AI servers need — multiple agents and users hitting the same model concurrently.

## Acknowledgments

- **[kyuz0](https://github.com/kyuz0)** for `vllm-therock-gfx1151` — the base image, AITER build pattern, and proof that this is possible.
- **[paudley/ai-notes](https://github.com/paudley/ai-notes)** for the original Strix Halo build research.
- **[sgl-project/sglang](https://github.com/sgl-project/sglang)** maintainers.
- **AMD's TheRock** team for the ROCm 7.13 nightlies that made gfx1151 PyTorch viable.

## License

Apache 2.0. See [LICENSE](LICENSE).
