#!/bin/bash
# Start Qwen3.8-Flash-Next (cyankiwi AWQ-INT4, FP8-converted PLE table) on
# gfx1151 with the PLE n-gram table read from disk via --ple-offload-backend file.
#
# Requires:
#   - image strix-halo-sglang:dev (built from this Dockerfile)
#   - converted checkpoint at $MODEL_DIR (see tools/convert_ple_fp8.py)
#   - the GPU to yourself: resident weights alone are ~76 GiB and the server
#     uses ~94 GiB of the 96 GiB VRAM carve-out once KV/mamba pools are up.
#
# Usage:
#   ./start-qwen38.sh [extra sglang args...]
#   SGLANG_PORT=30000 ./start-qwen38.sh
#
# Notes:
#   - Decode CUDA graphs are on for bs 1-8 (patch 14 fills the PLE prefetch
#     buffer from the host before each replay). QWEN38_CUDA_GRAPH_MAX_BS=0
#     is not a switch; pass --disable-cuda-graph as an extra argument instead.
#   - The PLE table file (~48 GiB fp8, sparse) is written on the first boot,
#     reused on later ones (patch 13) and random-read during decode; keep
#     $PLE_DIR on local NVMe.
#   - --mamba-ssm-dtype bfloat16 halves the GDN recurrent state (5.4 -> 2.7 GB
#     for 50 slots), so 20 requests can run instead of 10.
#   - TunableOp *tuning* is off by default (SGLANG_TUNABLEOP_TUNING=0): the
#     image enables TunableOp, and tuning benchmarks every GEMM solution for
#     each new prompt length, which cost 14-20 s of TTFT per novel length.
#     Recorded tunings (decode shapes etc.) are still used; untuned shapes
#     fall back to hipBLASLt heuristics. SGLANG_TUNABLEOP_TUNING=1 to record
#     more.
#   - The Triton/JIT kernel cache is persisted in $SGL_CACHE_DIR so restarts
#     do not recompile every kernel.

set -euo pipefail

IMAGE="${SGLANG_IMAGE:-strix-halo-sglang:dev}"
PORT="${SGLANG_PORT:-30001}"
NAME="${SGLANG_CONTAINER:-sglang-qwen38}"
CUDA_GRAPH_MAX_BS="${QWEN38_CUDA_GRAPH_MAX_BS:-8}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8}"
PLE_DIR="${PLE_DIR:-/opt/llm/ple-cache}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TUNABLE_DIR="${TUNABLE_DIR:-$HOME/.cache/strix-halo-sglang-tunableop}"
SGL_CACHE_DIR="${SGL_CACHE_DIR:-$HOME/.cache/strix-halo-sglang-cache}"
TUNABLEOP_TUNING="${SGLANG_TUNABLEOP_TUNING:-0}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.85}"
CONTEXT="${SGLANG_CONTEXT:-32768}"

test -d "$MODEL_DIR" || { echo "missing $MODEL_DIR (run tools/convert_ple_fp8.py first)" >&2; exit 1; }
mkdir -p "$PLE_DIR" "$HF_CACHE" "$TUNABLE_DIR" "$SGL_CACHE_DIR"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Note: removing existing container '$NAME'." >&2
    docker rm -f "$NAME" >/dev/null
fi

set -x

exec docker run --name "$NAME" \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --network=host \
    --security-opt seccomp=unconfined \
    -v "$MODEL_DIR:/models/qwen38:ro" \
    -v "$PLE_DIR:/ple" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$TUNABLE_DIR:/root/.tunableop" \
    -v "$SGL_CACHE_DIR:/root/.cache/sglang" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTORCH_TUNABLEOP_TUNING="$TUNABLEOP_TUNING" \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
    -e SGLANG_USE_AITER=0 \
    -e SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 \
    "$IMAGE" \
    python3 -m sglang.launch_server \
        --model-path /models/qwen38 \
        --served-model-name qwen38-flash-next \
        --host 0.0.0.0 --port "$PORT" \
        --ple-offload-embedding \
        --ple-offload-backend file \
        --ple-offload-dir /ple \
        --mem-fraction-static "$MEM_FRAC" \
        --context-length "$CONTEXT" \
        --attention-backend triton \
        --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
        --mamba-ssm-dtype bfloat16 \
        --reasoning-parser qwen3-thinking \
        --tool-call-parser qwen3_coder \
        "$@"
