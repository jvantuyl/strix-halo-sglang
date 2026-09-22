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
#   - Decode CUDA graphs are on for bs 1-20 (patch 14 fills the PLE prefetch
#     buffer from the host before each replay). QWEN38_CUDA_GRAPH_MAX_BS sets
#     the largest graph; QWEN38_MAX_RUNNING_REQUESTS (default: the same
#     number) sets the request cap, and the GDN state pool is sized from it
#     (--max-mamba-cache-size = 5 slots per request; the ratio-sized pool
#     came out at 99 and capped the server at 19). Eager decode above the
#     graph range is safe since patch 15. QWEN38_CUDA_GRAPH_MAX_BS=0 is not a
#     switch; pass --disable-cuda-graph as an extra argument instead.
#   - The PLE table file (~48 GiB fp8, sparse) is written on the first boot,
#     reused on later ones (patch 13) and random-read during decode; keep
#     $PLE_DIR on local NVMe.
#   - --mamba-ssm-dtype bfloat16 halves the GDN recurrent state (~54 MB per
#     slot instead of ~108), so 100 slots cost 5.4 GB instead of 10.8.
#   - TunableOp *tuning* is off by default (SGLANG_TUNABLEOP_TUNING=0): the
#     image enables TunableOp, and tuning benchmarks every GEMM solution for
#     each new prompt length, which cost 14-20 s of TTFT per novel length.
#     Recorded tunings (decode shapes etc.) are still used; untuned shapes
#     fall back to hipBLASLt heuristics. SGLANG_TUNABLEOP_TUNING=1 to record
#     more.
#   - The Triton/JIT kernel cache is persisted in $SGL_CACHE_DIR so restarts
#     do not recompile every kernel.
#   - KV cache is fp8_e4m3 (verified: identical answers, exact needle recall
#     at 3.9k tokens, same decode speed) and capped at 262144 tokens so the
#     halved pool becomes headroom (~3 GB) instead of a bigger pool. The
#     allocator uses expandable segments; headroom on this box is thin.

set -euo pipefail

IMAGE="${SGLANG_IMAGE:-strix-halo-sglang:dev}"
PORT="${SGLANG_PORT:-30001}"
NAME="${SGLANG_CONTAINER:-sglang-qwen38}"
CUDA_GRAPH_MAX_BS="${QWEN38_CUDA_GRAPH_MAX_BS:-20}"
MAX_RUNNING_REQUESTS="${QWEN38_MAX_RUNNING_REQUESTS:-$CUDA_GRAPH_MAX_BS}"
# With the radix cache on, SGLang reserves 5 GDN state slots per request.
MAMBA_CACHE_SIZE="${QWEN38_MAMBA_CACHE_SIZE:-$((5 * MAX_RUNNING_REQUESTS))}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8}"
PLE_DIR="${PLE_DIR:-/opt/llm/ple-cache}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TUNABLE_DIR="${TUNABLE_DIR:-$HOME/.cache/strix-halo-sglang-tunableop}"
SGL_CACHE_DIR="${SGL_CACHE_DIR:-$HOME/.cache/strix-halo-sglang-cache}"
TUNABLEOP_TUNING="${SGLANG_TUNABLEOP_TUNING:-0}"
MEM_FRAC="${SGLANG_MEM_FRAC:-0.85}"
CONTEXT="${SGLANG_CONTEXT:-131072}"
KV_DTYPE="${QWEN38_KV_DTYPE:-fp8_e4m3}"
MAX_TOTAL_TOKENS="${QWEN38_MAX_TOTAL_TOKENS:-262144}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# QWEN38_VISION=0 skips the vision tower (~0.9 GB of weights, text-only API).
MODEL_OVERRIDE='{}'
if [ "${QWEN38_VISION:-1}" = "0" ]; then
    MODEL_OVERRIDE='{"language_model_only": true}'
fi

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
    -e PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
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
        --kv-cache-dtype "$KV_DTYPE" \
        --max-total-tokens "$MAX_TOTAL_TOKENS" \
        --json-model-override-args "$MODEL_OVERRIDE" \
        --attention-backend triton \
        --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --max-mamba-cache-size "$MAMBA_CACHE_SIZE" \
        --mamba-ssm-dtype bfloat16 \
        --reasoning-parser qwen3-thinking \
        --tool-call-parser qwen3_coder \
        "$@"
