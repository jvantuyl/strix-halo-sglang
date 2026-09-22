# strix-halo-sglang — SGLang for AMD Strix Halo (gfx1151)
#
# Build:   docker build -t strix-halo-sglang:dev .
# Run:     see README.md
#
# SGLang at upstream 70b5b03e (2026-09-21): brings official Qwen4-Exp support
# (PR #37500) with the PLE n-gram offload (PR #37068, --ple-offload-backend
# file) and the upstream rocm-gfx1151 kernel patches (PR #33939, vendored as
# patches/sgl-kernel-gfx1151.sh, replacing the old fork patches 1/1b).

# Base is pinned by digest so `:stable` can't drift under us (same rationale as
# the SGL_BRANCH pin below). Override BASE_IMAGE to bump the base or to use a
# registry mirror when Docker Hub is unreachable,
# e.g. --build-arg BASE_IMAGE=mirror.gcr.io/kyuz0/vllm-therock-gfx1151:stable
# See docs/BUILDING.md for details.
ARG BASE_IMAGE=kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SGLANG_FORCE_NATIVE_LAYERNORM=1
ENV HF_HOME=/root/.cache/huggingface
ENV PYTORCH_ROCM_ARCH=gfx1151

# Perf flags — measured ~38% throughput uplift on gfx1151 vs disabled defaults.
# TunableOp autotunes GEMM kernels per-shape; results cached at $PYTORCH_TUNABLEOP_FILENAME.
# Mount /root/.tunableop as a volume to persist tunings across container restarts.
ENV PYTORCH_TUNABLEOP_ENABLED=1
ENV PYTORCH_TUNABLEOP_FILENAME=/root/.tunableop/tunableop_results.csv
ENV HIP_FORCE_DEV_KERNARG=1
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# aiter's compiled attention/MoE kernels are CDNA-only; keep sglang on the
# Triton paths. This is load-bearing beyond attention: aiter's RMSNorm uses
# v_pk_mul_f32, a CDNA-only instruction, and its CK attention templates
# assume wave64. (Matches upstream docker/rocm-gfx1151.Dockerfile.)
ENV SGLANG_USE_AITER=0

WORKDIR /sgl-workspace

ARG SGL_REPO=https://github.com/sgl-project/sglang.git
# Pinned to a commit verified against this base image. Unpinned `main` drifts,
# which is what broke fresh builds in issue #5. SGL_BRANCH accepts any ref —
# a branch, tag, or commit SHA — because we fetch+checkout rather than clone -b.
ARG SGL_BRANCH=70b5b03e78612c94f86ac98eb4d2d8d19ceda738
RUN git init sglang \
    && cd sglang \
    && git remote add origin ${SGL_REPO} \
    && git fetch --depth 1 origin ${SGL_BRANCH} \
    && git checkout FETCH_HEAD

WORKDIR /sgl-workspace/sglang

# Patches 1/1b — gfx1151 arch gate + wave32 WARP_SIZE pin, now via the
# upstream script (python/sglang/kernels moved from sgl-kernel/ to
# python/sglang/kernels/aot/; see patches/01-allow-gfx1151.md and
# patches/04-warp-size-wave32.md for the history this replaces).
COPY patches/sgl-kernel-gfx1151.sh /tmp/sgl-kernel-gfx1151.sh

# Patch 2 — RMSNorm native fallback on gfx1151 (see patches/02-layernorm-native-fallback.md).
RUN python3 - <<'PYEOF'
p = '/sgl-workspace/sglang/python/sglang/srt/layers/layernorm.py'
old = '''elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
    except ImportError:
        # Fallback: vllm not available, will use forward_native
        _has_vllm_rms_norm = False'''
new = '''elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
        import os as _os
        if _os.environ.get('SGLANG_FORCE_NATIVE_LAYERNORM', '0') == '1':
            _has_vllm_rms_norm = False
    except ImportError:
        _has_vllm_rms_norm = False'''
t = open(p).read()
assert old in t, 'layernorm.py: elif _is_hip block not found, upstream layout changed'
open(p, 'w').write(t.replace(old, new))
PYEOF

# Compile the AOT kernels for gfx1151. Upstream moved sgl-kernel to
# python/sglang/kernels/aot with a pyproject_rocm.toml swap; the vendored
# script lifts the arch gate and pins WARP_SIZE=32 across both compiler
# passes (see patches/sgl-kernel-gfx1151.sh header).
WORKDIR /sgl-workspace/sglang/python/sglang/kernels/aot
RUN rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && sh /tmp/sgl-kernel-gfx1151.sh setup_rocm.py \
    && AMDGPU_TARGET=gfx1151 MAX_JOBS=16 python3 setup_rocm.py install

# setuptools-rust builds the sglang-mm extension during the pip install below
# if the base image doesn't already ship cargo. Fail loudly if the rustup
# installer produces nothing (e.g. a truncated download) instead of erroring
# much later inside pip with a confusing 127.
RUN command -v cargo >/dev/null || { \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; \
      test -x /root/.cargo/bin/rustc; }
ENV PATH="/root/.cargo/bin:${PATH}"

# Install SGLang. At this pin the extras compose through self-references
# (srt_hip -> sglang[runtime_common] -> sglang[runtime_base]) and pip's
# resolver chokes on that cycle from an editable source checkout, so flatten
# the three groups and install the requirements, then the package itself
# without dependency solving (same flow as upstream's rocm-gfx1151.Dockerfile).
#
# The torch/torchvision constraints file still freezes the base image's
# gfx1151-compiled builds — the root cause of issue #5, where a fresh build
# pulled generic PyPI wheels and failed at runtime with `libc10_hip.so: cannot
# open shared object file` — so a future incompatibility fails the build
# loudly instead of silently breaking at runtime.
WORKDIR /sgl-workspace/sglang
RUN cp python/pyproject_other.toml python/pyproject.toml \
    && python3 -c "import torch, torchvision; open('/tmp/rocm-constraints.txt', 'w').write(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')"
RUN python3 - <<'PYEOF'
import os
import subprocess
import sys
import tomllib
from pathlib import Path

project = tomllib.loads(Path("python/pyproject.toml").read_text())["project"]
extras = project["optional-dependencies"]
requirements = list(project["dependencies"])
for group in ("runtime_base", "runtime_common", "srt_hip"):
    requirements.extend(
        r
        for r in extras[group]
        if not r.startswith("sglang[") and r != "torch"
    )
requirements = list(dict.fromkeys(requirements))
env = {**os.environ, "PIP_CONSTRAINT": "/tmp/rocm-constraints.txt"}
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", *requirements], env=env
)
PYEOF
RUN PIP_CONSTRAINT=/tmp/rocm-constraints.txt pip install --no-deps -e python \
    && (pip cache purge 2>/dev/null || true)

# --- gfx1151 Qwen4-Exp fixes (patch 11) ---
# CPU-side PLE gather (the UVA kernel would fault on Strix Halo), QSA decode
# via the pure-Triton sm121 kernel, fast_topk fallback chain. See
# patches/11-qwen4-exp-rocm.md.
COPY patches/patch_qwen4_exp_rocm.py /tmp/patch_qwen4_exp_rocm.py
RUN python3 /tmp/patch_qwen4_exp_rocm.py && rm /tmp/patch_qwen4_exp_rocm.py

# --- WNA16 Triton MoE zero points (patch 12) ---
# Upstream's ROCm auto-route drops the zero points of asymmetric
# compressed-tensors checkpoints; the Triton kernel then silently computes
# wrong MoE outputs (Qwen3.8-Flash-Next-AWQ-INT4 is symmetric: false, g32).
# See patches/12-wna16-triton-zp.md.
COPY patches/patch_wna16_zp.py /tmp/patch_wna16_zp.py
RUN python3 /tmp/patch_wna16_zp.py && rm /tmp/patch_wna16_zp.py

# --- reuse the file-backed PLE table across boots (patch 13) ---
# Upstream rewrites the whole 48 GiB table from the checkpoint on every start.
# Record a fingerprinted completion marker and skip the PLE shards when it
# matches. See patches/13-ple-table-reuse.md.
COPY patches/patch_ple_table_reuse.py /tmp/patch_ple_table_reuse.py
RUN python3 /tmp/patch_ple_table_reuse.py && rm /tmp/patch_ple_table_reuse.py

# --- CUDA graphs with the CPU-side PLE gather (patch 14) ---
# Patch 11's host gather cannot be recorded into a graph. Fill the static PLE
# prefetch buffer from the host before each decode replay instead; the graph
# only reads it. Anchors on patch 11. See patches/14-cuda-graph-ple.md.
COPY patches/patch_cuda_graph_ple.py /tmp/patch_cuda_graph_ple.py
RUN python3 /tmp/patch_cuda_graph_ple.py && rm /tmp/patch_cuda_graph_ple.py

# --- dedicated QSA packed-KV scratch for captured graphs (patch 15) ---
# Upstream shares one growable scratch between captured decode graphs and
# eager decode; an eager step larger than any graph re-allocates it and the
# graphs keep writing into freed memory (page fault after the next
# empty_cache). Key the scratch on is_cuda_graph. See patches/15-qsa-graph-scratch.md.
COPY patches/patch_qsa_graph_scratch.py /tmp/patch_qsa_graph_scratch.py
RUN python3 /tmp/patch_qsa_graph_scratch.py && rm /tmp/patch_qsa_graph_scratch.py

# --- compressed-tensors int4 dense Linear on ROCm (patch 16) ---
# The wNa16 dense scheme is Marlin-only and Marlin is CUDA-only, so any
# checkpoint that quantizes attention / shared-expert / lm_head layers dies
# in process_weights_after_loading. Dequantize the packed weight to bf16 once
# at load and serve it with F.linear. See patches/16-wna16-rocm-dense.md.
COPY patches/patch_wna16_rocm_dense.py /tmp/patch_wna16_rocm_dense.py
RUN python3 /tmp/patch_wna16_rocm_dense.py && rm /tmp/patch_wna16_rocm_dense.py

# --- idle scheduler sleeps instead of spinning a core (patch 10) ---
# Upstream busy-polls its ZMQ sockets while idle, pinning one CPU core at 100%
# forever (Tctl 38 -> 71 C on an idle Strix Halo). Default --sleep-on-idle to on;
# --no-sleep-on-idle restores upstream behaviour. Re-anchored for the
# arg_groups restructure. See patches/10-sleep-on-idle-default.md
COPY patches/patch_sleep_on_idle.py /tmp/patch_sleep_on_idle.py
RUN python3 /tmp/patch_sleep_on_idle.py && rm /tmp/patch_sleep_on_idle.py

# --- aiter gfx1151 MXFP4 fix (patch 9) ---
# Unlocks Quark/MXFP4 checkpoints on gfx1151. Two blockers in the base image's aiter:
#   1. is_fp4_avail() only whitelists gfx950 -> allow gfx1151.
#   2. no gfx1151 GEMM configs; gfx950 configs need 100KB smem, RDNA3.5 only has 64KB.
#      Copy the FP4 gfx950 configs to gfx1151 with block sizes clamped to fit 64KB.
# Every anchor is asserted: aiter lives in the base image, so CI cannot check it and
# a layout change has to fail the build here rather than at the first FP4 request.
# See patches/09-aiter-gfx1151-mxfp4.md
COPY patches/fix_aiter_gfx1151_mxfp4.py /tmp/fix_aiter_gfx1151_mxfp4.py
RUN python3 /tmp/fix_aiter_gfx1151_mxfp4.py && rm /tmp/fix_aiter_gfx1151_mxfp4.py

# --- mounted fused-MoE tile configs (patch 17) ---
# Upstream ships no Radeon_8060S_Graphics configs and its SGLANG_MOE_CONFIG_DIR
# replaces the builtin tree (fixed configs/triton_x_y_z layout, crashes on a
# missing directory). Make it a search path checked before the builtin tree,
# flat or tree layout, so tuned tiles are mounted per checkpoint at run time
# (configs/moe/<profile>, see configs/moe/README.md) instead of baked here.
COPY patches/patch_moe_config_dir.py /tmp/patch_moe_config_dir.py
RUN python3 /tmp/patch_moe_config_dir.py && rm /tmp/patch_moe_config_dir.py

# --- deterministic HyperConnection mix at decode sizes (patch 18) ---
# The sm_100 JIT mix is unavailable here, so every decode step used the
# persistent Triton kernel whose split-K atomic_add made greedy decode differ
# run to run (and whose software grid barrier assumes co-resident CTAs). Add a
# two-launch variant (per-split partials, fixed-order reduction) and use it on
# HIP. See patches/18-hc-mix-rocm.md.
COPY patches/patch_hc_mix_rocm.py /tmp/patch_hc_mix_rocm.py
RUN python3 /tmp/patch_hc_mix_rocm.py && rm /tmp/patch_hc_mix_rocm.py

# --- GPTQ/AWQ MoE kernel: mask the weight load on a partial K block (patch 19) ---
# When K % BLOCK_SIZE_K != 0 the kernel masks a and the scales but read the
# packed weights unmasked, past the last expert's rows: a layout-dependent GPU
# page fault (killed the tuner on the g128 checkpoint). Same code for even K.
# See patches/19-moe-wna16-kmask.md.
COPY patches/patch_moe_wna16_kmask.py /tmp/patch_moe_wna16_kmask.py
RUN python3 /tmp/patch_moe_wna16_kmask.py && rm /tmp/patch_moe_wna16_kmask.py

# --- decode QSA block selection honours the HIP top-k guard (patch 20) ---
# select_decode_tokens called the JIT fast_topk directly, bypassing patch 11's
# guard: unsafe on RDNA 3.5 and its output order varies past 512 blocks, so
# long-context greedy decode drifted. Use a vectorised, graph-capturable torch
# top-k on HIP (the reference loop syncs per row). Anchors on patch 11.
# See patches/20-qsa-decode-topk.md.
COPY patches/patch_qsa_decode_topk.py /tmp/patch_qsa_decode_topk.py
RUN python3 /tmp/patch_qsa_decode_topk.py && rm /tmp/patch_qsa_decode_topk.py

# File-level verification (build host has no GPU; runtime check on container start).
# The AOT build installs the sgl_kernel package into site-packages.
RUN python3 -c "import glob, os, sgl_kernel; sos = glob.glob(os.path.join(os.path.dirname(sgl_kernel.__file__), '*.so')); assert sos, 'no built sgl_kernel extensions found'; print(sos)"

EXPOSE 30000

CMD ["python3", "-m", "sglang.launch_server", "--help"]
