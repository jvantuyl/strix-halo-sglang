# Patch 48 — SGLANG_QSA_TILELANG gate for the QSA MQA kernels

Status: **written 2026-11-21**, part of the ROCm 10 base-port groundwork.
Default behavior is unchanged (TileLang stays the tuned path on the 7.13
base); set `SGLANG_QSA_TILELANG=0` on stacks where TileLang is broken.

## Problem

The ROCm 10 base's tilelang cannot compile the QSA indexer decode GEMM for
our geometry: during decode CUDA-graph capture,
`tilelang_qsa_mqa_decode` dies in
`GemmWarpPolicyNode::computeWarpPartition` with
`N must be divisible by 16, but got 8` and takes the scheduler down. The
7.13 base's tilelang compiles the same kernel (validated by patches 24/26's
history).

## Fix

A `qsa_tilelang_allowed()` gate (env `SGLANG_QSA_TILELANG`, default on) in
`qsa/mqa.py`'s decode and prefill dispatch. With the gate off, the validated
Triton MQA kernels (gfx1151 patches 24/26) serve both paths — the same
stand-ins that predate the TileLang kernels, so numerics are already
characterized. A TileLang that fails at *import* was already covered by
`HAS_TILELANG`; this gate covers the worse case where import succeeds and
capture dies.
