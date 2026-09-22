# Patch 16: compressed-tensors int4 dense Linear on ROCm

**File:** `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
**Script:** [`patch_wna16_rocm_dense.py`](patch_wna16_rocm_dense.py)
**Supersedes:** [patch 7](07-wna16-rocm-linear.md) (mount-only, anchored on an
older tree, symmetric only).

## Symptom

Any `compressed-tensors` `pack-quantized` checkpoint that quantizes a *dense*
Linear (attention `q/k/v/o`, shared experts, `lm_head`) fails to load on ROCm:

```
NameError: name 'gptq_marlin_repack' is not defined
```

`CompressedTensorsWNA16` is selected for those layers, `create_weights`
registers the packed parameters, and `process_weights_after_loading` goes
straight into the Marlin repack. Marlin's repack and GEMM are CUDA-only
(`sglang.kernels` JIT-compiles `gemm/marlin/*.cuh`), and the import is guarded
by `if _is_cuda`, so the method is undefined everywhere else.

This is why every public quantization of the Qwen3.5/3.6 MoE family that runs
on gfx1151 quantizes `mlp.experts.*` and nothing else (the MoE path has its own
Triton kernel), and why
`davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ`, which also quantizes the
12 full-attention layers' `q/k/v/o`, could not load here.

## Fix

On non-CUDA platforms, `process_weights_after_loading` unpacks the
compressed-tensors layout once into a dense `weight` in the scale dtype and
drops the packed parameters; `apply_weights` becomes `F.linear`.

```
weight_packed     [N, K/8]   int32, 8 nibbles per word along K, nibble = q + 2**(bits-1)
weight_scale      [N, K/G]   (or [N, 1] channelwise)
weight_zero_point [N/8, K/G] int32, 8 nibbles per word along N (asymmetric only)
weight_g_idx      [K]        group index per input column (actorder=group only)

w[n, k] = (nibble(n, k) - zp(n, g(k))) * scale[n, g(k)]
```

Zero points use the same `+ 2**(bits-1)` offset as the weights
(`compressed_tensors...pack_to_int32`), so the unsigned difference is the
signed one; symmetric checkpoints use the constant `2**(bits-1)`.

Everything else in the scheme (weight creation, sharded loading of fused
`qkv_proj`, TP partitioning) is untouched, so a checkpoint that loads on CUDA
loads the same way here.

## Cost

Correctness, not speed: the dense layers are served as bf16, so they cost the
same bandwidth as an unquantized checkpoint's. For Qwen3.8-Flash-Next that is
12 layers × 4 projections, well under 1 GB. The unpack runs once per layer at
load and materialises one `[N, K]` fp32 temporary. A fused int4 GEMM (patch 7
used vLLM's `gptq_gemm` with fp16 activations) can replace `F.linear` later
without touching the unpack.

## Verification

Synthetic check against a plain PyTorch reference, `[256, 1024]` group 128,
CPU and GPU, all four layouts (symmetric / asymmetric × with / without
`g_idx`): the dequantized weight matches to bf16 rounding
(`max|dw| ≤ 2e-3`, i.e. one ulp of the product) and `F.linear` output is
within `3e-3` relative of the reference.
