# Patch 12 — Zero points for CompressedTensorsWNA16TritonMoE

Applied by `patches/patch_wna16_zp.py` with asserted anchors.

Upstream routes ROCm WNA16 compressed-tensors MoE to
`CompressedTensorsWNA16TritonMoE` automatically (`compressed_tensors.py`:
`elif _is_hip: return CompressedTensorsWNA16TritonMoE(...)`). But the scheme
drops zero points end to end:

- `create_weights` registers `w13/w2_weight_zero_point` for asymmetric
  (`symmetric: false`) checkpoints and the loader fills them;
- `process_weights_after_loading` converts packed weights and scales for the
  Triton fused-MoE kernel but leaves the zero points untouched;
- `get_triton_quant_info` builds `TritonMoeQuantInfo` without `w13_zp/w2_zp`.

The GPTQ/AWQ Triton kernel (`fused_moe_kernel_gptq_awq`) then takes the
`has_zp=False` branch and subtracts the constant 8 from every 4-bit weight
instead of the per-group zero point — every MoE output of an asymmetric
checkpoint is silently wrong. (The runner plumbing exists:
`TritonRunnerCore.run` passes `w1_zp/w2_zp` into `_fused_moe_kernel_sequence`.)

The cyankiwi Qwen3.8-Flash-Next-AWQ-INT4 checkpoint is
`symmetric: false, group_size: 32`, so this fires for our model.

## The fix

Zero points are packed `[E, K/group_size, N/8]` int32 (8 zeros per int32 along
N, little nibble first). The kernel indexes them N-major like the scales
(`stride_bzk = zp.stride(2)`, `stride_bzn = zp.stride(1)`), expecting
`[E, N/2, K/group_size]` uint8. So:

1. In `process_weights_after_loading`, for `not self.sym`:
   `zp.data.view(torch.uint8).transpose(1, 2).contiguous()` — the byte view
   yields `[E, K/G, N/2]` with even n in the low nibble (matching the kernel's
   `(offs_bn % 2) * 4` shifter), and the transpose matches the scales.
2. In `get_triton_quant_info`, pass `w13_zp`/`w2_zp` when `not self.sym`.
3. In `fused_moe_triton/layer.py` (both the per-expert and fused
   `weight_loader` paths): the loader transposes compressed-tensors
   `weight_packed` / `weight_scale` (stored `[N, K/8]`, `[N, K/G]`) into the
   `is_transposed` parameters, but explicitly skips zero points
   (`and "zero" not in weight_name`). compressed-tensors packs
   `weight_zero_point` along dim 0, so it is stored `[N/8, K/G]` and needs the
   same flip to fit `[E, K/G, N/8]`. Without it the real checkpoint fails at
   load with `The size of tensor a (320) must match the size of tensor b (20)`
   on `down_proj.weight_zero_point`; `gate/up_proj.weight_zero_point` is
   square (`80×80` here) and would load silently transposed. Scoped to the
   Triton scheme so the Marlin/CUDA paths keep upstream behaviour.
4. In `model_loader/loader.py::DefaultModelLoader.postprocess_weights`, run
   `gc.collect()` after each MoE module's `process_weights_after_loading`.
   The Triton scheme replaces `w13/w2` packed weights, scales and zero points
   with transposed copies; the old device storage is only reclaimed by the
   cyclic collector once `stage_module_for_post_load` (which snapshots every
   registered tensor for its restore plan) has exited. Without the collection
   each of the 48 MoE layers left ~1.36 GiB of dead expert weights behind
   (measured 75.9 → 90.8 GiB allocated by layer 11) and the real checkpoint
   ran out of VRAM during post-load processing. With it, allocation returns to
   the post-`load_weights` baseline before every layer.

Verified on hardware by `tools/test_qwen38_rocm.py::test_moe_zp` (kernel vs
dequantized reference with random zero points, plus a negative control on the
untransposed layout).
