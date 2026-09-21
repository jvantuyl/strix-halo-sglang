#!/usr/bin/env python3
"""gfx1151 patch 12: zero points for CompressedTensorsWNA16TritonMoE.

Upstream routes ROCm WNA16 compressed-tensors MoE to
CompressedTensorsWNA16TritonMoE automatically, but the scheme drops zero
points end to end:

  * create_weights registers w13/w2_weight_zero_point for asymmetric
    (``symmetric: false``) checkpoints and the loader fills them;
  * process_weights_after_loading converts packed weights and scales for the
    Triton fused-MoE kernel but leaves the zero points untouched;
  * get_triton_quant_info builds TritonMoeQuantInfo without w13_zp/w2_zp.

The GPTQ/AWQ Triton kernel then takes the no-zp branch and subtracts the
constant 8 from every 4-bit weight instead of the per-group zero point, so
every MoE output of an asymmetric checkpoint is silently wrong.

The kernel indexes zero points N-major like the scales
(``b_zp_ptrs = ... + (offs_bn // 2) * stride_bzn + offs_k_true * stride_bzk``
with ``stride_bzk = zp.stride(2)``, ``stride_bzn = zp.stride(1)``), expecting
[E, N/2, K/group_size] uint8: two 4-bit zeros per byte, even n in the low
nibble. The stored layout is [E, K/group_size, N/8] int32 (8 zeros per int32
along N, little nibble first), so a little-endian byte view gives
[E, K/G, N/2] with even n in the low nibble, and the same transpose the
scales get yields the expected layout.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py"
text = open(p).read()

old = """        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        layer.is_triton_converted = True
"""
assert text.count(old) == 1, "wNa16 MoE: scale conversion anchor not found"
new = """        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        # gfx1151 patch 12: asymmetric checkpoints carry 4-bit zero points.
        # Without them the GPTQ/AWQ Triton kernel subtracts the constant 8
        # instead of the per-group zero point and every MoE output is wrong.
        # The kernel indexes zero points N-major like the scales: pack the
        # stored [E, K//group_size, N//8] int32 into [E, N//2, K//group_size]
        # uint8 (two 4-bit zeros per byte, even n in the low nibble).
        if not self.sym:
            w13_zp = layer.w13_weight_zero_point.data.view(torch.uint8)
            w13_zp = w13_zp.transpose(1, 2).contiguous()
            layer.w13_weight_zero_point = torch.nn.Parameter(
                w13_zp, requires_grad=False
            )
            w2_zp = layer.w2_weight_zero_point.data.view(torch.uint8)
            w2_zp = w2_zp.transpose(1, 2).contiguous()
            layer.w2_weight_zero_point = torch.nn.Parameter(
                w2_zp, requires_grad=False
            )

        layer.is_triton_converted = True
"""
text = text.replace(old, new, 1)

old = """        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=[0, self.group_size],
        )
"""
assert text.count(old) == 1, "wNa16 MoE: quant info anchor not found"
new = """        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            # gfx1151 patch 12: pass the repacked zero points for asymmetric
            # checkpoints (see the conversion in process_weights_after_loading).
            w13_zp=layer.w13_weight_zero_point if not self.sym else None,
            w2_zp=layer.w2_weight_zero_point if not self.sym else None,
            block_shape=[0, self.group_size],
        )
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# fused_moe_triton/layer.py -- the loader transposes compressed-tensors
# weight_packed / weight_scale ([N, K/8] and [N, K/G] on disk) into the
# is_transposed parameters ([E, K/8, N], [E, K/G, N]) but skips zero points.
# compressed-tensors packs weight_zero_point along dim 0, so it is stored
# [N/8, K/G] and needs the same transpose to fit [E, K/G, N/8]; without it
# w2 fails with a shape mismatch and w13 (square, N/8 == K/G for this
# checkpoint) would load silently transposed. Scoped to the Triton scheme.
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
text = open(p).read()

old_per_expert = """                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )
"""
assert text.count(old_per_expert) == 1, "moe layer: per-expert transpose anchor not found"
new_per_expert = """                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and (
                "zero" not in weight_name
                # gfx1151 patch 12: zero points are stored [N/8, K/G] and
                # must be flipped like the scales for the Triton scheme
                or method.__class__.__name__ == "CompressedTensorsWNA16TritonMoE"
            )
            else loaded_weight
        )
"""
text = text.replace(old_per_expert, new_per_expert, 1)

old_fused = """                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )
"""
assert text.count(old_fused) == 1, "moe layer: fused transpose anchor not found"
new_fused = """                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and (
                "zero" not in weight_name
                # gfx1151 patch 12: see the per-expert loader above
                or method.__class__.__name__ == "CompressedTensorsWNA16TritonMoE"
            )
            else loaded_weight
        )
"""
text = text.replace(old_fused, new_fused, 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# model_loader/loader.py -- the Triton scheme's process_weights_after_loading
# replaces w13/w2 packed weights, scales and zero points with transposed
# copies. The old device storage is only released by the cyclic garbage
# collector once stage_module_for_post_load has exited (it snapshots every
# registered tensor for the restore plan), so without a collection each MoE
# layer leaves ~1.4 GiB of dead expert weights behind and a 48-layer model
# runs out of VRAM part way through post-load processing. Collect after every
# MoE module so the peak stays at one layer's worth of temporaries.
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/model_loader/loader.py"
text = open(p).read()

old_post = """    @staticmethod
    def postprocess_weights(model, target_device):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
"""
assert text.count(old_post) == 1, "loader: postprocess_weights anchor not found"
new_post = """    @staticmethod
    def postprocess_weights(model, target_device):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                # gfx1151 patch 12: release the pre-conversion expert weights
                # before the next MoE layer allocates its transposed copies.
                if hasattr(module, "w13_weight_packed") or hasattr(
                    module, "w13_weight"
                ):
                    import gc

                    gc.collect()
"""
text = text.replace(old_post, new_post, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 12 (wna16 triton zero points) applied")
