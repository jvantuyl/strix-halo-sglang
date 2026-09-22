#!/usr/bin/env python3
"""Patch 16: compressed-tensors int4 dense Linear on ROCm via load-time dequant.

``CompressedTensorsWNA16`` (the scheme behind every ``pack-quantized`` int4/int8
*dense* Linear: attention projections, shared experts, lm_head) has exactly one
kernel path, Marlin, whose repack and GEMM are CUDA-only. On ROCm the scheme is
still selected, ``create_weights`` succeeds, and ``process_weights_after_loading``
dies on ``gptq_marlin_repack`` (imported only under ``_is_cuda``). That is why
public checkpoints that quantize anything beyond ``mlp.experts.*`` could not load
on gfx1151.

This adds a non-CUDA branch that unpacks the compressed-tensors layout once at
load time into a plain bf16/fp16 ``weight`` and serves it with ``F.linear``:

  weight_packed     [N, K/8]   int32, 8 nibbles per word along K, nibble = q + 8
  weight_scale      [N, K/G]   (or [N, 1] channelwise)
  weight_zero_point [N/8, K/G] int32, 8 nibbles per word along N (asymmetric only)
  weight_g_idx      [K]        group index per input column (actorder=group only)

  w[n, k] = (nibble(n, k) - zp(n, g(k))) * scale[n, g(k)],  zp = 8 when symmetric

The packed parameters are dropped afterwards, so steady-state memory is the
dense weight only. Fine for the handful of dense layers such checkpoints
quantize; not a bandwidth win, just correctness. A fused int4 GEMM (patch 7
used vLLM's gptq_gemm) can replace ``F.linear`` later without touching the
unpack.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
text = open(p).read()

# ---------------------------------------------------------------- imports
old = """from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
"""
assert text.count(old) == 1, "compressed_tensors_wNa16.py: is_cuda import anchor not found"
new = """from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
# Marlin (repack + GEMM) is CUDA-only. Everywhere else the packed int4/int8
# weight is dequantized once at load time and served through F.linear.
_WNA16_DENSE_FALLBACK = not _is_cuda
"""
text = text.replace(old, new, 1)

# ------------------------------------------- process_weights_after_loading
old = """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config
"""
assert text.count(old) == 1, "compressed_tensors_wNa16.py: process_weights_after_loading anchor not found"
new = """    def _dequantize_to_dense(self, layer: torch.nn.Module) -> None:
        \"\"\"Unpack the compressed-tensors layout into a dense ``weight``.

        Nibbles are stored ``q + 2**(bits-1)`` (unsigned), 8 per int32 along
        the packed dim, element i in bits [bits*i, bits*(i+1)). Zero points use
        the same encoding, so ``(q_u - zp_u)`` equals the signed difference.
        \"\"\"
        w_q = layer.weight_packed.data
        w_s = layer.weight_scale.data
        bits = 32 // self.pack_factor
        mask = (1 << bits) - 1
        shifts = torch.arange(
            0, 32, bits, device=w_q.device, dtype=torch.int32
        )
        n, k_packed = w_q.shape
        k = k_packed * self.pack_factor
        # [N, K/8, 8] -> [N, K]; ``& mask`` also undoes the arithmetic shift
        # sign extension of negative int32 words.
        q = ((w_q.unsqueeze(-1) >> shifts) & mask).reshape(n, k)

        num_groups = w_s.shape[1]
        if self.has_g_idx:
            g_idx = layer.weight_g_idx.data.to(torch.int64)
        else:
            g_idx = torch.arange(k, device=w_q.device) // (k // num_groups)

        if self.symmetric:
            zp = torch.full(
                (n, num_groups), 1 << (bits - 1), device=w_q.device, dtype=torch.int32
            )
        else:
            zp_packed = layer.weight_zero_point.data  # [N/8, K/G], packed along N
            zp = ((zp_packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask).reshape(
                n, num_groups
            )

        cols = g_idx.view(1, k).expand(n, k)
        w = (q - torch.gather(zp, 1, cols)).to(torch.float32) * torch.gather(
            w_s.to(torch.float32), 1, cols
        )
        layer.register_parameter(
            "weight", torch.nn.Parameter(w.to(w_s.dtype), requires_grad=False)
        )
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_shape",
            "weight_zero_point",
            "weight_g_idx",
        ):
            if name in layer._parameters:
                del layer._parameters[name]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _WNA16_DENSE_FALLBACK:
            self._dequantize_to_dense(layer)
            return

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config
"""
text = text.replace(old, new, 1)

# ----------------------------------------------------------- apply_weights
old = """    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.kernel_config
"""
assert text.count(old) == 1, "compressed_tensors_wNa16.py: apply_weights anchor not found"
new = """    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        if _WNA16_DENSE_FALLBACK:
            return torch.nn.functional.linear(x, layer.weight, bias)

        c = self.kernel_config
"""
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patch 16 (wna16 dense dequant fallback) applied")
