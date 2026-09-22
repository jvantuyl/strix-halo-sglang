#!/usr/bin/env python3
"""gfx1151 patch 19: mask the B load in the GPTQ/AWQ fused-MoE kernel for a
partial last K block.

`fused_moe_kernel_gptq_awq` masks `a` and `b_scale` (and `b_zp`) when K is
not a multiple of BLOCK_SIZE_K (`not even_Ks`), but loads the packed weight
block `b` unmasked. The last K iteration then reads BLOCK_SIZE_K - K % BLOCK_SIZE_K
rows past the expert's weights; for the last expert that is past the end of
the tensor, and when the allocation ends on a page boundary it is a GPU page
fault (amdgpu `[gfxhub] page fault`, HSA aborts the process). Whether it
faults depends on what the caching allocator placed after the tensor, so it
is layout-dependent: tuning the same E=512,N=320 shape passed on the g32
checkpoint and killed the worker on the g128 one, at the first config with
BLOCK_SIZE_K=128 (the tuner's tp-2 shard gives the down projection K=320).
The masked elements meet zeros in `a`, so results were never affected; only
the read itself is out of bounds.

The generic `fused_moe_kernel` in the same file already masks `b` under
`not even_Ks`; this mirrors it. `even_Ks` is a constexpr, so kernels for
K % BLOCK_SIZE_K == 0 (every shape the runtime uses for this model) compile
to exactly the same code as before.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py"
text = open(p).read()

old = """        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF
"""
assert text.count(old) == 1, "fused_moe_triton_kernels: gptq_awq B load anchor not found"
new = """        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        if not even_Ks:
            # gfx1151 patch 19: the partial last block must not read past the
            # expert's rows (past the tensor for the last expert). The masked
            # elements meet zeros in `a`, so `other` is irrelevant.
            b = tl.load(b_ptrs, mask=k_mask, other=0)
        else:
            b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 19 (moe wna16 k-mask) applied")
