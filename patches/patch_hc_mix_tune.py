#!/usr/bin/env python3
"""gfx1151 patch 38: launch-config tuning for the split HyperConnection mix.

Patch 18 replaced upstream's persistent hc-mix kernel with a deterministic
two-launch variant; its launch metadata was carried over from the persistent
kernel (num_warps=8, default num_stages). On the current model (hc=4,
hidden 2560, lowrank 320 → down GEMM 16x10240x320, up GEMM 16x320x10240)
that oversubscribes the tiny tiles: the down kernel chains five
tl.dot(16x256 x 256x32) iterations per CTA with no load/dot overlap, and the
measured pair costs ~56 us back-to-back (down 42.5, up 13.7) against a
~15 us/launch floor.

A standalone sweep over (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
at the model's shapes (~/scratch/rocmfp4-port/hc_mix_tune.py) found the grid
constants already optimal; everything is in the launch metadata. num_warps
8 -> 2 plus num_stages=2 (software pipelining overlaps the next K block's
loads with the chained dots) takes the pair to 27.2 us (down 13.5, up 13.8),
-51%, with the same max error against the fp32 reference (9.75e-04 at
rows 1, 1.90e-03 at rows 16) and 50/50 bit-identical repeat launches at
rows 1/5/16 — the reduction order is untouched (SPLIT_K and the fixed-order
up-kernel sum stay as patch 18 defined them), so the variant stays
deterministic in the patch-18 sense.

The up kernel is already near its memory floor (w_up, 6.55 MB, ~29 us at
peak DRAM; 13.6 us measured with L2 hits on t_part) and is left alone.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/hc_mix_triton.py"
text = open(p).read()

old = """        BLOCK_N=_SPLIT_MIX_BLOCK_N,
        BLOCK_K=_SPLIT_MIX_BLOCK_K,
        num_warps=8,
    )"""
new = """        BLOCK_N=_SPLIT_MIX_BLOCK_N,
        BLOCK_K=_SPLIT_MIX_BLOCK_K,
        # gfx1151 (patch 38): 2 warps + pipelining hide the chained-dot
        # latency; 8 warps ran the same tiles 2x slower (56 -> 27 us/pair).
        num_warps=2,
        num_stages=2,
    )"""
assert text.count(old) == 1, "hc-mix down-launch anchor not found (once expected)"
open(p, "w").write(text.replace(old, new))

print("patch_hc_mix_tune: ok")
