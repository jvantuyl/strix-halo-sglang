#!/usr/bin/env python3
"""Check patch 38: the split hc-mix down kernel uses the tuned launch config.

Runs in the image without a GPU: imports the patched module and asserts the
down launch carries num_warps=2 / num_stages=2 while the up launch and the
patch-18 grid constants are untouched.
"""
import inspect
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.layers import hc_mix_triton as m  # noqa: E402

# Patch 18's grid constants must be unchanged (the sweep found them optimal).
assert m._SPLIT_MIX_SPLIT_K == 8
assert m._SPLIT_MIX_BLOCK_N == 32
assert m._SPLIT_MIX_BLOCK_K == 256

src = inspect.getsource(m._fused_hc_mix_split)
assert "num_warps=2" in src, "down launch should use num_warps=2 (patch 38)"
assert "num_stages=2" in src, "down launch should use num_stages=2 (patch 38)"
# Only the down launch was retuned; the up kernel keeps 8 warps.
assert src.count("num_warps=8") == 1, "up launch should keep num_warps=8"

print("check_hc_mix_launch: ok")
