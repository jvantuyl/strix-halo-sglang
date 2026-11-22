#!/usr/bin/env python3
"""check_mxfp4_aiter_gate.py — verify patch 50 in the image, no GPU, no import.

Patch 50 gates mxfp4.py's module-level aiter import behind SGLANG_USE_AITER.
The quantization package probes the GPU at import time on this stack, so this
check inspects the patched source directly instead of importing it. The
dynamic equivalent is free at every boot: with SGLANG_USE_AITER unset, a boot
log no longer prints aiter's "start build [module_aiter_core]" line ~9 s in.

Run: python3 tests/check_mxfp4_aiter_gate.py   (inside the image)
"""

import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4.py"
NAMES = (
    "shuffle_scale",
    "shuffle_scale_a16w4",
    "shuffle_weight",
    "shuffle_weight_a16w4",
    "dynamic_mxfp4_quant",
    "e8m0_shuffle",
)


def main():
    failures = []
    try:
        src = open(PATH).read()
    except OSError as e:
        print(f"FAIL: cannot read {PATH}: {e}")
        sys.exit(1)

    # 1. The aiter import sits behind `if _use_aiter:`.
    gate = src.find("if _use_aiter:")
    imp = src.find("from aiter.ops.shuffle import")
    if gate == -1 or imp == -1 or not (0 < gate < imp):
        failures.append("aiter import not gated behind `if _use_aiter:`")

    # 2. All six names are pre-bound to None before the gate (chained
    #    assignments), so an unguarded use fails loudly (TypeError) instead
    #    of raising NameError.
    prelude = src[: gate if gate != -1 else 0]
    for line in (
        "shuffle_scale = shuffle_scale_a16w4 = None",
        "shuffle_weight = shuffle_weight_a16w4 = None",
        "dynamic_mxfp4_quant = e8m0_shuffle = None",
    ):
        if line not in prelude:
            failures.append(f"prelude missing assignment: {line}")
    for name in NAMES:
        if name not in prelude:
            failures.append(f"{name} not pre-bound to None before the gate")

    # 3. The gate is the patch's own marker (idempotence + provenance).
    if "gfx1151 patch 50" not in src:
        failures.append("patch 50 marker comment missing")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("check_mxfp4_aiter_gate: 3/3 OK (gate behind _use_aiter, names "
          "pre-bound None, marker present). Live check: boot with "
          "SGLANG_USE_AITER=0 prints no aiter 'start build' line.")


if __name__ == "__main__":
    main()
