#!/usr/bin/env python3
"""gfx1151 patch 50: gate mxfp4.py's module-level aiter import on SGLANG_USE_AITER.

sglang/srt/layers/quantization/mxfp4.py imports aiter at module top level
whenever the platform is HIP:

    _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
    ...
    if _is_hip:
        try:
            from aiter.ops.shuffle import ...
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.utility.fp4_utils import e8m0_shuffle
        except ImportError as err:
            dynamic_mxfp4_quant = e8m0_shuffle = err

SGLANG_USE_AITER gates whether aiter kernels get USED, but this import runs
on any HIP machine regardless of the flag. aiter's package __init__ JIT-builds
module_aiter_core (~15 s) on first import in every fresh interpreter, so every
boot pays it — the quantization/__init__ -> mxfp4 import chain runs during
server_args.resolve_once, ~9 s after start (found by a sys.meta_path trap in a
replica boot, 2026-11-22; the builtins.__import__ hook missed it because
importlib.import_module bypasses it). The quantization package is imported by
the model loader in every process: server, tuner workers, precompile
generations.

Patching the module to skip the import unless aiter kernels would actually be
used (SGLANG_USE_AITER truthy) mirrors upstream's own ImportError fallback:
the names bind to None in the skipped case, so any call site that runs without
the flag fails loudly with a TypeError at use, exactly like upstream's
`dynamic_mxfp4_quant = err` does today. All real uses are guarded by
`_use_aiter` or live in mxfp4-quant paths this model never executes.

Run from the Dockerfile with the target tree as argv[1].
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/layers/quantization/mxfp4.py"

OLD = """if _is_hip:
    # import aiter
    try:
        from aiter.ops.shuffle import (
            shuffle_scale,
            shuffle_scale_a16w4,
            shuffle_weight,
            shuffle_weight_a16w4,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
    except ImportError as err:
        dynamic_mxfp4_quant = e8m0_shuffle = err
"""

NEW = """# gfx1151 patch 50: gate the module-level aiter import on SGLANG_USE_AITER.
# Upstream imports aiter on every HIP machine even when the flag that gates
# aiter kernel USE is unset, so aiter's JIT rebuilds module_aiter_core
# (~15 s) in every process that imports the quantization package — server
# boots, tuner workers, precompile generations. Models that are not mxfp4
# never call these functions. When skipped, the names bind to None so any
# unguarded call fails loudly at use, mirroring upstream's ImportError
# fallback (`dynamic_mxfp4_quant = err`).
shuffle_scale = shuffle_scale_a16w4 = None
shuffle_weight = shuffle_weight_a16w4 = None
dynamic_mxfp4_quant = e8m0_shuffle = None
if _use_aiter:
    try:
        from aiter.ops.shuffle import (
            shuffle_scale,
            shuffle_scale_a16w4,
            shuffle_weight,
            shuffle_weight_a16w4,
        )
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
    except ImportError as err:
        dynamic_mxfp4_quant = e8m0_shuffle = err
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if "gfx1151 patch 50" in text:
        sys.exit("patch 50 already applied")
    assert text.count(OLD) == 1, "mxfp4.py aiter import anchor not found"
    text = text.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(text)
    print("patch 50 applied: mxfp4 aiter import gated on SGLANG_USE_AITER")


if __name__ == "__main__":
    main()
