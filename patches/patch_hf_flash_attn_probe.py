#!/usr/bin/env python3
"""gfx1151 patch 49: flash-attn availability probe without importing flash_attn.

sglang/__init__.py calls _patch_flash_attn_availability() on every process
start. The probe did `import flash_attn` to check the v2 API; on HIP,
flash_attn.flash_attn_interface imports aiter at module import, and aiter's
JIT rebuilds module_aiter_core (~14 s) in EVERY process start — server boot,
every tuner worker, every precompile generation — even with
SGLANG_USE_AITER=0 (sglang's own aiter imports are gated on that flag; this
path was not).

The check only needs the package's own __init__: importlib.util.find_spec
resolves exactly the module `import flash_attn` would execute, and reading
its source is faithful to the hasattr(flash_attn, "flash_attn_func") test
the old probe performed. A bare namespace package (the flash-attn-4
masquerade this guard exists for) has no __init__ origin file, so it still
neuters is_flash_attn_2_available. The trade-off, accepted deliberately: a
flash_attn distribution whose __init__ exports the name but fails at real
import time is no longer caught here — such a breakage surfaces loudly at
first real use instead of silently at probe time.

Run from the Dockerfile with the target tree as argv[1] (defaults to
/sgl-workspace/sglang, the in-image convention).
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/utils/hf_transformers_patches.py"

OLD = """    try:
        import flash_attn as _fa

        if not hasattr(_fa, "flash_attn_func"):
            import transformers.utils as _u
            import transformers.utils.import_utils as _ui

            _ui.is_flash_attn_2_available = lambda: False
            _u.is_flash_attn_2_available = lambda: False
    except ImportError:
        pass
"""

NEW = '''    # gfx1151 patch 49: probe without importing. On HIP, importing
    # flash_attn pulls in flash_attn_interface, which imports aiter, and
    # aiter's JIT rebuilds module_aiter_core (~14 s) in every process start
    # even with SGLANG_USE_AITER=0. find_spec resolves the same top-level
    # module `import flash_attn` would execute; checking its __init__ source
    # is faithful to the old hasattr(flash_attn, "flash_attn_func") test.
    # A bare namespace package (flash-attn-4 masquerade) has no __init__
    # origin file and still neuters the availability flag, as before.
    import importlib.util
    import os as _os

    _spec = importlib.util.find_spec("flash_attn")
    _has_v2 = False
    if _spec is not None and _spec.origin and _os.path.exists(_spec.origin):
        with open(_spec.origin) as _f:
            _has_v2 = "flash_attn_func" in _f.read()
    if _spec is not None and not _has_v2:
        import transformers.utils as _u
        import transformers.utils.import_utils as _ui

        _ui.is_flash_attn_2_available = lambda: False
        _u.is_flash_attn_2_available = lambda: False
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if "gfx1151 patch 49" in text:
        sys.exit("patch 49 already applied")
    assert text.count(OLD) == 1, "hf_transformers_patches.py probe anchor not found"
    text = text.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(text)
    print("patch 49 applied: flash-attn probe without aiter-triggering import")


if __name__ == "__main__":
    main()
