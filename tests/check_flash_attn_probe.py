#!/usr/bin/env python3
"""check_flash_attn_probe: verify patch 49's import-free flash-attn probe.

Run inside the image (no GPU needed): `python3 tests/check_flash_attn_probe.py`.

Covers three behaviors of the patched _patch_flash_attn_availability:
1. real flash_attn 2.8.4: the availability flag stays untouched and — the
   point of the patch — `aiter` is NOT imported by the probe (the old probe
   triggered aiter's ~14 s JIT rebuild on every process start);
2. bare-namespace masquerade (spec with no origin file): the availability
   flag IS neutered, matching the old hasattr-based behavior;
3. package absent (spec None): no neuter, matching the old ImportError path.

Each case restores transformers' original flag before returning.
"""

import importlib.machinery
import importlib.util
import sys

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def fresh_flags():
    import transformers.utils as u
    import transformers.utils.import_utils as ui

    return u.is_flash_attn_2_available, ui.is_flash_attn_2_available


def call_probe():
    from sglang.srt.utils import hf_transformers_patches as m

    m._patch_flash_attn_availability()


def main():
    import sglang  # noqa: F401  (applies all HF patches, incl. 49)

    case1 = fresh_flags()
    call_probe()
    u1, ui1 = fresh_flags()
    check("case 1: availability flag untouched for flash-attn 2.8.4",
          u1 is case1[0] and ui1 is case1[1])
    check("case 1: probe did not import aiter", "aiter" not in sys.modules)

    import sglang.srt.utils.hf_transformers_patches as m
    import importlib.util as _ilu

    for label, spec in (
        ("case 2: bare-namespace masquerade neuters the flag",
         importlib.machinery.ModuleSpec("flash_attn", None)),
        ("case 3: package absent leaves the flag alone", None),
    ):
        orig_find = _ilu.find_spec
        orig_flags = fresh_flags()
        _ilu.find_spec = lambda name, _s=spec: _s  # stdlib patch, restored
        try:
            call_probe()
        finally:
            _ilu.find_spec = orig_find
        import transformers.utils as u
        import transformers.utils.import_utils as ui
        if label.startswith("case 2"):
            check(label, u.is_flash_attn_2_available() is False
                  and ui.is_flash_attn_2_available() is False)
        else:
            check(label, u.is_flash_attn_2_available is orig_flags[0]
                  and ui.is_flash_attn_2_available is orig_flags[1])
        u.is_flash_attn_2_available = orig_flags[0]
        ui.is_flash_attn_2_available = orig_flags[1]

    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
