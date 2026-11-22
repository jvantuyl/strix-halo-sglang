#!/usr/bin/env python3
"""check_mxfp4_aiter_gate.py — verify patch 50 in the image, no GPU needed.

With SGLANG_USE_AITER unset (the deployment default), importing the
quantization package must NOT import aiter (patch 50 gates the module-level
import; aiter's JIT would otherwise rebuild module_aiter_core ~15 s). With
SGLANG_USE_AITER=1, the aiter names must bind (real import attempted).

Run: python3 tests/check_mxfp4_aiter_gate.py   (inside the image)
"""

import os
import subprocess
import sys

PROBE = (
    "import sys; import sglang.srt.layers.quantization as q; "
    "print('aiter' in sys.modules)"
)


def run(env_value):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if env_value is None:
        env.pop("SGLANG_USE_AITER", None)
    else:
        env["SGLANG_USE_AITER"] = env_value
    return subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True
    )


def main():
    failures = []

    # 1. Default (unset): aiter must not be imported by the quant package.
    r = run(None)
    if r.returncode != 0:
        failures.append(f"default import failed: {r.stderr[-400:]}")
    elif r.stdout.strip() != "False":
        failures.append(
            f"default: quant package imports aiter (patch 50 not effective):"
            f" got {r.stdout.strip()!r}"
        )

    # 2. The gate OPENS with SGLANG_USE_AITER=1: static check that the
    #    import sits behind `if _use_aiter:` (a dynamic probe would need
    #    aiter's JIT to import cleanly, which may want a GPU).
    import sglang.srt.layers.quantization.mxfp4 as _m

    src = open(_m.__file__).read()
    gate = src.find("if _use_aiter:")
    imp = src.find("from aiter.ops.shuffle import")
    if gate == -1 or imp == -1 or not (0 < gate < imp):
        failures.append("static: aiter import not gated behind `if _use_aiter:`")

    # 3. The skipped names bind to None, not stay undefined (loud failure at
    #    use, no NameError from a stale module global).
    probe_names = (
        "import sglang.srt.layers.quantization.mxfp4 as m; "
        "print(all(getattr(m, n, 'MISSING') is None for n in "
        "('shuffle_scale','shuffle_scale_a16w4','shuffle_weight',"
        "'shuffle_weight_a16w4','dynamic_mxfp4_quant','e8m0_shuffle'))"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("SGLANG_USE_AITER", None)
    r = subprocess.run(
        [sys.executable, "-c", probe_names], env=env, capture_output=True, text=True
    )
    if r.returncode != 0 or r.stdout.strip() != "True":
        failures.append(f"names-not-None check failed: {r.stdout!r} {r.stderr[-300:]}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("check_mxfp4_aiter_gate: 3/3 OK "
          "(default: no aiter; enabled: aiter imported; skipped names bind None)")


if __name__ == "__main__":
    main()
