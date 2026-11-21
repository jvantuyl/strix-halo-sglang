#!/usr/bin/env python3
"""gfx1151 patch 47: tolerate unregistered model_type in strict transformers.

The ROCm 10 base image ships transformers 5.20 (the resolver installs 5.12+
with the pinned SGLang). There, AutoConfig.from_pretrained raises ValueError
for a model_type that is not in its registry ("Unrecognized model ... Should
have a `model_type` key"); transformers 5.5 (the 7.13 base) tolerated unknown
types and returned a generic PretrainedConfig. Qwen4Exp checkpoints are
exactly such an unregistered type, so the pinned SGLang cannot load them on
the new base.

This patch keeps the AutoConfig call first (registered types keep every
transformers feature), and on the strict-mode ValueError falls back to
loading config.json into a plain PretrainedConfig — the same object the old
behavior produced. Anything else (missing file, malformed json) still raises.

Run from the Dockerfile with the target tree as argv[1] (defaults to
/sgl-workspace/sglang, the in-image convention).
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/utils/hf_transformers/config.py"

OLD = """        if config is None:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
"""

NEW = '''        if config is None:
            try:
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            except (KeyError, ValueError):
                # gfx1151 patch 47: transformers >= 5.12 raises for model
                # types outside its registry (5.5 returned a generic
                # PretrainedConfig); the exception type and message vary by
                # version (ValueError "Unrecognized model" / "does not
                # recognize this architecture" in 5.20, KeyError from
                # CONFIG_MAPPING in 5.12) — catch both. Missing/malformed
                # config.json surfaces as OSError/JSONDecodeError instead.
                import os  # config.py has no module-level os import

                path = (
                    model
                    if os.path.isdir(model)
                    else os.path.dirname(model)
                )
                with open(os.path.join(path, "config.json")) as f:
                    config = PretrainedConfig.from_dict(json.load(f))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if 'gfx1151 patch 47' in text:
        sys.exit("patch 47 already applied")
    assert text.count(OLD) == 1, "config.py parse anchor not found"
    text = text.replace(OLD, NEW, 1)
    # PretrainedConfig is already module-level imported (used by the mamba
    # and longcat helpers) — no import change needed.
    with open(path, "w") as f:
        f.write(text)
    print("patch 47 applied: unregistered model_type fallback")


if __name__ == "__main__":
    main()
