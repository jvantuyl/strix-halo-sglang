#!/usr/bin/env python3
"""gfx1151 patch 17: search SGLANG_MOE_CONFIG_DIR before the builtin MoE configs.

Upstream reads SGLANG_MOE_CONFIG_DIR as a *replacement* for the package's
config tree: it must contain configs/triton_<version>/<file>.json, a missing
directory raises FileNotFoundError from os.listdir, and every builtin config
disappears while it is set. That makes it useless for the thing it is for:
mounting a tuned tile file into a stock image.

After this patch the variable is an os.pathsep-separated list of extra
directories searched *before* the builtin tree. Each directory is tried flat
(<dir>/<file>.json, what the tuner writes) and in the builtin layout
(<dir>/configs/triton_<version>/<file>.json); directories that do not exist
are skipped. Lookups fall through to the builtin tree exactly as before, so
behaviour without the variable is unchanged and configs found under the old
semantics are still found.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py"
text = open(p).read()

old = """    config_dir = os.environ.get(
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
"""
assert text.count(old) == 1, "fused_moe_triton_config: config_dir anchor not found"
new = """    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"

    # gfx1151 patch 17: SGLANG_MOE_CONFIG_DIR lists extra directories
    # (os.pathsep-separated) searched before the builtin tree, so a tuned
    # config can be mounted into a stock image. Each is tried flat (the
    # tuner's output layout) and in the builtin configs/triton_x_y_z layout;
    # missing directories are skipped and the builtin tree stays the fallback.
    for extra_dir in os.environ.get("SGLANG_MOE_CONFIG_DIR", "").split(os.pathsep):
        if not extra_dir:
            continue
        for candidate in (
            os.path.join(extra_dir, json_file_name),
            os.path.join(extra_dir, "configs", version_dir, json_file_name),
        ):
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    logger.info(f"Using MoE kernel config from {candidate}.")
                    return {int(key): val for key, val in json.load(f).items()}

    config_dir = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 17 (moe config search path) applied")
