# Patch 17: `SGLANG_MOE_CONFIG_DIR` as a search path for mounted MoE tiles

**File:** `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`
**Script:** [`patch_moe_config_dir.py`](patch_moe_config_dir.py)

## Symptom

Upstream has one knob for supplying fused-MoE tile configs from outside the
package, `SGLANG_MOE_CONFIG_DIR`, and it is unusable for that purpose:

* it **replaces** the builtin config tree instead of extending it, so every
  shipped config disappears while it is set;
* it must reproduce the package layout, `<dir>/configs/triton_<ver>/<file>.json`,
  not the flat file the tuner writes;
* a directory that does not exist raises `FileNotFoundError` from
  `os.listdir` in the version-fallback loop and kills the server at the
  first MoE layer.

The only reliable way to ship tuned tiles was therefore to copy them into the
package inside the image, which welds a per-checkpoint tuning artefact to the
build and makes two checkpoints with the same expert shape (the file name is
`E=…,N=…,dtype=…`; `block_shape=[0, group]` is dropped by the `all()` check)
fight over one file.

## Fix

`SGLANG_MOE_CONFIG_DIR` is an `os.pathsep`-separated list of extra
directories searched **before** the builtin tree. Each is tried flat
(`<dir>/<file>.json`) and in the builtin layout
(`<dir>/configs/triton_<ver>/<file>.json`); a missing directory is skipped.
When nothing matches, the lookup proceeds through the builtin tree exactly
as before, including the other-Triton-version fallback and its warning.

Behaviour without the variable is unchanged. A directory that satisfied the
old semantics still matches (second candidate), so nothing that worked
before stops working; the difference is that the builtin tree is now a
fallback rather than shadowed.

## Use here

The launchers and `compose.yaml` mount one profile from
[`configs/moe/<profile>`](../configs/moe/) at `/moe-configs` and set
`SGLANG_MOE_CONFIG_DIR=/moe-configs`, so the image carries no tuning data
and each checkpoint keeps its own tiles (`MOE_CONFIG_DIR` /
`QWEN38_MOE_CONFIG_DIR` on the host side; empty disables the mount and runs
upstream's generic tiles). The server logs
`Using MoE kernel config from /moe-configs/E=….json` when the mount is
picked up.

## Verification

`tools/test_qwen38_rocm.py` item 6 uses a fabricated `(E, N)` so the builtin
tree never matches and checks: unset → `None` (generic tiles, no exception);
a missing directory → skipped; flat layout found; tree layout found; the
first directory in the list wins. End to end: the Qwen3.8 server started
with the mount logs the `/moe-configs` path and decodes at the tuned speed;
with `MOE_CONFIG_DIR=` it logs upstream's `Config file not found` and runs
at the generic speed (2.2× slower MoE at decode sizes).
