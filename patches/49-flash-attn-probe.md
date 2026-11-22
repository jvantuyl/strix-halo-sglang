# Patch 49: flash-attn availability probe without importing flash_attn

## Symptom

Every process start in the image — server boot, tuner worker, precompile
generation — logs aiter's

```
[aiter] start build [module_aiter_core] ...
[aiter] finish build [module_aiter_core], cost 14.0s
```

even with `SGLANG_USE_AITER=0`, and `Current hipcc not support:
-mllvm -amdgpu-coerce-illegal-types=1, skip it.` next to it. That is ~14 s of
pure startup overhead per process.

## Root cause

An import-stack trace on the tuner's hot path showed the trigger is not any
of sglang's gated `import aiter` sites (those all sit behind
`SGLANG_USE_AITER`): it is

```
sglang/__init__.py
  -> _apply_hf_patches()                      srt/utils/hf_transformers_patches.py
    -> _patch_flash_attn_availability()       import flash_attn  (probe)
      -> flash_attn/__init__.py               from flash_attn.flash_attn_interface import ...
        -> flash_attn_interface (HIP build)   import aiter
          -> aiter JIT: rebuild module_aiter_core (14 s, not cached)
```

The probe's purpose is narrow: flash-attn-4 registers a bare `flash_attn`
namespace that makes transformers' `is_flash_attn_2_available()` return True
while lacking the v2 API, so the probe imports the package and neuters the
flag when `flash_attn_func` is missing. On this image flash-attn is 2.8.4
with the v2 export present — the probe is a no-op that still pays the aiter
rebuild.

## Change

`_patch_flash_attn_availability` now resolves the package with
`importlib.util.find_spec` (the same module `import flash_attn` would
execute — no import machinery is patched) and reads the resolved
`__init__.py` source for `flash_attn_func` instead of importing it. A bare
namespace package has no `__init__` origin file and still neuters the flag,
so the flash-attn-4 masquerade case is unchanged.

Accepted trade-off: a distribution whose `__init__` exports the name but
fails at real import time is no longer caught at probe time; it surfaces at
first real use instead (loudly, per the user's call: we would notice it
unambiguously).

## Effect

Zero-cost probe: no aiter build, no flash_attn_interface load at sglang
import time. Saves ~14 s per process start (server boot once; every tuner
worker and precompile generation). flash-attn semantics on this image are
unchanged (v2.8.4, export present, availability flag untouched).
