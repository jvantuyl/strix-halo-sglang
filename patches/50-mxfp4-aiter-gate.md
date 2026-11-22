# Patch 50 — gate mxfp4.py's module-level aiter import on `SGLANG_USE_AITER`

**File:** `python/sglang/srt/layers/quantization/mxfp4.py` · **Test:** `tests/check_mxfp4_aiter_gate.py` (3/3 in-image)

## Symptom

Every fresh interpreter that imports the quantization package — server boot,
every tuner worker generation, every precompile generation — rebuilds aiter's
`module_aiter_core` JIT module (~15 s), even with `SGLANG_USE_AITER=0`. The
boot timeline shows it ~9 s after start, right after the "Registered linear
attn model" line: the `server_args.resolve_once` → `expert_pack_hook` →
`model_loader` → `quantization/__init__` → `mxfp4` import chain.

## Root cause

`mxfp4.py` imports aiter at module top level on any HIP machine:

```python
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
...
if _is_hip:            # <- not _use_aiter
    try:
        from aiter.ops.shuffle import shuffle_scale, ...
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
    except ImportError as err:
        dynamic_mxfp4_quant = e8m0_shuffle = err
```

`SGLANG_USE_AITER` gates whether aiter kernels get **used**, not whether the
module gets **imported**. This checkpoint is compressed-tensors INT4, not
mxfp4, so the imported functions are never called — pure dead cost.

## How it was found

A `sys.meta_path` trap in a replica boot (`os._exit(42)` on the first aiter
import) printed the importing stack. A `builtins.__import__` hook missed it —
the model loader imports via `importlib.import_module`, which bypasses the
builtin hook; the meta-path finder catches every mechanism.

## The patch

Move the import behind `if _use_aiter:` and pre-bind the six names to `None`
so a call site that runs without the flag fails loudly with a `TypeError`
instead of a `NameError` — mirroring upstream's own ImportError fallback
(`dynamic_mxfp4_quant = err`). All real uses are guarded by `_use_aiter` or
live in mxfp4-quant paths this model never executes.

## Effect

Boots, tuner workers and precompile generations drop ~15 s each. Combined
with patch 49 (the flash-attn probe that imported flash_attn, whose
`flash_attn_interface` imports aiter), no process start touches aiter at all
with `SGLANG_USE_AITER=0`.

## Trade-offs accepted

- An mxfp4 model served with `SGLANG_USE_AITER=0` now fails at first aiter
  use with a TypeError where upstream would have worked (it imports aiter
  regardless). The env knob must be set to run mxfp4 through aiter on this
  image — which is exactly the flag's documented meaning.
- The static half of the test asserts the gate by source inspection; it does
  not prove aiter imports cleanly with the flag set (the JIT may want a GPU).
