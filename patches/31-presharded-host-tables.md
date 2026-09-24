# Patch 31: presharded dump/reload with a host-resident PLE table

**Files:** `python/sglang/srt/model_loader/loader.py`
**Script:** [`patch_presharded_host_tables.py`](patch_presharded_host_tables.py)

## Why presharded

The Qwen3.8-Flash-Next load is not I/O-bound. A direct read of a weight
shard runs at 4.2 GB/s through dm-crypt (the 74 GB of non-PLE weights is
~18 s of I/O), `safe_open().get_tensor()` is a zero-copy mmap view, and
the 8-thread loader only parses headers. The 152 s go to the consumer:
**222,718 tensors, median 25 KiB**, each a Python `weight_loader` call
with a `narrow` and a tiny H2D copy, 0.68 ms apiece.

Upstream's `--load-format presharded` (`PreshardedModelLoader`) answers
exactly that: the first boot loads normally and dumps the post-processed
state (a few thousand full-layer tensors, ≤20 GiB files, a checksum plan,
a `READY` marker) into a subfolder keyed by quantization, dtype, parallel
config and the model's parameter shapes; later boots initialise the model,
run `process_weights_after_loading` on the empty parameters to get the
post-processing shapes, and copy the dump straight in. The draft gets its
own tree (`draft_presharded_path`). Patch 27 parks weights after the loader
returns, so both paths park identically.

## What breaks on this model

1. **The PLE table is a parameter.** Patch 11's `Qwen4ExpPinnedHostEmbedding`
   registers the n-gram table as `weight`, a 47.7 GB file-backed mmap on the
   host, so it is in `model.state_dict()`. The dump would SHA it, write it a
   second time (75 → 123 GB) and the reload would copy it back into the mmap.
2. **The table's completeness is only checked from `load_weights`.** Patch
   13's `weight_files_to_skip` hook verifies the completion marker and the
   reload never calls `load_weights`, so a missing or half-written table
   would be served as garbage without a word.
3. **`load_weights` ends with `finalize_fused_in_proj()`** on every GDN
   module: `in_proj_qkvz` and `in_proj_ba` (unquantized bf16 in this
   checkpoint) are stacked into one GEMM weight and the parameters become
   row views of it. The reload would serve the two separate GEMMs.

And two things about the cache directory itself:

4. **The key does not cover the code.** The subfolder name hashes
   quantization, dtype, parallel layout and the parameters' shapes, but not
   the code that post-processed the values. A rebuilt image whose patches
   change that post-processing (patch 12's zero points, say) would reuse the
   old dump and serve wrong numerics silently.
5. **Interrupted and stale dumps are indistinguishable.** A dump killed
   mid-write (the first attempt here was OOM-killed) leaves a subfolder
   without `READY` that the next boot rewrites file by file, possibly with a
   differently planned layout, leaving orphans; a subfolder from an older
   image or another deployment's configuration just sits there taking disk.

## Fix

- Dump and missing-key check skip tensors whose device type is not the
  target device's. The table stays with its own machinery.
- Before trusting a dump, `_host_tables_ready` resolves the checkpoint files
  and calls the model's `weight_files_to_skip` hook, which is where patch 13
  checks the marker and sets `_ple_table_reused`. A model with file-backed
  tables that did not confirm them falls back to the normal load (which
  fills the table) and refreshes the dump, with a warning. Models without
  the hook or without file tables (the MTP draft) pass.
- After the copy, every module's `finalize_fused_in_proj` is run, so the
  fused GEMM path and its numerics match the normal load.
- `SGLANG_PRESHARDED_STAMP`, when set, joins the shard config (and so the
  subfolder name and the stored `shard_config` that is compared on reload).
  The launcher passes the image ID, so every rebuild dumps afresh. Unset,
  the key is upstream's.
- Before choosing dump or reload, the cache root is audited. The current
  key's subfolder without `READY` is an interrupted dump: rank 0 removes it
  and the dump starts clean. Every other subfolder is logged with its size
  and whether it is complete, and left alone; it may be another
  deployment's, so pruning is a manual step.

## Verification

See the runbook's Load time section for the measured first-boot dump cost
and the reload time. Greedy output (`temperature 0`, 8 prompts × thinking
on/off, 160 tokens) compared against a set captured from the normally
loaded server before the change; idle VRAM compared to confirm patch 27
still parks the same 2.0 GiB; `du` of the dump to confirm the table is not
in it.
