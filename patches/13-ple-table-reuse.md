# Patch 13 — Reuse the file-backed PLE table across boots

Applied by `patches/patch_ple_table_reuse.py` with asserted anchors.

With `--ple-offload-backend file`, upstream keeps the PLE n-gram table in a
sparse file with a deterministic name and only re-creates it when the size
changes — but the weight loader still reads every PLE shard of the checkpoint
and copies all of it into the mmap on every boot. For Qwen3.8-Flash-Next that
is 48 GiB read from the checkpoint plus 48 GiB of page-faulting writes into
the table on every start, and it was most of the ~10 min load time.

## What the patch does

1. **`qwen4_exp_ple_table.py`** — helpers for a completion marker
   `<table>.complete.json` holding a fingerprint of the checkpoint's
   `model.safetensors.index.json` (size + mtime) and of the PLE-only shard
   files (name + size), plus the table size. `mark_ple_table_complete` fsyncs
   the table before writing the marker (temp file + rename). When
   `allocate_ple_host_table` has to (re)create the file, it removes any stale
   marker. `ple_only_weight_files` decides from the index which checkpoint
   files hold nothing but `…ngram_embedding.shard_N.weight` tensors; a
   checkpoint that interleaves the table with other tensors yields none and
   simply gets upstream behaviour.
2. **`qwen4_exp.py`** — the host embedding remembers its table path. The
   model exposes `weight_files_to_skip(hf_folder, files)`: if every file-backed
   table has a marker matching the fingerprint, it logs
   `PLE table: reusing N table file(s) from a previous boot, skipping M
   checkpoint shard files` and returns the PLE-only files; otherwise it keeps
   the fingerprint. At the end of `load_weights`, if every one of the
   `split_ngram_parts` shards of a table was written, the marker is armed for
   the next boot.
3. **`model_loader/loader.py`** — `Source` gains a `weight_files_filter`
   populated from the model (the same pattern as `allow_patterns_overrides`),
   and both file-resolution paths (`_get_weights_iterator` and
   `resolve_model_weights`) apply it after `maybe_add_mtp_safetensors`.

## Invalidation

Any of these forces a full rewrite (which re-arms the marker): a different
or re-converted checkpoint (index or shard sizes change), a table file that
was deleted or re-created, a truncated table, a boot that did not see every
shard. Deleting `<table>.complete.json` by hand does the same.

Verified on CPU with the marker lifecycle test (create, arm, reuse, stale on
re-create, checkpoint change) and on hardware with two consecutive boots of
the real checkpoint.
