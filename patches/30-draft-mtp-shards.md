# Patch 30: the MTP draft loads only the shards that hold MTP tensors

**Files:** `python/sglang/srt/models/qwen4_exp_mtp.py`
**Script:** [`patch_draft_mtp_shards.py`](patch_draft_mtp_shards.py)

## Symptom

With `--speculative-algorithm NEXTN` the boot log shows two weight loads:

```
Load weight end. elapsed=152.41 s, type=Qwen4ExpForConditionalGeneration
Load weight begin.
Load weight end. elapsed=49.46 s,  type=Qwen4ExpForCausalLMMTP
```

The draft is one MoE layer plus norms and projections: 31 tensors, 5 GB,
in `model-mtp-merged-rest00{0,1,2}.safetensors`. 49 s for that is a third
of the target's time for a fifteenth of the bytes.

## Cause

`Qwen3_5ForCausalLMMTP.load_weights` skips every checkpoint name without
`mtp` in it (the draft's token embedding and head are placeholders that
`set_embed_and_head` later replaces with the target's), but the loader
still walked every shard for it: the 4 main shards with their 222,718
mostly 25 KiB expert tensors, the 26 PLE shards (the target skips those via
patch 13's `weight_files_to_skip`; the draft class had no such hook), and
the shared-expert file. The per-tensor cost that makes the target load slow
(see the runbook's Load time section) applies to every tensor the iterator
yields, whether or not the model keeps it.

## Fix

`Qwen4ExpForCausalLMMTP.weight_files_to_skip`, the hook the loader already
consults through `Source.init_new`: read `model.safetensors.index.json`,
skip every indexed file that has no key containing `mtp`. Files not in the
index (an out-of-index `mtp.safetensors`, the case `maybe_add_mtp_safetensors`
exists for) are kept, and without an index the hook returns nothing. Logs
`MTP draft: skipping 31 of 34 checkpoint shard files without mtp tensors`.

## Verification

- Dry run of the hook against the DERISKED index: keeps the three
  `model-mtp-merged-rest*` shards and an un-indexed `mtp.safetensors`.
- Boot: draft `Load weight end elapsed` 49.5 s → see the runbook's Load
  time table for the measured value.
- Greedy output (`temperature 0`, 8 prompts × thinking on/off, 160 tokens)
  identical before and after: the same 31 tensors are loaded from the same
  files.
