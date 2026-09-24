#!/usr/bin/env python3
"""MTP draft loads only the shards that hold MTP tensors (patch 30).

`Qwen3_5ForCausalLMMTP.load_weights` drops every checkpoint name without
"mtp" in it (the draft's embedding and head are placeholders the target
later shares in), yet the draft's loader walked all 34 shards of the
Qwen3.8-Flash-Next checkpoint: 222,718 mostly 25 KiB expert tensors plus
26 PLE shards, to keep 31 tensors that live in three files. That walk is
49 s of the boot, right after the 152 s target load did the same walk for
real.

The loader already asks the model for shards it may skip
(`weight_files_to_skip`, the hook patch 13 uses on the target for PLE
shards). This gives `Qwen4ExpForCausalLMMTP` one that reads the
safetensors index and skips every file without a single "mtp" tensor.
Files absent from the index are kept. Without an index: no-op.

The anchor is asserted, so an upstream move fails the build instead of
silently walking the checkpoint again. See patches/30-draft-mtp-shards.md.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/models/qwen4_exp_mtp.py"
text = open(p).read()

old_imports = "import copy\nimport logging\n"
assert text.count(old_imports) == 1, "qwen4_exp_mtp.py: import anchor not found"
text = text.replace(old_imports, "import copy\nimport json\nimport logging\nimport os\n", 1)

old = "    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:\n"
assert text.count(old) == 1, "qwen4_exp_mtp.py: _init_pre_fc_norms anchor not found"
new = '''    def weight_files_to_skip(self, hf_folder: str, weight_files: list) -> list:
        """Checkpoint shards without an "mtp" tensor (gfx1151 patch 30).

        load_weights discards every other name, so walking those shards is
        pure overhead. Decided from the safetensors index; files the index
        does not list are kept.
        """
        try:
            with open(os.path.join(hf_folder, "model.safetensors.index.json")) as f:
                weight_map = json.load(f)["weight_map"]
        except (OSError, KeyError, ValueError):
            return []
        indexed = set(weight_map.values())
        with_mtp = {file_name for name, file_name in weight_map.items() if "mtp" in name}
        skipped = [
            wf
            for wf in weight_files
            if os.path.basename(wf) in indexed and os.path.basename(wf) not in with_mtp
        ]
        if skipped:
            logger.info(
                "MTP draft: skipping %d of %d checkpoint shard files without mtp tensors",
                len(skipped),
                len(weight_files),
            )
        return skipped

''' + old
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
