#!/usr/bin/env python3
"""Build a tiny Qwen4-Exp checkpoint directory (config + tokenizer, no weights)
from the real cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 config, for a
``--load-format dummy`` end-to-end smoke test on gfx1151.

Kept at production geometry where kernels care about it:
  - full attention: 12 q heads / 1 kv head, head_dim 256 (QSA decode contract)
  - group_size 32 asymmetric AWQ (WNA16 Triton MoE + zero points)
  - PLE: 16 n-gram heads, fp8 storage, file-backed offload
Everything else is shrunk so the whole thing fits in a few hundred MiB.

usage: make_mini_qwen38_config.py SRC_DIR DST_DIR
"""
import json
import os
import shutil
import sys

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def main(src: str, dst: str) -> None:
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)

    tc = cfg["text_config"]
    n_layers = 4
    tc.update(
        hidden_size=512,
        num_hidden_layers=n_layers,
        layer_types=tc["layer_types"][:n_layers],  # lin, lin, lin, full
        ple_layer_ids=[2],
        ple_embed_dim=512,
        ngram_vocab_size_base=20000,
        ple_embedding_dtype="float8_e4m3fn",
        hc_lowrank=64,
        num_attention_heads=12,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        max_position_embeddings=8192,
    )
    tc["mtp"]["num_hidden_layers"] = 1

    vc = cfg["vision_config"]
    vc.update(
        depth=2,
        hidden_size=256,
        num_heads=4,
        intermediate_size=512,
        out_hidden_size=tc["hidden_size"],
    )

    # keep the quant config; ignore entries for missing modules never match
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    for name in TOKENIZER_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, name))
    print(f"wrote mini config to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
