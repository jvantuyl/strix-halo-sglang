"""Exercise the draft converter's apply path on a tiny two-shard checkpoint."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    converter = Path(__file__).resolve().parents[1] / "tools" / "quantize_draft_moe.py"
    with tempfile.TemporaryDirectory(prefix="draft-moe-test-") as tmp:
        model = Path(tmp)
        layer = "mtp.layers.0."
        experts = layer + "mlp.experts"
        q = layer + "self_attn.q_proj"
        up = layer + "mlp.shared_expert.up_proj"
        ignore = [experts, r"re:mtp\..*", q, up, "lm_head"]
        config = model / "config.json"
        config.write_text(json.dumps({"quantization_config": {"ignore": ignore}}))
        unchanged = torch.ones(32, 32, dtype=torch.bfloat16)
        torch.manual_seed(0)
        save_file(
            {
                experts + ".gate_up_proj": torch.randn(2, 64, 32).bfloat16(),
                q + ".weight": unchanged,
                up + ".weight": unchanged.clone(),
            },
            model / "gate-up.safetensors",
        )
        save_file(
            {experts + ".down_proj": torch.randn(2, 32, 32).bfloat16()},
            model / "down.safetensors",
        )
        index = model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {
            experts + ".gate_up_proj": "gate-up.safetensors",
            experts + ".down_proj": "down.safetensors",
            q + ".weight": "gate-up.safetensors",
            up + ".weight": "gate-up.safetensors",
        }}))
        subprocess.run([sys.executable, str(converter), "--model", tmp, "--apply"], check=True)
        assert json.loads(config.read_text())["quantization_config"]["ignore"] == [
            q, up, "lm_head", layer + "self_attn.qkv_proj",
            layer + "mlp.shared_expert.gate_up_proj",
        ]
        expected = {q + ".weight": "gate-up.safetensors", up + ".weight": "gate-up.safetensors"}
        for expert in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("packed", "scale", "zero_point", "shape"):
                    expected[f"{experts}.{expert}.{proj}.weight_{suffix}"] = (
                        "down.safetensors" if proj == "down_proj" else "gate-up.safetensors"
                    )
        assert json.loads(index.read_text())["weight_map"] == expected
        for shard in ("gate-up.safetensors", "down.safetensors"):
            with safe_open(model / shard, framework="pt") as f:
                assert set(f.keys()) == {key for key, path in expected.items() if path == shard}
                if shard == "gate-up.safetensors":
                    assert torch.equal(f.get_tensor(q + ".weight"), unchanged)
                    assert torch.equal(f.get_tensor(up + ".weight"), unchanged)
    print("Draft MoE conversion: shard names, index, and BF16 exclusions match")


if __name__ == "__main__":
    main()
