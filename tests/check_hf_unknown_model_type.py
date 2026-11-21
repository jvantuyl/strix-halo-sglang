"""Checks for patch 47: unregistered model_type fallback in the HF config loader.

Runs in-image (needs sglang + transformers):
    PYTHONPATH=/sgl-workspace/sglang/python python3 tests/check_hf_unknown_model_type.py
"""

import json
import os
import sys
import tempfile

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def main():
    import transformers
    from sglang.srt.utils.hf_transformers.config import get_model_config_parser

    # Behavior gates on the installed transformers version: strict mode
    # (>=5.12) rejects unregistered types; 5.5 tolerated them. Either way the
    # parser must return a config with the checkpoint's architectures.
    strict = int(transformers.__version__.split(".")[1]) >= 12
    print(f"transformers {transformers.__version__} (strict={strict})")

    parser = get_model_config_parser("hf")

    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "model_type": "qwen4_exp",  # real: known to sglang, NOT to transformers
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "text_config": {"nested": True},
        }
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        config = parser.parse(d, trust_remote_code=False)
        check(
            "unknown-type-parse",
            config is not None
            and config.architectures == ["Qwen4ExpForConditionalGeneration"],
        )
        check("nested-preserved", getattr(config, "text_config", None) is not None)
        # registered types keep the normal path (transformers builtin)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({**cfg, "model_type": "llama"}, f)
        config2 = parser.parse(d, trust_remote_code=False)
        check(
            "registered-type-unaffected",
            config2 is not None and config2.model_type == "llama",
        )
        # malformed json must still raise (no silent swallowing)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{not json")
        try:
            parser.parse(d, trust_remote_code=False)
            check("malformed-raises", False, "no exception")
        except Exception:
            check("malformed-raises", True)

    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL PASS'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
