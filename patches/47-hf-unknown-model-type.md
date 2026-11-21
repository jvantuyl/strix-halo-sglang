# Patch 47 — unregistered model_type fallback in the HF config loader

Status: **written 2026-11-21**, part of the ROCm 10 base-port groundwork
(see 48). Not required on the 7.13 base; harmless there (the fallback only
fires where `AutoConfig.from_pretrained` raises).

## Problem

`kyuz0/vllm-therock-gfx1151:latest` (ROCm 10.0.0, torch 2.11) ships
transformers 5.20 and the dependency resolver installs 5.12+. There,
`AutoConfig.from_pretrained` rejects model types outside its registry —
`ValueError` ("Unrecognized model ...") in 5.20, `KeyError` from
`CONFIG_MAPPING` in 5.12 — where 5.5 (the 7.13 base) returned a generic
`PretrainedConfig`. Qwen4Exp checkpoints are exactly such an unregistered
type, so nothing loads.

## Fix

In `hf_transformers/config.py`'s `HfModelConfigParser.parse`, wrap the
`AutoConfig.from_pretrained` call in `except (KeyError, ValueError)` and
fall back to `PretrainedConfig.from_dict(json.load(config.json))` — the
same object the old behavior produced. Both exception types are caught
because the message/type varies by transformers version; missing or
malformed config.json surfaces as OSError/JSONDecodeError and still raises.
Registered types keep the full AutoConfig path (no behavior change).

## Tests

`tests/check_hf_unknown_model_type.py` (in-image): unregistered type parses
with architectures and nested text_config preserved; a registered type
(llama) still goes through AutoConfig; malformed config.json still raises.
