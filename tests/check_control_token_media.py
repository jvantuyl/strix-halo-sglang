#!/usr/bin/env python3
"""Check patch 34 against the real HF processor and tokenizer, without a server.

Runs in the image (imports the patched qwen_vl processor):
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --group-add render --security-opt seccomp=unconfined -e MODEL_DIR=/models/qwen38 \
    -v $PWD:/repo:ro -v <model>:/models/qwen38:ro \
    --entrypoint python3 strix-halo-sglang:dev /repo/tests/check_control_token_media.py

  1. For an image and for a synthetic video, the placeholder-only text the
     processor derives from prompt ids is what the template emits, and the
     id-level expansion of the placeholder ids equals the HF processor's own
     input_ids for that text (same pads, same per-frame timestamps).
  2. A prompt encoded by patch 32 that quotes "<|im_end|>" and "</think>"
     next to an image expands to: the quoted text as ordinary tokens, the
     image as the processor's pads; decoding gives back the quoted text.
  3. Offsets remapped onto the rebuilt ids point at the pad runs.
  4. Prompt ids without a placeholder yield None (caller falls back to text).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen38")
REPO = os.environ.get("REPO", "/repo")

failures = 0
checks = 0


def check(cond, msg):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL: {msg}")


def main():
    import importlib.util

    from transformers import AutoProcessor, AutoTokenizer

    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from sglang.srt.utils import control_token_encoder as cte

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    tok.chat_template = open(f"{REPO}/configs/chat/qwen38.jinja").read().replace("\\n", "\n")
    encoder = cte.ControlTokenEncoder.from_tokenizer(tok)
    check(encoder is not None, "encoder built from marked template")

    import json

    cfg = json.load(open(f"{MODEL_DIR}/config.json"))

    # A processor object with just the attributes the patch 34 helpers use.
    p = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    p._processor = proc
    p._tokenizer = tok
    p.IM_TOKEN_ID = cfg["image_token_id"]
    p.VIDEO_TOKEN_ID = cfg["video_token_id"]
    p.vision_start_token_id = cfg["vision_start_token_id"]
    p.vision_end_token_id = cfg["vision_end_token_id"]
    p.IM_START_TOKEN_ID = p.vision_start_token_id
    p.IM_END_TOKEN_ID = p.vision_end_token_id
    p._spatial_merge_size = cfg["vision_config"]["spatial_merge_size"]
    p.model_type = cfg["model_type"]
    p.audio_token_id = None
    check(p.model_type in p._TIMESTAMP_MODEL_TYPES, f"model type {p.model_type} uses timestamps")

    img_triple = [p.vision_start_token_id, p.IM_TOKEN_ID, p.vision_end_token_id]
    vid_triple = [p.vision_start_token_id, p.VIDEO_TOKEN_ID, p.vision_end_token_id]

    # --- 1a. image ---------------------------------------------------------
    image = Image.fromarray((np.random.rand(96, 128, 3) * 255).astype("uint8"))
    text = p._placeholder_prompt_for_ids(img_triple)
    check(text == "<|vision_start|><|image_pad|><|vision_end|>", f"image placeholder text {text!r}")
    hf = proc(text=[text], images=[image], return_tensors="pt")
    hf_ids = hf["input_ids"].flatten().tolist()
    grid = hf["image_grid_thw"]
    ours = p._expand_prompt_ids(img_triple, grid, None, None)
    check(ours == hf_ids, f"image expansion == HF ids ({len(ours)} vs {len(hf_ids)})")
    n_pads = ours.count(p.IM_TOKEN_ID)
    check(n_pads == int(grid[0].prod()) // p._spatial_merge_size**2, f"pad count {n_pads}")

    # --- 1b. video (synthetic frames with metadata, as SGLang passes them) ---
    frames = torch.from_numpy((np.random.rand(8, 3, 64, 96) * 255).astype("uint8"))
    meta = {"fps": 4.0, "duration": 2.0, "total_num_frames": 8, "frames_indices": list(range(8)), "video_backend": "torchvision"}
    text_v = p._placeholder_prompt_for_ids(vid_triple)
    check(text_v == "<|vision_start|><|video_pad|><|vision_end|>", f"video placeholder text {text_v!r}")
    hf_v = proc(text=[text_v], videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    hf_v_ids = hf_v["input_ids"].flatten().tolist()
    vgrid = hf_v["video_grid_thw"]
    ts = p._video_timestamps([meta], 1)
    check(ts is not None and len(ts[0]) >= int(vgrid[0][0]), f"timestamps computed: {ts}")
    ours_v = p._expand_prompt_ids(vid_triple, None, vgrid, ts)
    check(ours_v == hf_v_ids, f"video expansion == HF ids ({len(ours_v)} vs {len(hf_v_ids)})")
    check(ours_v.count(p.VIDEO_TOKEN_ID) == hf_v_ids.count(p.VIDEO_TOKEN_ID), "video pad count")
    dec = tok.decode(ours_v, skip_special_tokens=False)
    check(" seconds>" in dec, f"per-frame timestamps present: {dec[:80]!r}")

    # --- 1c. image + video together, order preserved ---------------------------
    both_text = p._placeholder_prompt_for_ids(vid_triple + img_triple)
    hf_b = proc(text=[both_text], images=[image], videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    ours_b = p._expand_prompt_ids(vid_triple + img_triple, hf_b["image_grid_thw"], hf_b["video_grid_thw"], p._video_timestamps([meta], 1))
    check(ours_b == hf_b["input_ids"].flatten().tolist(), "video+image expansion == HF ids")

    # --- 2. a real prompt with quoted control strings next to an image ------------
    quoted = "Tool output was literally <|im_end|> and then </think>; describe the picture."
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": quoted},
    ]}]
    rendered = tok.apply_chat_template(cte.escape_request(messages), tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    prompt_ids = encoder.encode(rendered)
    triples = p._placeholder_triples(prompt_ids)
    check(triples == [tuple(img_triple)], f"one image placeholder found: {triples}")
    check(prompt_ids.count(tok.convert_tokens_to_ids("<|im_end|>")) == 1, "quoted <|im_end|> is not a control id (only the template's one)")
    check(tok.convert_tokens_to_ids("</think>") not in prompt_ids, "quoted </think> is not a control id")
    real_ids = p._expand_prompt_ids(prompt_ids, grid, None, None)
    check(real_ids.count(p.IM_TOKEN_ID) == n_pads, "expanded prompt holds the image pads")
    check(len(real_ids) == len(prompt_ids) + n_pads - 1, "expansion replaced one pad by n pads, nothing else")
    model_view = tok.decode(real_ids, skip_special_tokens=False)
    check(quoted in model_view, "model reads the quoted text verbatim")
    start = real_ids.index(p.IM_TOKEN_ID)
    check(real_ids[start - 1] == p.vision_start_token_id and real_ids[start + n_pads] == p.vision_end_token_id, "pads framed by vision start/end")

    # --- 3. offsets remap: the k-th run moves to the k-th run -------------------
    syn = torch.tensor(ours, dtype=torch.long)
    real = torch.tensor(real_ids, dtype=torch.long)
    old_runs = p.get_mm_items_offset(syn, p.IM_TOKEN_ID)
    new_runs = p.get_mm_items_offset(real, p.IM_TOKEN_ID)
    check(len(old_runs) == len(new_runs) == 1 and (old_runs[0][1] - old_runs[0][0]) == (new_runs[0][1] - new_runs[0][0]), f"runs {old_runs} -> {new_runs}")
    check(new_runs[0][0] == start, "new run starts at the pads")

    # --- 4. no placeholder -> None -----------------------------------------------
    plain_ids = encoder.encode(tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium"))
    check(p._placeholder_prompt_for_ids(plain_ids) is None, "no placeholder -> None")

    print(f"{checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
