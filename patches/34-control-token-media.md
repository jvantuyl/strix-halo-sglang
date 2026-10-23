# Patch 34: requests with images or video keep the split-encoded prompt ids

**Files:** `python/sglang/srt/entrypoints/openai/serving_chat.py`,
`python/sglang/srt/multimodal/processors/qwen_vl.py`
**Script:** [`patch_control_token_mm.py`](patch_control_token_mm.py)

## Symptom

Patch 32 covers text-only requests. A request that carries an image or a
video still logged `Request carries media; the prompt text is re-tokenized
and quoted control strings are not kept apart`, and a quoted `<|im_end|>` or
`</think>` in such a request reached the model as the control token again,
with the same consequences (strict thinking masks it, the turn structure
breaks, the model never reads the quote). Agents that attach screenshots
alongside file contents hit it.

## Cause

`_engine_prompt` sends the rendered text, not the ids, whenever the request
has media, because the multimodal processor needs the text: the Qwen-VL
processor (`QwenVLImageProcessor`) hands the prompt to the HF
`Qwen3VLProcessor`, which expands each `<|image_pad|>` into
`prod(grid_thw) / merge_size²` pads, rewrites each `<|video_pad|>` into
per-frame `<t seconds>` text plus pads, and tokenizes the whole string with
the added tokens enabled. Sending the ids did not help either: `load_mm_data`
decodes a list prompt back to text before any of that.

SGLang already has an id-level expansion for exactly this
(`QwenVLImageProcessor.build_input_ids` and
`build_input_ids_with_timestamps`, used by the artifact cache path); it was
not reachable from the chat endpoint's ids.

## Fix

The processor accepts prompt ids and expands the placeholders inside them,
with the media pipeline itself unchanged.

- `QwenVLImageProcessor.accepts_prompt_ids = True`. A list prompt bypasses
  the artifact cache path (which composes on text) and goes to
  `_process_mm_data_uncached`.
- `_placeholder_prompt_for_ids(ids)` derives a placeholder-only text from
  the ids: every `(<|vision_start|>, <|image_pad|> | <|video_pad|>, next)`
  triple decoded from the ids themselves, concatenated. The unchanged
  pipeline (`load_mm_data`, `process_and_combine_mm_data_async`:
  preprocessing, pixel values, grids, timestamps, feature hashing) runs on
  that text.
- `_rebuild_from_prompt_ids` expands the placeholder-only ids with
  `build_input_ids_with_timestamps` (video on a timestamp model type;
  timestamps from the request's `video_metadata` through the HF processor's
  own `_calculate_timestamps`) or `build_input_ids`, and requires the result
  to equal the processor's `input_ids` for the placeholder-only text token
  for token. Only then does it expand the *original* prompt ids the same
  way, remap every item's offsets (the k-th pad run in the processor's ids
  is the k-th pad run in the rebuilt ids, same length) and drop the
  processor's `padded_input_ids` / `mrope_positions` / `mrope_position_delta`
  so they are recomputed from the rebuilt ids.
- On any disagreement (no placeholder, timestamps unavailable, expansion
  differs, run counts or lengths differ, builder raises) the request falls
  back to today's text path with one warning; nothing fails.
- `serving_chat.py`: `_engine_prompt` also prefers the ids when the encoder
  exists, the processor declares `accepts_prompt_ids`, the request has no
  audio, and the ids hold exactly one `(<|vision_start|>, pad)` pair per
  image and per video (`_media_placeholders_match`). Otherwise the text path
  and its warning remain, reworded to say why.

Prompts that quote nothing expand to the same ids as the text path (the
check above enforces it per request), so prefix caching and model behaviour
are unchanged. The media preprocess (artifact) cache is bypassed for the ids
path. Audio keeps the text path.

## Verification

[`tests/check_control_token_media.py`](../tests/check_control_token_media.py)
(21 checks; runs in the image, see its header for the `docker run` line)
with the real `AutoProcessor` and tokenizer: for an image, a synthetic
8-frame video (`fps` 4, `video_metadata`) and video plus image, the
placeholder-only text equals what the template emits and the id-level
expansion equals the HF processor's `input_ids` (pads and per-frame
timestamps); a patch 32 prompt quoting `<|im_end|>` and `</think>` next to
an image expands to the quoted text as ordinary tokens with the image as the
processor's pads, and decodes back to the quoted text; remapped offsets
point at the pad runs; ids without a placeholder yield `None`.

Live: an image request quoting `<|im_end|>` and `</think>` in the text gets
the image described and the quotes read back; no `Request carries media`
warning in the journal.
