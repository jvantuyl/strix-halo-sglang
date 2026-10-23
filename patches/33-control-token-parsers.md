# Patch 33: the reasoning and tool-call parsers match control tokens by id

**Files:** `python/sglang/srt/parser/control_token_stream.py` (new),
`python/sglang/srt/parser/reasoning_parser.py`,
`python/sglang/srt/function_call/base_format_detector.py`,
`python/sglang/srt/function_call/function_call_parser.py`,
`python/sglang/srt/function_call/qwen3_coder_detector.py`,
`python/sglang/srt/entrypoints/openai/serving_chat.py`
**Script:** [`patch_control_token_parsers.py`](patch_control_token_parsers.py)

## Symptom

The output half of patch 32's problem. With the prompt fixed, the model
still has to *write* about the chat format when an agent asks it to
(review a template, explain a transcript, quote a tool result). It spells
`</think>` with ordinary tokens (`</`, `think`, `>`), the detokenizer
produces the same eight characters, and `--reasoning-parser qwen3` ends
the thinking block there: the rest of the reasoning streams as `content`.
`--tool-call-parser qwen3_coder` has the same blind spot: a spelled-out
`<tool_call>` opens a call, and a `</think>` inside a tool parameter (a
`sed` expression, a heredoc) cuts the call in half. Reproduced in the
test below from the real tokenizer and the real parsers.

## Cause

Both parsers work on text. `BaseReasoningFormatDetector` tests
`self.think_end_token in text`; `Qwen3CoderDetector` runs
`tool_call_regex` over the text and `startswith(self.tool_call_start_token)`
on its buffer. The control tokens are single added tokens with their own
ids, and the serving layer already receives every chunk's `output_ids`
beside its `text`, but the parsers are never shown the ids.

Upstream has started on this for other models (sglang #45458 / #45587,
`ControlTokenStream` for GLM and DeepSeek, not at this pin and not wired
to the Qwen3 parsers); vLLM #45915 did the same with its Qwen3 adapter
still text-based.

## Fix

A `ControlTokenStream` per choice that decodes the request's output ids
beside the text the parsers receive, and a *view* the parsers search
instead of the text.

- `parser_markers(reasoning_parser, tool_call_parser)` asks the configured
  detector classes for their marker strings (`<think>`, `</think>`,
  `<tool_call>`, `</tool_call>` for qwen3 + qwen3_coder);
  `resolve_control_ids(tokenizer, markers)` keeps those that are single
  tokens whose decode round-trips. A marker that is not a single token is
  matched as text, as before.
- `ControlTokenStream.feed(output_ids, cumulative)` runs the ids through
  `tokenizers.decoders.DecodeStream` (honouring the request's
  `skip_special_tokens`) and records the character offset at which each
  control id's text begins. `consume(text)` records the text handed to the
  first parser; `set_tail_lag(n)` tells it that the next parser's text ends
  `n` characters earlier (the reasoning parser holds back a possible
  partial marker before passing text on). `view(text)` checks that `text`
  is a suffix of the consumed text and a prefix of the decoded ids where
  both exist, then returns a same-length copy in which the first character
  of every marker occurrence that does not begin at a control id is
  replaced by `\x00`. Indices found in the view slice the real text. The
  view is never returned to a client; nothing is rewritten.
- If the text ever disagrees with the decoded ids, or no ids were fed, the
  stream deactivates with one warning and `view` returns the text
  unchanged: upstream's text matching for the rest of that output.
- `BaseReasoningFormatDetector` and `BaseFormatDetector` get `_control`
  and `_marker_view(text)`; every marker search in
  `_detect_and_parse_impl`, `_parse_streaming_increment_impl` and
  `_strip_leading_think_start` goes through the view. `Qwen3CoderDetector`
  finds `<tool_call>`/`</tool_call>` in the view (one-shot `finditer`
  slicing the text, streaming `startswith` on `buffer_view`, and the
  potential-tag probe, so a complete spelled-out tag at the end of the
  buffer is not held forever). A bare `<function=` is still parsed as a
  call, as upstream does; only the framing is fixed.
- `ReasoningParser.attach_control_stream`, `held_back_len()` and
  `FunctionCallParser.attach_control_stream`. `serving_chat.py` resolves
  the ids in `__init__` (logs `(patch 33)` once), creates one stream per
  choice in the streaming loop (`feed` with the chunk's `output_ids`,
  cumulative unless `incremental_streaming_output`; `consume(delta)`;
  attach to the reasoning parser; `set_tail_lag(held_back_len())`; attach
  to the tool-call parser) and once in the non-streaming path after
  `_decode_response`.

## Verification

[`tests/check_control_token_parsers.py`](../tests/check_control_token_parsers.py)
(148 checks; runs in the image, see its header for the `docker run` line)
builds model output from ids (content as ordinary tokens, real control ids
between), runs them through a mirror of the detokenizer's incremental loop
under 11 chunkings and hands text and ids to the parsers the way
`serving_chat.py` does: quoted `</think>`, `<think>`, `<tool_call>` and
`</tool_call>` in reasoning and in content stay where they were; a real
tool call with `</think>` inside a parameter parses whole; streaming and
one-shot agree with each other and with the intended split for every
chunking, with `skip_special_tokens` on and off; the same output without
the stream reproduces the early exit (the bug); a stream whose text
disagrees with its ids falls back to text matching and equals the
no-stream result; a real `<think>` id with forced reasoning off is still
stripped; a spelled-out `<tool_call>` block is content.

Live: a Droid-style request (an agent reading a file that contains
`<|im_start|>` and discussing `</think>` in its thinking) keeps
`reasoning_content` whole and the quotes in the text; tool calls at
`temperature` 0 and 0.7 still return `tool_calls`.
