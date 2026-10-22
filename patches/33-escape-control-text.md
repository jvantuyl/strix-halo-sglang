# Patch 33: entity-escape spelled-out control strings in detokenizer output

**Files:** `python/sglang/srt/managers/detokenizer_manager.py`, new
`python/sglang/srt/managers/control_tag_escape.py`
**Script:** [`patch_escape_control_text.py`](patch_escape_control_text.py)
**Switch:** `SGLANG_UNESCAPE_CONTROL_TAGS=1` (shared with patch 32; the
launcher sets it when it mounts the repo template)
**Pairs with:** [patch 32](32-unescape-control-tags.md) and item 3 of
[`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja)

## Symptom

With patch 32 live, the thinking no longer ended at a quoted `<|im_end|>`:
`reasoning_content` carried the whole quote. But the answer did not. Asked
to repeat a note that contained the control strings, the model returned
`content` = `" and "`; without tools it returned text with a literal
`</think>` in the middle. The token dump (`return_token_ids`) showed why:
the model does not echo the entities it was shown. It writes the strings
as ordinary tokens, `' </'`, `'think'`, `'>'`, `' <'`, `'tool'`, `'_call'`,
`'>'`, `' <|'`, `'im'`, `'_end'`, `'|'`, `'>'`. Only one real `</think>`
(the end of the thinking) and the final `<|im_end|>` were in the output.

## Cause

The reasoning parser (`qwen3`) and the tool-call parser (`qwen3_coder`)
work on detokenized text. A spelled-out `</think>` is the same string as
the real token's text, so the parser closes the reasoning block there; a
spelled-out `<tool_call>` opens a tool call. Patch 32 cannot help: it
decodes entities, and there were none. Nothing in the serving layer knows
which `</think>` came from token 248069 and which from three plain tokens.

## Fix

The detokenizer knows. For each row whose text contains `<`, decode the
ids in runs between the real control token ids (`<|im_start|>`,
`<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`, `<tool_call>`,
`</tool_call>`, `<tool_response>`, `</tool_response>`), rewrite
`<X>` → `&lt;X&gt;` for those nine strings inside each run, and keep
whatever the tokenizer gives for the real tokens themselves (their literal
text, or nothing under `skip_special_tokens`). The parsers then only ever
match genuine tokens, and patch 32 turns the entities back into the raw
strings for the client, so the model's `content` reads `</think>` where it
wrote `</think>`. Rows without `<` are not re-decoded.

Streaming: a trailing proper prefix of a control string (`</thi`, at most
the longest string minus one character) is held back the same way the
existing code holds an incomplete UTF-8 sequence, so the committed text
never ends inside a string and the rewrite is identical for any chunking.
A stop string the scheduler matched on the raw text is looked up as sent
first (it may be the real token's literal text) and in its escaped form
otherwise, so the trim still finds it.

Byte-level BPE makes the run-wise decode equal to the whole decode (added
tokens are atomic, nothing merges across them); a SentencePiece tokenizer
that strips a leading space at the start of a decode would not be, which
is one reason the switch is off by default. Patch 32 decodes the entities
again for chat completions, `/v1/completions` and `/generate`; the
Anthropic, Ollama and Responses endpoints would see the escaped form.

## Verification

- [`tests/check_control_tag_escape.py`](../tests/check_control_tag_escape.py)
  (75 checks on the host, 96 inside the image, where the real patched
  `DetokenizerManager` runs the same scenarios as the host's mirror of it):
  escape and hold-back arithmetic; the incremental loop emits the same escaped text one-shot,
  token-at-a-time and for 300 random chunkings, holds back a partial
  string, keeps the real `</think>` literal, trims a stop string in either
  form, leaves rows without a control string untouched, and is the identity
  with the switch off; on the real Qwen tokenizer, run-wise decode equals
  whole decode for 200 random sequences with real control tokens inserted,
  and the model's own spelling of the strings round-trips.
- Live: see the runbook row for `--chat-template`.
