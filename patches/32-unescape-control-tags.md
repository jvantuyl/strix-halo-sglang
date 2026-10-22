# Patch 32: un-escape control strings in chat completion output

**Files:** `python/sglang/srt/entrypoints/openai/serving_chat.py`,
`python/sglang/srt/entrypoints/openai/serving_completions.py`,
`python/sglang/srt/entrypoints/http_server.py`, new
`python/sglang/srt/entrypoints/openai/control_tag_unescape.py`
**Script:** [`patch_unescape_control_tags.py`](patch_unescape_control_tags.py)
**Switch:** `SGLANG_UNESCAPE_CONTROL_TAGS=1` (the launcher sets it when it
mounts the repo template; off otherwise)
**Pairs with:** item 3 of [`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja)

## Symptom

Under Droid, the model's thinking closed mid-sentence and the rest of the
reasoning streamed as `content`, with a stray `</think>` later in the
answer. The screenshot that pinned it down: the model was quoting a file
it had just written that discussed `<|im_start|>` / `<|im_end|>`; the
thinking ended exactly at the second string.

## Cause

The server renders the chat template to one string and tokenizes it with
special tokens enabled. A tool result that merely *mentions* `<|im_end|>`
therefore put the real EOS token (id 248046) into the prompt,
indistinguishable from the structural ones (checked: position 95 of the
rendered scenario). The model, quoting the file back inside `<think>`,
predicted that very token; strict thinking masks it there, so the next-best
token, `</think>`, was sampled. The reasoning parser cannot tell that
`</think>` from a real one. `</think>` (248069) and `<tool_call>` (248058)
in content caused the same class of problem for the parsers that key on
them. A system-prompt rule not to write the strings did not hold against
"quote the file accurately".

## Fix

Two halves, one scheme:

1. **Template (input).** `escape_control` in the repo template HTML-entity
   escapes nine strings inside message content before tokenization:
   `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`,
   `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`
   (`&lt;…&gt;`). A string that already arrives as an entity (an HTML file
   quoting the tags) is encoded one level further (`&amp;lt;…&amp;gt;`). The
   change has to be *inside* the string: the tokenizer matches added tokens
   wherever they occur, so a backslash prefix leaves the control token
   intact (checked; that was the first attempt).
2. **This patch (output).** After the reasoning and tool-call parsers have
   run, decode one level in `content`, `reasoning_content` and tool-call
   `arguments`: `&lt;X&gt;` → `<X>`, `&amp;lt;X&amp;gt;` → `&lt;X&gt;`, for
   those nine strings only. Other entities (`&lt;div&gt;`, `&amp;`) are
   never touched. Streaming holds back a tail that could still grow into
   an escaped tag (at most the longest pattern minus one character) until
   the next delta settles it, and flushes whatever is left into the
   choice's finish chunk (as a `tool_calls` arguments delta when it belongs
   to a tool call). Non-streaming decodes the finished message.

Together they are the identity on client text: raw control strings and
their entity forms both come back as sent, including inside the arguments
of an `Edit` against a file that contains them; the model only ever sees
and produces the escaped forms. The remaining ambiguity is a
double-encoded form (`&amp;lt;/think&amp;gt;`) arriving in input, which
comes back one level down.

Placement matters: the decode wraps the SSE generator in
`_handle_streaming_request` (so the first chunk's validation error still
propagates as an HTTP 400) and the built response in
`_handle_non_streaming_request`. `/v1/completions` gets the same pair
(each choice's `text`), and `/generate` decodes the `text` field of every
output: cumulative streaming text (the default) is decoded up to a possible
dangling entity prefix so each chunk's text stays a prefix of the next,
`--incremental-streaming-output` text goes through the per-index
hold-back, a finished output is decoded whole. Those two endpoints take a
raw prompt, so the template's input-side encoding never ran for them: an
entity that was literally in the prompt and is echoed back comes out one
level down. Spelled-out control strings in output (the thing patch 33
escapes) are far more common than that, which is why the decode is on for
them too. The Anthropic, Ollama and Responses endpoints are not decoded. Decoding any earlier, in the detokenizer
or before `_process_reasoning_stream`, would hand the parsers a literal
`</think>` again.

This half assumed the model would echo the entities it was shown. It does
not: it spells the strings out as plain tokens, which the text-based
parsers match like the real ones. [Patch 33](33-escape-control-text.md)
has the detokenizer rewrite those to the same entities, so this decode
covers both.

## Verification

- [`tests/check_chat_template.py`](../tests/check_chat_template.py): 50
  renderings byte-identical to stock, 30 renderings where template encode
  followed by the one-level decode is the identity and the tokenized prompt
  holds exactly the structural control tokens.
- [`tests/check_control_tag_unescape.py`](../tests/check_control_tag_unescape.py)
  (host, no image): 95 checks; every 2- and 3-way split of every pattern,
  character-at-a-time and 300 random chunkings decode identically to the
  whole text; SSE rewriting limited to the three fields with the tail
  flushed into the finish chunk; completion SSE events and responses;
  `/generate` cumulative (prefix-stable), incremental and finished outputs;
  tag list equal to the template's.
- Live: pending the rebuild (see the runbook row for `--chat-template`).
