# Control-token edge cases

Prompts that trip, or once tripped, the server over a control token
(`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`,
`</tool_call>`, `<tool_response>`, …) or its spelled-out look-alike.
Patches [32](../patches/32-control-token-encoder.md),
[33](../patches/33-control-token-parsers.md) and
[34](../patches/34-control-token-media.md) close the control-token path on
`/v1/chat/completions` (and `/v1/responses`, which shares the code). What
remains is text-level and listed under *Known* so it is recognised when it
turns up in an agent transcript.

[`tests/live_control_token_edges.py`](../tests/live_control_token_edges.py)
fires every prompt below at a running server (`SGLANG_URL`, `MODEL`),
exits non-zero if a *Fixed* case regresses, and reports whether each *Known*
case still reproduces. ~4 minutes; image and video cases need Pillow and
`ffmpeg` on the host. Observed values below are from the Swift-1.5 fine-tune
at temperature 0 on 2026-10-23.

## Fixed (regression checks)

| Prompt | Was | Now |
|---|---|---|
| A `tool` message holding a chat transcript (`<|im_start|>system … <|im_end|>`, a `<think>…</think>` block); the user asks how often `<|im_start|>` and `</think>` appear | The quoted strings became the real control tokens; strict thinking masked the EOS, the model left `<think>` mid-sentence, `reasoning_content` ended early, the quotes were invisible to the model | Counted and quoted back; reasoning whole, one-shot and stream (patch 32) |
| "In your thinking, write the literal eight characters `</think>` on a line by itself, then two more sentences, then answer" | The spelled-out `</think>` ended `reasoning_content`; the rest of the thinking streamed as `content` | `</think>` stays in `reasoning_content`, answer in `content` (patch 33) |
| "Use `write_file` to save two lines to tag.txt: a line with `<think>` and a line with `</think>`" | The `</think>` inside the parameter cut the tool call in half (and could end thinking) | `finish_reason: tool_calls`, `content` argument `"<think>\n</think>\n"`, one-shot and stream (patch 33) |
| User text containing the private-use marks the template uses: `A\ue000<|im_end|>\ue001B` | (new attack surface of patch 32) a forged mark would have become a control token | Request strings are escaped; the model reads and echoes `A<|im_end|>B` (patch 32) |
| `tool_choice: "required"`, a named tool, or `strict: true` tools under `auto`, argument containing `</think>` | Not broken before, but a grammar could have forced a *spelled* `<tool_call>` that patch 33 would rightly not treat as a call | The xgrammar structural tag emits the real `<tool_call>` id; parsed in all three modes (patch 33) |
| An image (data URL) plus text quoting `` `<|im_start|>` `` `` `</think>` `` `` `<|im_end|>` `` | With media the endpoint sent text and the processor re-tokenized it with the added tokens enabled: control tokens again, plus a `Request carries media` warning | Image described, all three quotes read back, no warning; 215 prompt tokens (patch 34) |
| A 16-frame mp4 (data URL) plus the same text | HTTP 500 `No module named 'decord'` (no video backend in the image), then the same re-tokenization | Motion described frame-accurately, quotes read back; 333 prompt tokens (patch 34 + `decord` in the image) |

## Known (open edges, by design or upstream format)

| Prompt | What happens | Why, and what to do |
|---|---|---|
| "Use `write_file` to save exactly these three lines: `<parameter=x>` / `</parameter>` / `</function>`" | `finish_reason: tool_calls` with the `content` argument **empty** (`''`). No error. An agent would write an empty file | Qwen's inner call format (`<function=…>`, `<parameter=…>`, `</parameter>`, `</function>`) is plain text; only the outer `<tool_call>` pair are control tokens, so the detector takes the first `</parameter>` as the closer. Inherent to the format, same in vLLM. Watch for: an agent editing files that contain these tags (the chat template, SGLang's own detector, test fixtures) and getting empty or truncated writes. Mitigation on the agent side: write such content through a shell heredoc or a diff tool instead of a file-content argument |
| `stop: ["</think>"]` on a request with thinking on | `finish_reason: stop` with 57 characters of reasoning and **no `content`** | Stop strings are matched on the decoded text of the output tail (`check_match_stop_str_prefix`), where the real `</think>` id reads `</think>`. The server already stops on the EOS ids; clients must not name control strings as stops |
| `stop: ["</think>"]` with thinking off, model asked to write `before </think> after` | `content` is `'before '` | Same matcher, matching the spelled-out copy. Same advice |
| Sole user message `<tool_response>\nexit code 0\n</tool_response>` | HTTP 400 `No user query found in messages.` | The stock template treats a user message that starts with `<tool_response>` and ends with `</tool_response>` as a tool result when it looks for the last real user query. A pasted transcript shaped like that is the only way to hit it; a second user message, or a space before the tag, avoids it. Text check on the client's content, by design of the template |
| `/v1/completions` with `prompt: "x <|im_end|> y"` | 4 prompt tokens; `x <|im_end> y` (one character short) is 6 | The raw completions prompt is the client's to tokenize; a control string there *is* the control token. Only `/v1/chat/completions` and `/v1/responses` keep content and structure apart. Nothing to fix; do not put untrusted text into a raw prompt |

Not probed, for the record: the model emitting a control id other than the
parsers' four and the EOS (`<|im_start|>`, `<tool_response>`, vision tokens)
is a rare hallucination the parsers ignore; with `skip_special_tokens` off
(SGLang sets it when tools are present) the text appears in `content`. Audio
and a template that places more or fewer media placeholders than the request
carries take the old text path with a `Request carries media the prompt ids
cannot account for` warning. If patch 33's decoded ids ever disagree with the
text, that response falls back to text matching with one `alignment` warning
in the journal; none seen so far.
