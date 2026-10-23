# Patch 32: content can never become a control token

**Files:** `python/sglang/srt/utils/control_token_encoder.py` (new),
`python/sglang/srt/entrypoints/openai/serving_chat.py`,
[`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja)
**Script:** [`patch_control_token_encoder.py`](patch_control_token_encoder.py)

## Symptom

Under an agent (Droid, Hermes, anything that feeds file contents and shell
output back as tool results) the model kept leaving `<think>` early, often
mid-sentence, with the rest of its reasoning streamed as `content`. It was
worst when the tool results quoted the chat format itself: a grep through a
Jinja template, a tokenizer config, a transcript. A direct probe confirmed
the two halves: a user message containing the literal text `<|im_start|>`
or `</think>` reached the model with that text **gone** (the decoded prompt
showed the control token in its place), and with
`--enable-strict-thinking` the request ended with `finish_reason: stop` and
no `content` at all.

## Cause

`OpenAIServingChat` renders the chat template to one string and tokenizes
it with the added-token table enabled (whichever of its three paths it
takes: `apply_chat_template(tokenize=True)`, the tokenizer manager's
re-tokenization of the rendered text, or the multimodal processor). The
tokenizer cannot tell the template's `<|im_end|>` from the same six
characters inside a tool result, so both become id 248046, the EOS. The
same for `</think>` (248069), `<tool_call>` (248058), `<tool_response>`
(248066) and the rest of the 33 added tokens.

A quoted `</think>` inside the thinking block is then a real
end-of-thinking token as far as the model's context goes; a quoted
`<|im_end|>` is an end of turn. With strict thinking on,
`ReasonerGrammarObject` masks EOS until it has seen `</think>`, and the
model, whose turn structure is already broken, produces one to escape;
the reasoning parser switches streams at that point.

This is the same class of bug the ecosystem is converging on: transformers
#47386 (merged) adds `apply_chat_template(sanitize_control_tokens=True)`,
tokenizers #2225 is open, SGLang #30962 (input side) was closed unmerged,
vLLM #45915 merged for the output side. None of them is in this image's
pinned SGLang, and the sanitizing ones rewrite the content (HTML entities),
which an agent then greps for and does not find. This patch does not
rewrite anything.

## Fix

Keep structure and content apart at the tokenizer, not in the text.

1. **The template marks its control tokens.** Every control token the repo
   template emits (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`,
   `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`,
   `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>`)
   is written as `\ue000<name>\ue001` (private-use code points, 22 sites,
   including the `<tool_call>` examples in the tool-format instruction so
   the ids stay identical to stock). The `startswith`/`endswith` checks on
   message content are untouched; they test the client's text.
2. **A split encoder.** `ControlTokenEncoder.from_tokenizer` returns `None`
   unless the loaded template contains a mark (then nothing changes). With
   marks, it builds a second `tokenizers.Tokenizer` from the same JSON with
   `added_tokens` emptied, and refuses a template that marks a name the
   tokenizer has no added token for. `encode(rendered)` scans the rendered
   string: a marked name becomes its added-token id, everything between
   marks goes through the plain tokenizer, so a quoted `<|im_end|>` is the
   ordinary BPE pieces `<`, `|`, `im`, `_end`, `|`, `>`.
3. **Request strings cannot forge a mark.** Before rendering, every string
   in `messages`, `tools` and the template kwargs has its U+E000/U+E001
   doubled (`escape_request`); the encoder folds a doubled sentinel back to
   the single character and treats a stray single one as text. The model
   reads exactly the characters the client sent.
4. **Hooks in `serving_chat.py`.** `__init__` builds the encoder and logs
   `(patch 32)` once; `_render_and_encode_chat_template` gets a first
   branch (escape, render with `tokenize=False`, split-encode);
   `_append_assistant_prefix_to_prompt_ids` (`continue_final_message`)
   encodes the prefix as plain text; `_engine_prompt` prefers the ids over
   the rendered text whenever the encoder exists, so the tokenizer manager
   does not re-tokenize (this model reports multimodal, which otherwise
   routes the text back through the tokenizer). A request that carries
   images or video keeps the ids too since [patch 34](34-control-token-media.md);
   audio still takes the processor's text path, with a warning.

Prompts whose content never spells a control token encode to the same ids
as before, so prefix caching and model behaviour are unchanged. The
`/generate`, embedding and Ollama-compat paths call `apply_chat_template`
directly and are not covered; `/v1/chat/completions` is.

## Verification

[`tests/check_control_token_encoder.py`](../tests/check_control_token_encoder.py)
(66 checks, host and in the image) against the real tokenizer: clean
conversations (system, tools, multi-turn tool calls and results,
`reasoning_content`, `enable_thinking=false`, `default_system_prompt`,
every `reasoning_effort`) encode byte-identically to the stock template's
whole-string tokenization; content quoting every control string, and
content trying to forge a mark with the sentinels, adds no control id
beyond the template's own, and the decoded ids reproduce the original text;
the escape round trip is lossless; a template marking an unknown name is
refused. [`tests/check_chat_template.py`](../tests/check_chat_template.py)
strips the marks before comparing to stock (50 renderings).

Live, with the template mounted and the patched server: a user message
quoting `<|im_start|>` and `</think>` comes back quoted in the model's
answer, `prompt_tokens` for a clean prompt is unchanged, and the
`continue_final_message` prefix lands as text.
