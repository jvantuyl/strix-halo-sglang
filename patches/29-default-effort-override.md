# Patch 29: a request's `reasoning_effort` beats `--default-chat-template-kwargs`

**Files:** `python/sglang/srt/entrypoints/openai/serving_chat.py`
**Script:** [`patch_default_effort_override.py`](patch_default_effort_override.py)

## Symptom

The Qwen 3.8 launcher passes
`--default-chat-template-kwargs '{"reasoning_effort": "medium", ...}'` so
the template's `xhigh` default (and its "think carefully, validate key
assumptions, consider plausible alternatives" preamble) is not injected
into every prompt. With that default in place, a request asking for
another level was ignored: `"reasoning_effort": "xhigh"` at the top level,
or `"chat_template_kwargs": {"reasoning_effort": "low"}`, both returned
`prompt_tokens: 35`, the same as `medium`, with byte-identical reasoning
traces. Rendering the template directly with the same kwargs gives 25 /
51 / 63 tokens for `medium` / `low` / `xhigh` on a one-word prompt, so the
template was fine; the server never handed it the request's value.

## Cause

Three steps in `OpenAIServingChat`, in order:

1. `_convert_to_internal_request` pops `reasoning_effort` out of
   `request.chat_template_kwargs` into `request.reasoning_effort` (the
   top-level field lands there directly).
2. `_process_messages` merges the server defaults into
   `request.chat_template_kwargs` with `setdefault`. The slot emptied in
   step 1 is refilled with the default.
3. `extra_template_kwargs` is built as `{"reasoning_effort":
   request.reasoning_effort}` and then `.update(request.chat_template_kwargs)`,
   so the default from step 2 overwrites the request's value.

Without a server default, step 2 is skipped and the request's value
survives, which is why the bug only shows once `--default-chat-template-kwargs`
carries `reasoning_effort`. Upstream `main` has the same code at the time
of writing.

## Fix

Before the `setdefault` loop, if the request carries a `reasoning_effort`
and the server defaults would supply one, `setdefault` the request's own
value into the kwargs first. The default then only fills requests that did
not set an effort. Kwargs that still hold their own `reasoning_effort`
(the Hunyuan path, where step 1 does not pop) are left alone, and the
other default keys merge as before.

## Verification

Against the served DERISKED checkpoint on this box with the launcher's
defaults (`medium` plus a one-line system prompt), user message `hi`,
`max_tokens: 1`. Before the patch all four rows returned the same
`prompt_tokens` (35 on the longer test message used then); after:

| request | `prompt_tokens` |
|---|---|
| no effort | 33 |
| top-level `reasoning_effort: "xhigh"` | 71 |
| `chat_template_kwargs.reasoning_effort: "low"` | 59 |
| `chat_template_kwargs.reasoning_effort: "medium"` | 33 |
| top-level `low` + `chat_template_kwargs.default_system_prompt: ""` | 41 |

The last row shows the other default key still merging independently. The
`default_system_prompt` override and `enable_thinking: false` behave as
before.
