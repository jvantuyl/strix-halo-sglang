#!/usr/bin/env python3
"""Let a request's reasoning_effort beat --default-chat-template-kwargs (patch 29).

`--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` is the
server-wide default. A request that asks for another level, either as the
top-level OpenAI `reasoning_effort` field or inside `chat_template_kwargs`,
should win. Upstream drops it:

  1. `_convert_to_internal_request` pops `reasoning_effort` out of
     `request.chat_template_kwargs` into `request.reasoning_effort`.
  2. `_process_messages` then `setdefault`s the server defaults into the
     kwargs, so the popped slot is refilled with the default.
  3. `extra_template_kwargs` is built as the request's effort first, then
     `.update(request.chat_template_kwargs)`, so the default overwrites it.

Every request renders at the default level; `prompt_tokens` and the
reasoning trace are identical for low / medium / xhigh. This seeds the
merge with the request's own effort before the defaults go in, so the
default is only used when the request did not set one. Same code on
upstream main at the time of writing.

The anchor is asserted, so an upstream rewrite fails the build instead of
silently reverting. See patches/29-default-effort-override.md.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/entrypoints/openai/serving_chat.py"
text = open(p).read()
old = """        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
"""
assert text.count(old) == 1, "serving_chat.py: default_chat_template_kwargs merge anchor not found"
new = """        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            # gfx1151 patch 29: the request's effort was already hoisted out
            # of chat_template_kwargs into request.reasoning_effort; put it
            # back first so the server default below cannot displace it.
            if (
                request.reasoning_effort is not None
                and "reasoning_effort" in self.default_chat_template_kwargs
            ):
                ctk.setdefault("reasoning_effort", request.reasoning_effort)
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
