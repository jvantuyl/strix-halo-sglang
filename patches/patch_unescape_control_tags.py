#!/usr/bin/env python3
"""Un-escape control strings in chat completion output (patch 32).

The repo chat template (configs/chat/qwen38.jinja) HTML-entity escapes the
serving control strings when they occur inside message content, so a tool
result or user message that quotes <|im_end|> or </think> no longer puts the
real control token into the prompt (where the model would reproduce it and,
inside <think>, fall through the strict-thinking mask to an early </think>).
The model sees and echoes the escaped form. This patch decodes it one level
on the way out, after the reasoning and tool-call parsers have run, so the
client gets the string it sent in, including inside tool-call arguments (a
file the model edits gets the real string back).

Two levels, shared with the template: &lt;X&gt; -> <X> and
&amp;lt;X&amp;gt; -> &lt;X&gt;, for the nine tags in CONTROL_TAGS only.
Other HTML entities are never touched. Streaming holds back a tail that
could be the start of an escaped tag until the next delta (or the choice's
finish chunk) completes or rules it out.

Off unless SGLANG_UNESCAPE_CONTROL_TAGS is set to 1; the launcher sets it
when it mounts the repo template. Hooks: the streaming generator is wrapped
in _handle_streaming_request and the non-streaming response is rewritten in
_handle_non_streaming_request, both in serving_chat.py. The anchors are
asserted, so an upstream rewrite fails the build instead of silently
dropping the decode. See patches/32-unescape-control-tags.md.
"""
import sys

MODULE_SRC = '''\
"""One-level un-escape of HTML-entity encoded control strings in chat output.

gfx1151 patch 32; pairs with the content escaping in the repo chat template.
"""
import os

import orjson

CONTROL_TAGS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


def _entity(tag):
    return "&lt;" + tag[1:-1] + "&gt;"


def _double(tag):
    return "&amp;lt;" + tag[1:-1] + "&amp;gt;"


# Level 1 before level 2. A level-2 form contains no level-1 form (its "&lt;"
# is spelled "&amp;lt;"), so one pass per rule decodes exactly one level.
RULES = tuple((_entity(t), t) for t in CONTROL_TAGS) + tuple(
    (_double(t), _entity(t)) for t in CONTROL_TAGS
)
PATTERNS = tuple(p for p, _ in RULES)
_MAX_HOLD = max(len(p) for p in PATTERNS) - 1


def enabled():
    return os.environ.get("SGLANG_UNESCAPE_CONTROL_TAGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def unescape(text):
    if not text or "&" not in text:
        return text
    for pattern, replacement in RULES:
        text = text.replace(pattern, replacement)
    return text


def _held_suffix_len(text):
    """Longest suffix of text that is a proper prefix of some pattern.

    A complete pattern at the end is not held (it is decoded); anything
    shorter that could still grow into one waits for the next delta.
    """
    for n in range(min(_MAX_HOLD, len(text)), 0, -1):
        suffix = text[-n:]
        if any(len(p) > n and p.startswith(suffix) for p in PATTERNS):
            return n
    return 0


class StreamUnescaper:
    """Per-field decoder for one streamed choice."""

    def __init__(self):
        self.pending = ""

    def feed(self, text):
        s = self.pending + text
        n = _held_suffix_len(s)
        if n:
            self.pending = s[-n:]
            s = s[:-n]
        else:
            self.pending = ""
        return unescape(s)

    def flush(self):
        # Never a complete pattern (feed would have decoded it): emit as is.
        out = self.pending
        self.pending = ""
        return out


def unescape_chat_response(response):
    """Non-streaming: decode content, reasoning_content and tool-call arguments."""
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        if isinstance(getattr(message, "content", None), str):
            message.content = unescape(message.content)
        if isinstance(getattr(message, "reasoning_content", None), str):
            message.reasoning_content = unescape(message.reasoning_content)
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if function is not None and isinstance(
                getattr(function, "arguments", None), str
            ):
                function.arguments = unescape(function.arguments)
    return response


def _finish_pending(states, idx, delta):
    changed = False
    for key, state in states.items():
        if key[0] != idx or not state.pending:
            continue
        rest = state.flush()
        if key[1] == "tool":
            tool_calls = delta.get("tool_calls") or []
            tool_calls.append({"index": key[2], "function": {"arguments": rest}})
            delta["tool_calls"] = tool_calls
        else:
            delta[key[1]] = (delta.get(key[1]) or "") + rest
        changed = True
    return changed


def _parse_event(event):
    """The JSON object of a "data: {...}" SSE event with a choices list, else None."""
    if not (isinstance(event, str) and event.startswith("data: {")):
        return None
    try:
        obj = orjson.loads(event[len("data: "):])
    except Exception:
        return None
    if not isinstance(obj.get("choices"), list):
        return None
    return obj


def _emit_event(obj):
    return "data: " + orjson.dumps(obj).decode() + "\\n\\n"


def unescape_sse_event(event, states):
    """Decode one chat completion SSE event string in place of the original;
    states persists across the stream, keyed by (choice index, field[,
    tool-call index])."""
    obj = _parse_event(event)
    if obj is None:
        return event
    choices = obj["choices"]
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index", 0)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        for field in ("content", "reasoning_content"):
            value = delta.get(field)
            if isinstance(value, str) and value:
                delta[field] = states.setdefault(
                    (idx, field), StreamUnescaper()
                ).feed(value)
                changed = True
        for tool_call in delta.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if isinstance(function, dict) and isinstance(
                function.get("arguments"), str
            ) and function["arguments"]:
                function["arguments"] = states.setdefault(
                    (idx, "tool", tool_call.get("index", 0)), StreamUnescaper()
                ).feed(function["arguments"])
                changed = True
        if choice.get("finish_reason") is not None and _finish_pending(
            states, idx, delta
        ):
            choice["delta"] = delta
            changed = True
    if not changed:
        return event
    return _emit_event(obj)


async def unescape_sse_stream(generator):
    states = {}
    async for event in generator:
        yield unescape_sse_event(event, states)


def unescape_completion_response(response):
    """Non-streaming /v1/completions: decode each choice's text."""
    for choice in getattr(response, "choices", None) or []:
        if isinstance(getattr(choice, "text", None), str):
            choice.text = unescape(choice.text)
    return response


def unescape_completion_sse_event(event, states):
    """Decode one /v1/completions SSE event; states keyed by choice index.
    A held tail is flushed into the choice's finish chunk."""
    obj = _parse_event(event)
    if obj is None:
        return event
    changed = False
    for choice in obj["choices"]:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index", 0)
        text = choice.get("text")
        if isinstance(text, str) and text:
            choice["text"] = states.setdefault(idx, StreamUnescaper()).feed(text)
            changed = True
        if choice.get("finish_reason") is not None:
            state = states.get(idx)
            if state is not None and state.pending:
                choice["text"] = (choice.get("text") or "") + state.flush()
                changed = True
    if not changed:
        return event
    return _emit_event(obj)


async def unescape_completion_sse_stream(generator):
    states = {}
    async for event in generator:
        yield unescape_completion_sse_event(event, states)


def unescape_generate_output(out, states, incremental):
    """/generate: out is one output dict or a list of them. Cumulative text
    (the default) is decoded up to a possible dangling entity prefix, so the
    text of each chunk stays a prefix of the next chunk's; incremental text
    (--incremental-streaming-output) goes through a per-index
    StreamUnescaper; a finished output is decoded whole. Returns a shallow
    copy when the text changes, the same object otherwise."""
    if isinstance(out, list):
        return [unescape_generate_output(o, states, incremental) for o in out]
    if not isinstance(out, dict):
        return out
    text = out.get("text")
    if not isinstance(text, str) or not text:
        return out
    meta = out.get("meta_info")
    finished = isinstance(meta, dict) and meta.get("finish_reason") is not None
    if incremental and states is not None:
        state = states.setdefault(out.get("index", 0), StreamUnescaper())
        new = state.feed(text)
        if finished:
            new += state.flush()
    elif finished:
        new = unescape(text)
    else:
        n = _held_suffix_len(text)
        new = unescape(text[:-n] if n else text)
    if new == text:
        return out
    out = dict(out)
    out["text"] = new
    return out
'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    module = f"{path}/python/sglang/srt/entrypoints/openai/control_tag_unescape.py"
    with open(module, "w") as f:
        f.write(MODULE_SRC)
    print("wrote", module)

    p = f"{path}/python/sglang/srt/entrypoints/openai/serving_chat.py"
    text = open(p).read()

    old = "from sglang.srt.parser.reasoning_parser import ReasoningParser\n"
    assert text.count(old) == 1, "serving_chat.py: ReasoningParser import anchor not found"
    new = (
        "from sglang.srt.entrypoints.openai import control_tag_unescape\n"
        + old
    )
    text = text.replace(old, new, 1)

    old = """        generator = self._generate_chat_stream(adapted_request, request, raw_request)
"""
    assert text.count(old) == 1, "serving_chat.py: _generate_chat_stream call anchor not found"
    new = """        generator = self._generate_chat_stream(adapted_request, request, raw_request)
        # gfx1151 patch 32: decode the template's control-string escaping after
        # the reasoning and tool-call parsers; the wrapper passes the first
        # chunk's validation errors through unchanged.
        if control_tag_unescape.enabled():
            generator = control_tag_unescape.unescape_sse_stream(generator)
"""
    text = text.replace(old, new, 1)

    old = """        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response
"""
    assert text.count(old) == 1, "serving_chat.py: _build_chat_response call anchor not found"
    new = """        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )
        # gfx1151 patch 32: see _handle_streaming_request.
        if control_tag_unescape.enabled():
            response = control_tag_unescape.unescape_chat_response(response)

        return response
"""
    text = text.replace(old, new, 1)
    open(p, "w").write(text)
    print("patched", p)

    p = f"{path}/python/sglang/srt/entrypoints/openai/serving_completions.py"
    text = open(p).read()

    old = "from sglang.srt.entrypoints.openai.protocol import (\n"
    assert text.count(old) == 1, "serving_completions.py: protocol import anchor not found"
    text = text.replace(
        old, "from sglang.srt.entrypoints.openai import control_tag_unescape\n" + old, 1
    )

    old = """        generator = self._generate_completion_stream(
            adapted_request, request, raw_request
        )
"""
    assert text.count(old) == 1, "serving_completions.py: _generate_completion_stream call anchor not found"
    new = """        generator = self._generate_completion_stream(
            adapted_request, request, raw_request
        )
        # gfx1151 patch 32: decode the detokenizer's control-string escaping
        # (patch 33) in each choice's text; see serving_chat.py.
        if control_tag_unescape.enabled():
            generator = control_tag_unescape.unescape_completion_sse_stream(generator)
"""
    text = text.replace(old, new, 1)

    old = """        response = self._build_completion_response(
            request,
            ret,
            int(time.time()),
        )

        return response
"""
    assert text.count(old) == 1, "serving_completions.py: _build_completion_response call anchor not found"
    new = """        response = self._build_completion_response(
            request,
            ret,
            int(time.time()),
        )
        # gfx1151 patch 32: see _handle_streaming_request.
        if control_tag_unescape.enabled():
            response = control_tag_unescape.unescape_completion_response(response)

        return response
"""
    text = text.replace(old, new, 1)
    open(p, "w").write(text)
    print("patched", p)

    p = f"{path}/python/sglang/srt/entrypoints/http_server.py"
    text = open(p).read()

    old = "from sglang.srt.entrypoints.openai.protocol import (\n"
    assert text.count(old) == 1, "http_server.py: protocol import anchor not found"
    text = text.replace(
        old, "from sglang.srt.entrypoints.openai import control_tag_unescape\n" + old, 1
    )

    old = """        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\\n\\n"
"""
    assert text.count(old) == 1, "http_server.py: /generate stream_results anchor not found"
    new = """        async def stream_results() -> AsyncIterator[bytes]:
            # gfx1151 patch 32: decode the detokenizer's control-string
            # escaping (patch 33) in the streamed text.
            unescape = control_tag_unescape.enabled()
            unescape_states = {}
            unescape_incremental = get_serving().incremental_streaming_output
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    if unescape:
                        out = control_tag_unescape.unescape_generate_output(
                            out, unescape_states, unescape_incremental
                        )
                    yield b"data: " + dumps_json(out) + b"\\n\\n"
"""
    text = text.replace(old, new, 1)

    old = """            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
"""
    assert text.count(old) == 1, "http_server.py: /generate non-stream anchor not found"
    new = """            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            # gfx1151 patch 32: see stream_results.
            if control_tag_unescape.enabled():
                ret = control_tag_unescape.unescape_generate_output(ret, None, False)
            return orjson_response(ret)
"""
    text = text.replace(old, new, 1)
    open(p, "w").write(text)
    print("patched", p)


if __name__ == "__main__":
    main()
