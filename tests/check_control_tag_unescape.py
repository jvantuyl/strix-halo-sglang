#!/usr/bin/env python3
"""Exercise patch 32's output un-escape without an image or a GPU.

Loads the module source embedded in patches/patch_unescape_control_tags.py and
checks: one-level decode of both entity levels and nothing else; streaming
decode identical to whole-text decode for every split point of every pattern
and for random chunking; SSE events rewritten only in content,
reasoning_content and tool-call arguments, with held text flushed into the
choice's finish chunk; the tag list matches the chat template's.

Run locally with: python3 tests/check_control_tag_unescape.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "patch_unescape_control_tags.py"
TEMPLATE = ROOT / "configs" / "chat" / "qwen38.jinja"

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'✓' if ok else '✗'} {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def load_module():
    spec = importlib.util.spec_from_file_location("patch32", PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    try:
        import orjson  # noqa: F401
    except ImportError:  # the image has it; stand in with json here
        shim = types.ModuleType("orjson")
        shim.loads = json.loads
        shim.dumps = lambda o: json.dumps(o, separators=(",", ":")).encode()
        sys.modules["orjson"] = shim
    mod = types.ModuleType("control_tag_unescape")
    exec(compile(patch.MODULE_SRC, "control_tag_unescape.py", "exec"), mod.__dict__)
    return mod


def stream_decode(mod, chunks):
    s = mod.StreamUnescaper()
    out = "".join(s.feed(c) for c in chunks)
    return out + s.flush()


def main() -> int:
    mod = load_module()
    tags = mod.CONTROL_TAGS
    ent = mod._entity
    dbl = mod._double

    print("tag list:")
    jinja = TEMPLATE.read_text()
    m = re.search(r"\{%- for s in \[([^\]]*)\] %\}", jinja)
    template_tags = tuple(re.findall(r"'([^']+)'", m.group(1))) if m else ()
    check("patch CONTROL_TAGS == template escape list", template_tags == tags,
          f"template={template_tags} patch={tags}")

    print("whole-text decode:")
    for t in tags:
        check(f"{ent(t)} -> {t}", mod.unescape(f"a {ent(t)} b") == f"a {t} b")
        check(f"{dbl(t)} -> {ent(t)}", mod.unescape(f"a {dbl(t)} b") == f"a {ent(t)} b")
    for other in ["&lt;div&gt;", "&amp;", "&lt;script&gt;alert(1)&lt;/script&gt;", "&nbsp;",
                  "&lt;thinking&gt;", "&lt;tool_calls&gt;", "&amp;lt;div&amp;gt;", "<think>", "</think>"]:
        check(f"untouched: {other}", mod.unescape(other) == other)
    check("empty/None pass through", mod.unescape("") == "" and mod.unescape(None) is None)

    print("streaming decode == whole-text decode:")
    for t in tags:
        for form in (ent(t), dbl(t)):
            text = f"x {form} y"
            ok = all(
                stream_decode(mod, [text[:i], text[i:]]) == mod.unescape(text)
                for i in range(len(text) + 1)
            )
            check(f"every 2-way split of '{form}'", ok)
            ok3 = all(
                stream_decode(mod, [text[:i], text[i:j], text[j:]]) == mod.unescape(text)
                for i in range(len(text) + 1) for j in range(i, len(text) + 1)
            )
            check(f"every 3-way split of '{form}'", ok3)
    # Character-at-a-time, the worst case for the hold-back.
    corpus = " ".join([ent(t) for t in tags] + [dbl(t) for t in tags] + ["&lt;div&gt;", "& ampersand", "&amp;lt;", "&lt;/thin", "a&b&c"])
    check("char-at-a-time stream", stream_decode(mod, list(corpus)) == mod.unescape(corpus))
    rng = random.Random(32)
    ok = True
    for _ in range(300):
        pieces, i = [], 0
        while i < len(corpus):
            j = min(len(corpus), i + rng.randint(1, 12))
            pieces.append(corpus[i:j])
            i = j
        if stream_decode(mod, pieces) != mod.unescape(corpus):
            ok = False
            break
    check("300 random chunkings", ok)
    # A dangling prefix at the end is flushed as-is, never dropped.
    check("flush returns a dangling prefix", stream_decode(mod, ["see &lt;/thin"]) == "see &lt;/thin")
    # Hold-back never retains more than the longest pattern minus one.
    s = mod.StreamUnescaper()
    s.feed("&amp;lt;/tool_response&amp;gt")
    check("hold-back bounded", len(s.pending) <= mod._MAX_HOLD and s.pending == "&amp;lt;/tool_response&amp;gt")

    print("SSE events:")
    def ev(obj):
        return "data: " + json.dumps(obj) + "\n\n"
    def chunk(delta, finish=None, idx=0):
        return ev({"id": "c", "object": "chat.completion.chunk", "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]})
    def parse(e):
        return json.loads(e[len("data: "):])
    states = {}
    out = []
    events = [
        chunk({"role": "assistant", "reasoning_content": "quote &lt;|im_"}),
        chunk({"reasoning_content": "end|&gt; done"}),
        chunk({"content": "file has &lt;/think&gt; and &lt;div&gt;"}),
        chunk({"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "Edit", "arguments": "{\"old\": \"&lt;/thi"}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "nk&gt; &amp;lt;think&amp;gt; tail &lt;/th"}}]}),
        chunk({}, finish="tool_calls"),
        ev({"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 1}}),
        "event: sglext_ids\ndata: {\"x\": 1}\n\n",
        "data: [DONE]\n\n",
    ]
    for e in events:
        out.append(mod.unescape_sse_event(e, states))
    reasoning = "".join(parse(o)["choices"][0]["delta"].get("reasoning_content") or "" for o in out[:6])
    content = "".join(parse(o)["choices"][0]["delta"].get("content") or "" for o in out[:6])
    args = "".join(
        tc["function"]["arguments"]
        for o in out[:6] for tc in (parse(o)["choices"][0]["delta"].get("tool_calls") or [])
    )
    check("reasoning_content decoded across a split entity", reasoning == "quote <|im_end|> done")
    check("content decoded, &lt;div&gt; untouched", content == "file has </think> and &lt;div&gt;")
    check("tool-call arguments decoded, two levels, tail flushed at finish",
          args == "{\"old\": \"</think> &lt;think&gt; tail &lt;/th")
    first_tc = parse(out[3])["choices"][0]["delta"]["tool_calls"][0]
    check("tool-call id/name/type preserved", first_tc.get("id") == "t1" and first_tc["function"].get("name") == "Edit" and first_tc.get("type") == "function")
    finish = parse(out[5])["choices"][0]
    check("finish chunk keeps finish_reason", finish["finish_reason"] == "tool_calls")
    check("usage chunk passes through unchanged", out[6] == events[6])
    check("non-data event passes through unchanged", out[7] == events[7])
    check("[DONE] passes through unchanged", out[8] == events[8])
    check("nothing pending after finish", all(not s.pending for s in states.values()))

    # Two choices keep separate state.
    states = {}
    a = mod.unescape_sse_event(chunk({"content": "&lt;/thi"}, idx=0), states)
    b = mod.unescape_sse_event(chunk({"content": "nk&gt;"}, idx=1), states)
    c = mod.unescape_sse_event(chunk({"content": "nk&gt;"}, idx=0), states)
    check("choices decoded independently",
          parse(a)["choices"][0]["delta"]["content"] == "" and
          parse(b)["choices"][0]["delta"]["content"] == "nk&gt;" and
          parse(c)["choices"][0]["delta"]["content"] == "</think>")

    # The async wrapper yields one event per input event, in order.
    async def gen():
        for e in events:
            yield e
    async def collect():
        return [e async for e in mod.unescape_sse_stream(gen())]
    streamed = asyncio.run(collect())
    check("async wrapper preserves event count and order", len(streamed) == len(events) and streamed[-1] == events[-1])

    print("non-streaming response:")
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    resp = Obj(choices=[Obj(message=Obj(
        content="see &lt;|im_end|&gt;", reasoning_content="&amp;lt;think&amp;gt;",
        tool_calls=[Obj(function=Obj(name="Create", arguments="{\"c\": \"&lt;tool_call&gt;\"}"))]))])
    mod.unescape_chat_response(resp)
    msg = resp.choices[0].message
    check("content/reasoning/arguments decoded", msg.content == "see <|im_end|>" and msg.reasoning_content == "&lt;think&gt;"
          and msg.tool_calls[0].function.arguments == "{\"c\": \"<tool_call>\"}")
    check("content=None tolerated", mod.unescape_chat_response(Obj(choices=[Obj(message=Obj(content=None, reasoning_content=None, tool_calls=None))])) is not None)

    print("/v1/completions:")
    def cchunk(text, finish=None, idx=0):
        return ev({"id": "c", "object": "text_completion", "choices": [{"index": idx, "text": text, "finish_reason": finish}]})
    states = {}
    cout = [mod.unescape_completion_sse_event(e, states) for e in [
        cchunk("see &lt;/thi"), cchunk("nk&gt; and &lt;div&gt; &amp;lt;tool_call&amp;gt; &lt;|im"),
        cchunk("", finish="stop"), ev({"choices": [], "usage": {"total_tokens": 1}}), "data: [DONE]\n\n"]]
    ctext = "".join(parse(o)["choices"][0]["text"] for o in cout[:3])
    check("completion stream decoded, tail flushed into the finish chunk",
          ctext == "see </think> and &lt;div&gt; &lt;tool_call&gt; &lt;|im", repr(ctext))
    check("completion finish chunk keeps finish_reason", parse(cout[2])["choices"][0]["finish_reason"] == "stop")
    check("completion usage / [DONE] pass through", cout[3] == ev({"choices": [], "usage": {"total_tokens": 1}}) and cout[4] == "data: [DONE]\n\n")
    states = {}
    a = mod.unescape_completion_sse_event(cchunk("&lt;/thi", idx=0), states)
    b = mod.unescape_completion_sse_event(cchunk("nk&gt;", idx=1), states)
    c = mod.unescape_completion_sse_event(cchunk("nk&gt;", idx=0), states)
    check("completion choices decoded independently",
          parse(a)["choices"][0]["text"] == "" and parse(b)["choices"][0]["text"] == "nk&gt;"
          and parse(c)["choices"][0]["text"] == "</think>")
    resp = Obj(choices=[Obj(text="a &lt;|im_end|&gt; b"), Obj(text=None)])
    mod.unescape_completion_response(resp)
    check("completion response text decoded, None tolerated",
          resp.choices[0].text == "a <|im_end|> b" and resp.choices[1].text is None)

    print("/generate:")
    def gout(text, finish=None, index=None):
        o = {"text": text, "output_ids": [1], "meta_info": {"id": "r", "finish_reason": finish}}
        if index is not None:
            o["index"] = index
        return o
    # Cumulative (default): each chunk's text is a prefix of the next's.
    cum = ["see &lt;/thi", "see &lt;/think&gt; and &lt;|im", "see &lt;/think&gt; and &lt;|im_end|&gt; &lt;/th"]
    got = [mod.unescape_generate_output(gout(t), None, False)["text"] for t in cum]
    fin = mod.unescape_generate_output(gout(cum[-1], finish={"type": "stop"}), None, False)["text"]
    check("cumulative stream: dangling prefix withheld, decoded once complete",
          got == ["see ", "see </think> and ", "see </think> and <|im_end|> "], repr(got))
    check("cumulative stream: prefixes stay prefixes", all(fin.startswith(g) for g in got) and all(got[i + 1].startswith(got[i]) for i in range(len(got) - 1)))
    check("cumulative finish: decoded whole, dangling prefix kept", fin == "see </think> and <|im_end|> &lt;/th", repr(fin))
    # Incremental (--incremental-streaming-output): stateful per index.
    states = {}
    inc = [mod.unescape_generate_output(gout(t, index=0), states, True)["text"] for t in ["see &lt;/thi", "nk&gt; and &lt;|im"]]
    inc.append(mod.unescape_generate_output(gout("_end|&gt; &lt;/th", finish={"type": "stop"}, index=0), states, True)["text"])
    check("incremental stream: decoded across chunks, tail flushed at finish",
          "".join(inc) == "see </think> and <|im_end|> &lt;/th", repr(inc))
    plain = gout("nothing here", finish={"type": "stop"})
    check("unchanged output returned as the same object", mod.unescape_generate_output(plain, None, False) is plain)
    orig = gout("x &lt;think&gt;", finish={"type": "stop"})
    dec = mod.unescape_generate_output(orig, None, False)
    check("decoded output is a copy; original untouched", dec["text"] == "x <think>" and orig["text"] == "x &lt;think&gt;" and dec["output_ids"] is orig["output_ids"])
    lst = mod.unescape_generate_output([gout("&lt;/think&gt;", finish={"type": "stop"}), gout(None), "junk"], None, False)
    check("batched list handled per item; text=None and non-dicts pass through",
          lst[0]["text"] == "</think>" and lst[1]["text"] is None and lst[2] == "junk")

    print("switch:")
    import os
    os.environ.pop("SGLANG_UNESCAPE_CONTROL_TAGS", None)
    off = mod.enabled()
    os.environ["SGLANG_UNESCAPE_CONTROL_TAGS"] = "1"
    on = mod.enabled()
    check("off unless SGLANG_UNESCAPE_CONTROL_TAGS=1", not off and on)

    print(f"\n{checks - len(failures)}/{checks} checks passed.")
    if failures:
        print("FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
