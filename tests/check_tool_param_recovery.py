#!/usr/bin/env python3
"""Check patch 36 with the real qwen3_coder tool-call parser, without a GPU:
array and object parameters survive a missing final closer, a schema that
hides a property behind a union branch or a local $ref is still read, and
nothing a repair would have to guess at is touched.

  1. A value one or two closers short parses to the schema's type, in the
     one-shot and the streaming path alike, under several chunkings; the
     parsed call equals the one for the well-formed value.
  2. A value broken in the middle (a closer dropped mid-array, a missing
     comma), one cut off inside a string, one that closes what it never
     opened, and one whose closed form has the wrong type stay strings.
  3. A string parameter is never converted, whatever it looks like.
  4. Properties behind oneOf/anyOf beside top-level properties and behind
     "#/$defs" or "#/definitions" refs are typed; a parameter the schema does
     not list comes through as the array or object it spells, or as the string
     it is.

Needs the patched sglang tree (the image); the device flags only let the
sglang import initialise Triton, the check itself runs on the CPU:
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \\
    --group-add render --security-opt seccomp=unconfined -v $PWD:/repo:ro \\
    --entrypoint python3 strix-halo-sglang:dev /repo/tests/check_tool_param_recovery.py
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.ERROR)

failures = 0
checks = 0


def check(cond, msg):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL: {msg}")


def main():
    import sglang
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
    from sglang.srt.function_call.utils import (
        close_unterminated_json,
        get_schema_properties,
    )

    print(f"sglang tree: {sglang.__file__}")

    def tool(name, parameters):
        return Tool(type="function", function=Function(name=name, parameters=parameters))

    def call_text(func, params):
        body = "".join(
            f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()
        )
        return f"<tool_call>\n<function={func}>\n{body}</function>\n</tool_call>"

    def one_shot(text, tools):
        det = Qwen3CoderDetector()
        res = det.detect_and_parse(text, tools)
        return [(c.name, json.loads(c.parameters)) for c in res.calls]

    def chunkings(text):
        yield [text]
        yield list(text)
        yield [text[i : i + 7] for i in range(0, len(text), 7)]
        half = len(text) // 2
        yield [text[:half], text[half:]]

    def streamed(chunks, tools):
        det = Qwen3CoderDetector()
        by_index = {}
        for chunk in chunks:
            res = det.parse_streaming_increment(chunk, tools)
            for c in res.calls:
                name, params = by_index.get(c.tool_index, (None, ""))
                by_index[c.tool_index] = (c.name or name, params + c.parameters)
        return [(n, json.loads(p)) for n, p in (by_index[i] for i in sorted(by_index))]

    def parsed_all_ways(text, tools):
        """One-shot result, after checking every chunking agrees with it."""
        want = one_shot(text, tools)
        for chunks in chunkings(text):
            got = streamed(chunks, tools)
            check(got == want, f"streaming ({len(chunks)} chunks) {got} != one-shot {want}")
        return want

    # ---- 1. closers missing at the end ------------------------------------
    handoff_schema = {
        "type": "object",
        "properties": {
            "featureId": {"type": "string"},
            "handoff": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "object"}},
                    "feedback": {"type": "object"},
                },
            },
            "calls": {"type": "array", "items": {"type": "object"}},
        },
    }
    tools = [tool("EndFeatureRun", handoff_schema)]
    handoff = {
        "summary": 'quotes "inside", brackets ]} and }] in text, escape \\" too',
        "issues": [{"severity": "non_blocking", "description": "a [b] {c}"}],
        "feedback": {"followedProcedure": True, "deviations": []},
    }
    full = json.dumps(handoff)
    calls = [{"name": "x", "arguments": {"q": {"a": 1}}}, {"name": "y", "arguments": {}}]
    full_calls = json.dumps(calls)

    want = parsed_all_ways(
        call_text("EndFeatureRun", {"featureId": "f1", "handoff": full, "calls": full_calls}),
        tools,
    )
    check(
        want == [("EndFeatureRun", {"featureId": "f1", "handoff": handoff, "calls": calls})],
        f"well-formed baseline {want}",
    )
    for short in (1, 2):
        got = parsed_all_ways(
            call_text(
                "EndFeatureRun",
                {"featureId": "f1", "handoff": full[:-short], "calls": full_calls[:-short]},
            ),
            tools,
        )
        check(got == want, f"{short} closer(s) short: {got}")
    # trailing whitespace after the short value, as the model often leaves it
    got = parsed_all_ways(
        call_text("EndFeatureRun", {"handoff": full[:-1] + " \n"}), tools
    )
    check(got == [("EndFeatureRun", {"handoff": handoff})], f"short with trailing space: {got}")

    # ---- 2. what stays a string ---------------------------------------------
    mid_broken = full.replace('{c}"}]', '{c}"]', 1)  # } dropped before ] mid-array
    check(mid_broken != full, "mid-structure fixture applies")
    no_comma = full.replace('"issues"', '"issues"', 1).replace('", "issues"', '" "issues"', 1)
    in_string = full[: full.index("brackets") + 4]
    over_closed = full + "}"
    wrong_type = '{"a": 1'  # closes to a dict, schema says array
    for label, value, key in (
        ("closer dropped mid-structure", mid_broken, "handoff"),
        ("missing comma", no_comma, "handoff"),
        ("cut off inside a string", in_string, "handoff"),
        ("closes what it never opened", over_closed, "handoff"),
        ("closed form has the wrong type", wrong_type, "calls"),
    ):
        got = parsed_all_ways(call_text("EndFeatureRun", {key: value}), tools)
        check(
            got == [("EndFeatureRun", {key: value})],
            f"{label}: expected the raw string back, got {got!r:.200}",
        )
    for value in (full, "", "   ", "plain text", '"a string"', "[]", "{}"):
        check(close_unterminated_json(value) is None, f"close_unterminated_json({value!r}) is not None")
    check(close_unterminated_json('[{"a": [1, 2') == ('[{"a": [1, 2]}]', 3), "closes three levels")
    check(close_unterminated_json('["a]\\"" ') == ('["a]\\""]', 1), "escapes inside strings")

    # ---- 3. string parameters are never converted ---------------------------
    tools_str = [tool("Edit", {"type": "object", "properties": {"old_str": {"type": "string"}}})]
    for value in ('{"a": 1}', '{"a": 1', "[1, 2", "[1, 2]"):
        got = parsed_all_ways(call_text("Edit", {"old_str": value}), tools_str)
        check(got == [("Edit", {"old_str": value})], f"string param {value!r} -> {got}")

    # ---- 4. schemas the parser used to miss ---------------------------------
    union_schema = {
        "type": "object",
        "properties": {"questionnaire": {"type": "string"}},
        "oneOf": [
            {"properties": {"questions": {"type": "array", "items": {"type": "object"}}}},
            {"properties": {"questionnaire": {"type": "integer"}}},  # top level wins
        ],
    }
    props = get_schema_properties(union_schema)
    check(set(props) == {"questionnaire", "questions"}, f"merged properties {sorted(props)}")
    check(props["questionnaire"] == {"type": "string"}, "top-level property wins over a branch")
    questions = [{"question": "Which?", "options": [{"label": "a"}, {"label": "b"}]}]
    got = parsed_all_ways(
        call_text("AskUser", {"questions": json.dumps(questions)}), [tool("AskUser", union_schema)]
    )
    check(got == [("AskUser", {"questions": questions})], f"property behind oneOf: {got}")

    ref_schema = {
        "$defs": {"Handoff": {"type": "object", "properties": {"s": {"type": "string"}}}},
        "definitions": {"Q": {"type": "array"}},
        "type": "object",
        "properties": {
            "handoff": {"$ref": "#/$defs/Handoff", "description": "kept"},
            "questions": {"$ref": "#/definitions/Q"},
            "dangling": {"$ref": "#/$defs/Missing"},
            "remote": {"$ref": "https://example.invalid/schema#/x"},
        },
    }
    props = get_schema_properties(ref_schema)
    check(props["handoff"].get("type") == "object" and props["handoff"].get("description") == "kept",
          f"$defs ref resolved with siblings kept: {props['handoff']}")
    check(props["questions"] == {"type": "array"}, f"definitions ref: {props['questions']}")
    check(props["dangling"] == {"$ref": "#/$defs/Missing"}, "dangling ref left alone")
    check(props["remote"] == {"$ref": "https://example.invalid/schema#/x"}, "remote ref left alone")
    got = parsed_all_ways(
        call_text("T", {"handoff": '{"s": "v"}', "questions": '[1, 2]', "dangling": '{"k": 1}'}),
        [tool("T", ref_schema)],
    )
    check(
        got == [("T", {"handoff": {"s": "v"}, "questions": [1, 2], "dangling": '{"k": 1}'})],
        f"$ref-typed parameters: {got}",
    )
    whole_ref = {"$defs": {"P": {"type": "object", "properties": {"n": {"type": "integer"}}}}, "$ref": "#/$defs/P"}
    got = parsed_all_ways(call_text("T", {"n": "3"}), [tool("T", whole_ref)])
    check(got == [("T", {"n": 3})], f"whole-schema $ref: {got}")
    cyclic = {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"}
    check(get_schema_properties(cyclic) == {}, "cyclic refs terminate")

    # a parameter the schema does not list
    tools_u = [tool("AskUser", {"type": "object", "properties": {"other": {"type": "string"}}})]
    for value, want_v in (
        (json.dumps(questions), questions),
        ('{"a": [1]}', {"a": [1]}),
        ("plain text", "plain text"),
        ("[not json", "[not json"),
        ('["a"', '["a"'),  # no repair for undeclared parameters
        ("42", "42"),
        ('"quoted"', '"quoted"'),
    ):
        got = parsed_all_ways(call_text("AskUser", {"questions": value}), tools_u)
        check(got == [("AskUser", {"questions": want_v})], f"undeclared {value!r:.60} -> {got!r:.120}")
    # ...and with no schema at all
    got = parsed_all_ways(call_text("Nope", {"questions": json.dumps(questions)}), tools_u)
    check(got == [("Nope", {"questions": questions})], f"unknown tool: {got!r:.120}")

    print(f"{checks} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
