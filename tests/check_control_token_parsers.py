#!/usr/bin/env python3
"""Check patch 33 with the real parsers and the real tokenizer, without a GPU
server: the reasoning parser (qwen3) and the tool-call parser (qwen3_coder)
switch state only on control token ids once a ControlTokenStream is attached.

Model output is built from token ids: content pieces encoded as ordinary
tokens (so a spelled-out "</think>" is the pieces "</", "think", ">") and the
real control token ids in between. The ids run through a mirror of SGLang's
incremental detokenizer under several chunkings, and the resulting text and
ids are handed to the parsers the way serving_chat.py does.

  1. Streaming and one-shot parsing agree with each other and with the
     intended split for every chunking; spelled-out markers stay in the
     text they appear in; real markers switch state; tool calls parse with
     markers quoted inside parameter values.
  2. Without the stream (upstream behaviour) the spelled-out "</think>" ends
     the reasoning early: the bug this patch fixes.
  3. A stream whose text disagrees with its ids falls back to text matching.
  4. skip_special_tokens=False with the trimmed EOS still aligns.

Needs the patched sglang tree (the image) and the tokenizer:
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \\
    --group-add render --security-opt seccomp=unconfined -e MODEL_DIR=/models/qwen38 \\
    -v $PWD:/repo:ro -v <model>:/models/qwen38:ro \\
    --entrypoint python3 strix-halo-sglang:dev /repo/tests/check_control_token_parsers.py
"""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import List

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen38")

failures = 0
checks = 0


def check(cond, msg):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL: {msg}")


def main():
    from transformers import AutoTokenizer

    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser import control_token_stream as cts
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    backend = tok.backend_tokenizer
    markers = cts.parser_markers("qwen3", "qwen3_coder")
    check(markers == ["</think>", "</tool_call>", "<think>", "<tool_call>"], f"markers {markers}")
    control_ids = cts.resolve_control_ids(tok, markers)
    check(sorted(control_ids.values()) == markers, f"control ids {control_ids}")
    THINK, END_THINK = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    TC, END_TC = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</tool_call>")
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")

    # Ordinary tokens only: a tokenizer copy without added tokens.
    import json

    from tokenizers import Tokenizer

    spec = json.loads(backend.to_str())
    spec["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(spec))

    def P(text: str) -> List[int]:
        ids = plain.encode(text, add_special_tokens=False).ids
        assert all(i not in control_ids for i in ids)
        return ids

    tools = [
        Tool(
            type="function",
            function=Function(
                name="read_file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}},
            ),
        )
    ]

    # --- a mirror of DetokenizerManager's incremental loop ------------------
    from sglang.utils import find_printable_text

    def detokenize(ids: List[int], chunks: List[int], skip: bool, trim_eos: bool):
        """Yield (cumulative_ids, cumulative_text) per step, as the tokenizer
        manager hands them to serving (non-incremental streaming)."""
        decode_ids: List[int] = []
        surr = read = 0
        decoded = ""
        sent = 0
        pos = 0
        steps = []
        for n in chunks:
            new = ids[pos : pos + n]
            pos += n
            if not new:
                continue
            decode_ids.extend(new)
            finished = pos >= len(ids)
            read_ids = decode_ids[surr:]
            if finished and trim_eos and read_ids and read_ids[-1] == IM_END:
                read_ids = read_ids[:-1]
            surr_text = tok.decode(decode_ids[surr:read], skip_special_tokens=skip)
            read_text = tok.decode(read_ids, skip_special_tokens=skip)
            new_text = read_text[len(surr_text) :]
            if not finished:
                pending = sent - len(decoded)
                if new_text and not new_text.endswith("\ufffd"):
                    decoded += new_text
                    surr, read = read, len(decode_ids)
                    sent = len(decoded)
                    out = new_text[pending:] if pending else new_text
                else:
                    printable = find_printable_text(new_text)
                    sent = len(decoded) + len(printable)
                    out = printable[pending:] if pending else printable
                steps.append((list(decode_ids), (decoded if sent == len(decoded) else decoded + printable)))
                if sent != len(decoded):
                    # text already shown but not committed: steps carry the
                    # shown prefix, as state.get_text() would
                    steps[-1] = (steps[-1][0], decoded + printable)
            else:
                if new_text.endswith("\ufffd"):
                    new_text = find_printable_text(new_text)
                decoded += new_text
                steps.append((list(decode_ids), decoded))
        return steps

    def run_stream(ids, chunks, *, skip, trim_eos, attach, force_reasoning=True, misalign=False):
        stream = cts.ControlTokenStream(backend, control_ids, skip_special_tokens=skip) if attach else None
        rp = ReasoningParser("qwen3", stream_reasoning=True, force_reasoning=force_reasoning)
        rp.attach_control_stream(stream)
        fp = FunctionCallParser(tools, "qwen3_coder", tokenizer=tok)
        fp.attach_control_stream(stream)
        reasoning, normal, calls = [], [], []
        shown = 0
        steps = detokenize(ids, chunks, skip, trim_eos)
        for k, (cum_ids, cum_text) in enumerate(steps):
            delta = cum_text[shown:]
            shown = len(cum_text)
            if misalign and delta and k == 0:
                # Text that is not what the ids decode to.
                delta = "X" + delta[1:]
            if stream is not None:
                stream.feed(cum_ids, cumulative=True)
                stream.consume(delta)
            r, n = rp.parse_stream_chunk(delta)
            last = k == len(steps) - 1
            if last:
                er, en = rp.parse_stream_end()
                r = (r or "") + (er or "")
                n = (n or "") + (en or "")
            if stream is not None:
                stream.set_tail_lag(rp.held_back_len())
            if r:
                reasoning.append(r)
            n2, c = fp.parse_stream_chunk(n or "")
            if last:
                en2, ec = fp.parse_stream_end()
                n2 = (n2 or "") + en2
                c = list(c) + ec
            if n2:
                normal.append(n2)
            calls.extend(c)
        return "".join(reasoning), "".join(normal), calls, (stream.active if stream else None)

    def run_oneshot(ids, text, *, skip, attach, force_reasoning=True):
        stream = cts.ControlTokenStream(backend, control_ids, skip_special_tokens=skip) if attach else None
        if stream is not None:
            stream.feed(ids, cumulative=True)
            stream.consume(text)
        rp = ReasoningParser("qwen3", stream_reasoning=False, force_reasoning=force_reasoning)
        rp.attach_control_stream(stream)
        r, n = rp.parse_non_stream(text)
        fp = FunctionCallParser(tools, "qwen3_coder", tokenizer=tok)
        fp.attach_control_stream(stream)
        calls = []
        if fp.has_tool_call(n):
            n, calls = fp.parse_non_stream(n)
        return r, n, calls

    def chunkings(n, seed=0):
        rnd = random.Random(seed)
        yield [1] * n
        yield [n]
        for size in (2, 3, 7):
            yield [size] * (n // size + 1)
        for _ in range(6):
            out = []
            left = n
            while left > 0:
                k = rnd.randint(1, 9)
                out.append(k)
                left -= k
            yield out

    def joined_calls(calls):
        out = {}
        for c in calls:
            if c.name:
                out[c.tool_index] = [c.name, ""]
            if c.parameters:
                out.setdefault(c.tool_index, [None, ""])[1] += c.parameters
        return [(v[0], json.loads(v[1]) if v[1] else {}) for k, v in sorted(out.items())]

    # --- scenario 1: quotes inside reasoning and content, then a tool call ---
    reasoning_text = (
        "The user grepped for </think> and <tool_call> in notes.md 🚀, so I'll quote "
        "them: `</think>` `<tool_call>` `</tool_call>` `<think>` and move on é.\n"
    )
    answer_text = "Found 2 matches for </think> in notes.md; also one <tool_call>.\n\n"
    call_text = "\n<function=read_file>\n<parameter=path>\nnotes/</think>.md\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n"
    ids = P(reasoning_text) + [END_THINK] + P(answer_text) + [TC] + P(call_text) + [END_TC] + [IM_END]
    full_text_skip = tok.decode(ids[:-1], skip_special_tokens=True)
    full_text_keep = tok.decode(ids[:-1], skip_special_tokens=False)

    for skip, trim_eos, full_text in ((True, True, full_text_skip), (False, True, full_text_keep)):
        r, n, calls = run_oneshot(ids, full_text, skip=skip, attach=True)
        check(r == reasoning_text, f"one-shot reasoning (skip={skip}): {r!r}")
        check(n == answer_text, f"one-shot normal (skip={skip}): {n!r}")
        check(
            joined_calls(calls) == [("read_file", {"path": "notes/</think>.md", "n": 3})],
            f"one-shot calls (skip={skip}): {calls}",
        )
        for ci, chunks in enumerate(chunkings(len(ids))):
            r, n, calls, active = run_stream(ids, chunks, skip=skip, trim_eos=trim_eos, attach=True)
            check(active, f"stream stayed aligned (skip={skip}, chunking {ci})")
            check(r == reasoning_text, f"stream reasoning (skip={skip}, chunking {ci}): {r!r}")
            check(n == answer_text, f"stream normal (skip={skip}, chunking {ci}): {n!r}")
            check(
                joined_calls(calls) == [("read_file", {"path": "notes/</think>.md", "n": 3})],
                f"stream calls (skip={skip}, chunking {ci}): {calls}",
            )

    # --- scenario 2: upstream behaviour without the stream shows the bug ---
    r, n, calls = run_oneshot(ids, full_text_skip, skip=True, attach=False)
    check(r != reasoning_text and r.startswith("The user grepped for "), f"without ids the quoted </think> ends reasoning: {r!r}")
    r, n, calls, active = run_stream(ids, [1] * len(ids), skip=True, trim_eos=True, attach=False)
    check(r != reasoning_text, "without ids the stream also breaks out early")

    # --- scenario 3: misaligned text falls back to text matching, once -----
    logging.getLogger("sglang.srt.parser.control_token_stream").setLevel(logging.ERROR)
    r, n, calls, active = run_stream(ids, [3] * (len(ids) // 3 + 1), skip=True, trim_eos=True, attach=True, misalign=True)
    check(active is False, "misaligned stream deactivates")
    r0, n0, calls0, _ = run_stream(ids, [3] * (len(ids) // 3 + 1), skip=True, trim_eos=True, attach=False, misalign=True)
    check((r, n) == (r0, n0), "misaligned stream parses like no stream")

    # --- scenario 4: real <think> token at the start, force_reasoning off ---
    ids2 = [THINK] + P("\nquoting <think> and </think> here\n") + [END_THINK] + P("\n\nDone.") + [IM_END]
    text2 = tok.decode(ids2[:-1], skip_special_tokens=True)
    r, n, calls = run_oneshot(ids2, text2, skip=True, attach=True, force_reasoning=False)
    check(r == "\nquoting <think> and </think> here\n", f"one-shot with <think> token: {r!r}")
    check(n == "\n\nDone.", f"one-shot normal with <think> token: {n!r}")
    for ci, chunks in enumerate(chunkings(len(ids2), seed=1)):
        r, n, calls, active = run_stream(ids2, chunks, skip=True, trim_eos=True, attach=True, force_reasoning=False)
        check((r, n) == ("\nquoting <think> and </think> here\n", "\n\nDone."), f"stream with <think> token, chunking {ci}: {(r, n)!r}")

    # --- scenario 5: a spelled-out <tool_call> block in the answer is text ---
    ids3 = P("ok\n") + [END_THINK] + P("Here is the format:\n<tool_call>\n<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>\nThat is all.") + [IM_END]
    text3 = tok.decode(ids3[:-1], skip_special_tokens=True)
    r, n, calls = run_oneshot(ids3, text3, skip=True, attach=True)
    check(calls == [] and n == text3.split("</think>", 1)[1], f"spelled-out tool call block stays text (one-shot): {n!r} {calls}")
    for ci, chunks in enumerate(chunkings(len(ids3), seed=2)):
        r, n, calls, active = run_stream(ids3, chunks, skip=True, trim_eos=True, attach=True)
        check(active and r == "ok\n", f"spelled-out block, chunking {ci}: reasoning {r!r}")
        # The spelled-out <function=...> inside is still parsed as a call by the
        # upstream detector (no control token exists for it); only the
        # <tool_call> framing is text here.
        check("<tool_call>" in n and "</tool_call>" in n, f"spelled-out <tool_call> framing is content, chunking {ci}: {n!r}")

    # --- scenario 6: a real tool call whose parameter quotes the markers ---
    ids4 = P("plan\n") + [END_THINK] + P("\n") + [TC] + P("\n<function=read_file>\n<parameter=path>\na</tool_call>b<tool_call>c\n</parameter>\n</function>\n") + [END_TC] + [IM_END]
    text4 = tok.decode(ids4[:-1], skip_special_tokens=True)
    r, n, calls = run_oneshot(ids4, text4, skip=True, attach=True)
    check(joined_calls(calls) == [("read_file", {"path": "a</tool_call>b<tool_call>c"})], f"quoted markers inside a parameter (one-shot): {joined_calls(calls)}")
    for ci, chunks in enumerate(chunkings(len(ids4), seed=3)):
        r, n, calls, active = run_stream(ids4, chunks, skip=True, trim_eos=True, attach=True)
        check(joined_calls(calls) == [("read_file", {"path": "a</tool_call>b<tool_call>c"})], f"quoted markers inside a parameter, chunking {ci}: {joined_calls(calls)}")

    print(f"{checks - failures}/{checks} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
