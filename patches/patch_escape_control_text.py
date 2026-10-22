#!/usr/bin/env python3
"""Entity-escape spelled-out control strings in detokenizer output (patch 33).

Patch 32 and the repo chat template keep the real control tokens out of
message content, so a quoted <|im_end|> no longer breaks the model out of
<think> at the sampler. What remains is text: the reasoning parser and the
tool-call parser work on detokenized strings, so a model that writes
"</think>" as the ordinary tokens "</", "think", ">" (quoting a file, or a
prompt the template escaped) still ends the reasoning block, and a
spelled-out "<tool_call>" still opens a tool call, exactly as the real
tokens would. Seen live after patch 32: the thinking survived, but the
answer was cut at the first spelled-out </think>.

The detokenizer knows which ids are the real tokens. For rows whose text
contains "<", decode the ids in runs between the control token ids, rewrite
<X> -> &lt;X&gt; inside each run for the nine control strings only, and
keep whatever the tokenizer gives for the real tokens themselves. The
parsers then only ever match genuine tokens, and patch 32 turns the
entities back into the raw strings for the client. A trailing proper
prefix of a control string ("</thi") is held back across streaming steps
the same way an incomplete UTF-8 sequence is, so the rewrite is stable
whatever the chunking. A stop string matched by the scheduler is escaped
before the trim so it is still found.

Off unless SGLANG_UNESCAPE_CONTROL_TAGS is set to 1 (the same switch as
patch 32; one is not useful without the other). Hooks in
detokenizer_manager.py: the escaper is created in init_running_status, the
decoded texts are rewritten before the incremental loop, the streaming
branch holds back a partial control string, and the finished branch
escapes the matched stop string. Anchors are asserted so an upstream
rewrite fails the build. See patches/33-escape-control-text.md.
"""
import sys

MODULE_SRC = '''\
"""Entity-escape spelled-out control strings in detokenizer output.

gfx1151 patch 33; pairs with patch 32 (one-level decode in serving_chat.py)
and the content escaping in the repo chat template. The tag list and the
switch are the same as patch 32's and must stay identical.
"""
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

_ENTITY = {tag: "&lt;" + tag[1:-1] + "&gt;" for tag in CONTROL_TAGS}
_MAX_HOLD = max(len(tag) for tag in CONTROL_TAGS) - 1

# (ids per row, skip_special_tokens per row, spaces_between_special_tokens per row) -> texts
DecodeFn = Callable[[List[List[int]], List[bool], List[bool]], List[str]]


def enabled():
    return os.environ.get("SGLANG_UNESCAPE_CONTROL_TAGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def escape(text):
    """<X> -> &lt;X&gt; for the control strings; everything else untouched."""
    if "<" not in text:
        return text
    for tag, entity in _ENTITY.items():
        text = text.replace(tag, entity)
    return text


def held_suffix_len(text):
    """Length of the longest suffix of text that is a proper prefix of a
    control string, 0 if there is none. A complete control string at the
    end is not held (escape rewrites it)."""
    start = text.rfind("<", max(0, len(text) - _MAX_HOLD))
    if start < 0:
        return 0
    tail = text[start:]
    n = len(tail)
    for tag in CONTROL_TAGS:
        if len(tag) > n and tag.startswith(tail):
            return n
    return 0


def control_token_ids(tokenizer):
    """ids of the control strings that are single tokens of this tokenizer."""
    out: Dict[int, str] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    for tag in CONTROL_TAGS:
        try:
            tid = tokenizer.convert_tokens_to_ids(tag)
            if not isinstance(tid, int) or tid < 0 or tid == unk:
                continue
            if tokenizer.convert_ids_to_tokens(tid) != tag:
                continue
        except Exception:
            continue
        out[tid] = tag
    return out


class ControlTagEscaper:
    """Run-wise decode with escaping, for one tokenizer."""

    def __init__(self, control_ids: Dict[int, str], decode_fn: DecodeFn):
        self.control_ids = control_ids
        self.decode_fn = decode_fn
        self._control_text: Dict[Tuple[int, bool, bool], str] = {}

    @classmethod
    def create(cls, tokenizer, decode_fn: DecodeFn) -> Optional["ControlTagEscaper"]:
        if not enabled() or tokenizer is None:
            return None
        ids = control_token_ids(tokenizer)
        if not ids:
            return None
        return cls(ids, decode_fn)

    def control_text(self, tid: int, skip: bool, space: bool) -> str:
        key = (tid, skip, space)
        text = self._control_text.get(key)
        if text is None:
            text = self.decode_fn([[tid]], [skip], [space])[0]
            self._control_text[key] = text
        return text

    def decode(
        self,
        ids_list: Sequence[Sequence[int]],
        skip_list: Sequence[bool],
        space_list: Sequence[bool],
    ) -> Tuple[List[str], List[int]]:
        """Decode each row in runs between control tokens, escape the runs.

        Returns the texts and, per row, the length of a trailing proper
        prefix of a control string (left raw; the caller holds it back).
        """
        plans: List[List[Tuple[bool, object]]] = []
        run_ids: List[List[int]] = []
        run_skip: List[bool] = []
        run_space: List[bool] = []
        for ids, skip, space in zip(ids_list, skip_list, space_list):
            plan: List[Tuple[bool, object]] = []
            run: List[int] = []
            for tid in ids:
                if tid in self.control_ids:
                    if run:
                        plan.append((False, len(run_ids)))
                        run_ids.append(run)
                        run_skip.append(skip)
                        run_space.append(space)
                        run = []
                    plan.append((True, self.control_text(tid, skip, space)))
                else:
                    run.append(tid)
            if run:
                plan.append((False, len(run_ids)))
                run_ids.append(run)
                run_skip.append(skip)
                run_space.append(space)
            plans.append(plan)
        run_texts = self.decode_fn(run_ids, run_skip, run_space) if run_ids else []
        texts: List[str] = []
        held: List[int] = []
        for plan in plans:
            parts: List[str] = []
            h = 0
            for is_control, item in plan:
                if is_control:
                    parts.append(item)
                    h = 0
                else:
                    raw = run_texts[item]
                    parts.append(escape(raw))
                    h = held_suffix_len(raw)
            texts.append("".join(parts))
            held.append(h)
        return texts, held

    def rewrite(
        self,
        surr_ids: Sequence[Sequence[int]],
        read_ids: Sequence[Sequence[int]],
        surr_texts: List[str],
        read_texts: List[str],
        skip_list: Sequence[bool],
        space_list: Sequence[bool],
    ) -> List[int]:
        """Re-decode, in place, the rows whose read text contains "<" (the
        others cannot hold a control string). Returns the per-row hold-back."""
        n = len(read_texts)
        held = [0] * n
        rows = [i for i in range(n) if "<" in read_texts[i]]
        if not rows:
            return held
        skips = [skip_list[i] for i in rows]
        spaces = [space_list[i] for i in rows]
        texts, hs = self.decode(
            [surr_ids[i] for i in rows] + [read_ids[i] for i in rows],
            skips + skips,
            spaces + spaces,
        )
        m = len(rows)
        for k, i in enumerate(rows):
            surr_texts[i] = texts[k]
            read_texts[i] = texts[m + k]
            held[i] = hs[m + k]
        return held

    @staticmethod
    def escape_finished_reason(finished_reason, text):
        """The scheduler matched the stop string on raw text; the trim runs
        on the escaped text. A stop string that is itself a control string
        may have matched the real token (literal in text) or a spelled-out
        copy (escaped in text): keep the raw match when text holds it,
        otherwise return a copy with the match escaped. The original object
        is never changed (it is forwarded to the client as is)."""
        if isinstance(finished_reason, dict):
            matched = finished_reason.get("matched")
            if isinstance(matched, str) and "<" in matched and matched not in text:
                escaped = escape(matched)
                if escaped != matched and escaped in text:
                    finished_reason = dict(finished_reason)
                    finished_reason["matched"] = escaped
        return finished_reason
'''


def replace_once(text, old, new, what):
    assert text.count(old) == 1, f"detokenizer_manager.py: {what} anchor not found"
    return text.replace(old, new, 1)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    module = f"{path}/python/sglang/srt/managers/control_tag_escape.py"
    with open(module, "w") as f:
        f.write(MODULE_SRC)
    print("wrote", module)

    p = f"{path}/python/sglang/srt/managers/detokenizer_manager.py"
    text = open(p).read()

    text = replace_once(
        text,
        "from sglang.srt.managers.io_struct import (\n",
        "from sglang.srt.managers.control_tag_escape import ControlTagEscaper\n"
        "from sglang.srt.managers.io_struct import (\n",
        "io_struct import",
    )

    text = replace_once(
        text,
        """        self.is_tool_call_parser_gpt_oss = get_serving().tool_call_parser == "gpt-oss"
""",
        """        self.is_tool_call_parser_gpt_oss = get_serving().tool_call_parser == "gpt-oss"

        # gfx1151 patch 33: entity-escape spelled-out control strings in the
        # decoded text so the text-based reasoning and tool-call parsers only
        # match the real tokens (SGLANG_UNESCAPE_CONTROL_TAGS=1; patch 32
        # decodes the entities again for the client).
        if self.disable_tokenizer_batch_decode:

            def _decode_rows(ids_list, skip_list, space_list):
                return [
                    self.tokenizer.decode(
                        ids, skip_special_tokens=skip, spaces_between_special_tokens=space
                    )
                    for ids, skip, space in zip(ids_list, skip_list, space_list)
                ]

        else:
            _decode_rows = self._grouped_batch_decode
        self.control_tag_escaper = ControlTagEscaper.create(self.tokenizer, _decode_rows)
""",
        "is_tool_call_parser_gpt_oss",
    )

    text = replace_once(
        text,
        """
        # Incremental decoding
        output_strs = []
""",
        """
        # gfx1151 patch 33: rows whose text holds "<" are re-decoded in runs
        # between the real control tokens and the runs entity-escaped, so a
        # spelled-out </think> or <tool_call> never reaches the text parsers.
        read_held = None
        if self.control_tag_escaper is not None:
            read_held = self.control_tag_escaper.rewrite(
                surr_ids,
                read_ids,
                surr_texts,
                read_texts,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )

        # Incremental decoding
        output_strs = []
""",
        "Incremental decoding",
    )

    text = replace_once(
        text,
        """                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("\ufffd"):
""",
        """                pending = s.sent_offset - s.decoded_text_len
                # gfx1151 patch 33: a trailing proper prefix of a control
                # string is held back like an incomplete UTF-8 sequence.
                held = read_held[i] if read_held is not None else 0
                if new_text and not new_text.endswith("\ufffd") and not held:
""",
        "streaming commit condition",
    )

    text = replace_once(
        text,
        """                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
""",
        """                    if held and not new_text.endswith("\ufffd"):
                        printable = new_text[: max(0, len(new_text) - held)]
                    else:
                        printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
""",
        "streaming printable prefix",
    )

    text = replace_once(
        text,
        """            output_str = self.trim_matched_stop(
                s.get_decoded_text() + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
""",
        """            output_str = s.get_decoded_text() + new_text
            finished_reason = recv_obj.finished_reasons[i]
            if self.control_tag_escaper is not None:
                # gfx1151 patch 33: a stop string matched on raw text may be
                # present only in its escaped form here.
                finished_reason = self.control_tag_escaper.escape_finished_reason(
                    finished_reason, output_str
                )
            output_str = self.trim_matched_stop(
                output_str,
                finished_reason,
                recv_obj.no_stop_trim[i],
            )
""",
        "finished trim",
    )

    open(p, "w").write(text)
    print("patched", p)


if __name__ == "__main__":
    main()
