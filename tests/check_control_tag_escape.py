#!/usr/bin/env python3
"""Exercise patch 33's detokenizer-side escaping without a GPU.

Loads the module source embedded in patches/patch_escape_control_text.py and
checks: escape and hold-back arithmetic; run-wise decode around real control
tokens equals whole decode when nothing is spelled out; the patched
incremental loop emits the same escaped text for any chunking, holds back a
partial control string, keeps real tokens literal, trims stop strings in
either form, and leaves rows without "<" untouched; the tag list matches
patch 32's and the template's.

The incremental loop is the real DetokenizerManager when sglang imports
(inside the image: docker run with this repo and the model mounted),
otherwise a mirror of the patched method below. With MODEL_DIR set (or the
default checkpoint present) the same scenarios run on the real tokenizer.

Run locally with: python3 tests/check_control_tag_escape.py
In the image (the sglang import probes the GPU, so pass the devices):
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --group-add render --security-opt seccomp=unconfined \
    -e SGLANG_UNESCAPE_CONTROL_TAGS=1 -e MODEL_DIR=/model \
    -v $PWD:/repo:ro -v /opt/llm/models/<ckpt>:/model:ro \
    --entrypoint python3 strix-halo-sglang:dev /repo/tests/check_control_tag_escape.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import re
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "patch_escape_control_text.py"
PATCH32 = ROOT / "patches" / "patch_unescape_control_tags.py"
TEMPLATE = ROOT / "configs" / "chat" / "qwen38.jinja"
DEFAULT_MODEL = "/opt/llm/models/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ-ple-fp8"

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'✓' if ok else '✗'} {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def load_module(patch_path, name):
    spec = importlib.util.spec_from_file_location(name + "_patch", patch_path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    mod = types.ModuleType(name)
    exec(compile(patch.MODULE_SRC, name + ".py", "exec"), mod.__dict__)
    return mod


# --- a byte-level tokenizer stand-in -----------------------------------------

class FakeTokenizer:
    """Pieces are bytes (so a multibyte character can be split across tokens,
    as with byte-level BPE); added tokens decode to their literal string, the
    special ones to "" under skip_special_tokens."""

    is_fast = True
    unk_token_id = None

    def __init__(self, pieces: List[bytes], added: Dict[str, bool]):
        self.pieces = list(pieces)
        self.added = {}
        for tag, special in added.items():
            self.added[tag] = (len(self.pieces), special)
            self.pieces.append(tag.encode())
        self.special_ids = {i for i, s in self.added.values() if s}

    def __len__(self):
        return len(self.pieces)

    def convert_tokens_to_ids(self, tok):
        if tok in self.added:
            return self.added[tok][0]
        return self.unk_token_id

    def convert_ids_to_tokens(self, i):
        return self.pieces[i].decode("utf-8", errors="replace")

    def decode(self, ids, skip_special_tokens=False, spaces_between_special_tokens=True):
        return b"".join(
            self.pieces[i] for i in ids if not (skip_special_tokens and i in self.special_ids)
        ).decode("utf-8", errors="replace")

    def batch_decode(self, ids_list, skip_special_tokens=False, spaces_between_special_tokens=True):
        return [self.decode(ids, skip_special_tokens, spaces_between_special_tokens) for ids in ids_list]


PIECES = [b"The", b" note", b" says", b" </", b"think", b">", b" and", b" <|", b"im", b"_end",
          b"|", b">", b" ok", b"\n\n", b"Answer", b" <", b"tool", b"_call", b" x", b"assistant",
          b"\n", b"\xc3", b"\xa9", b" plain", b" text", b" 1 < 2", b" <div>"]
ADDED = {"<|im_start|>": True, "<|im_end|>": True, "<|endoftext|>": True, "<think>": False,
         "</think>": False, "<tool_call>": False, "</tool_call>": False,
         "<tool_response>": False, "</tool_response>": False}


# --- the incremental loop ----------------------------------------------------

def find_printable_text(text: str):
    # sglang.utils.find_printable_text without the CJK cases (ASCII fixtures).
    if text.endswith("\n"):
        return text
    return text[: text.rfind(" ") + 1]


@dataclasses.dataclass
class DecodeStatus:
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    sent_offset: int = 0
    decoded_text_len: int = dataclasses.field(init=False)
    decoded_text_chunks: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.decoded_text_len = len(self.decoded_text)

    def append_decoded_text(self, text: str):
        if text:
            self.decoded_text_chunks.append(text)
            self.decoded_text_len += len(text)

    def get_decoded_text(self) -> str:
        if self.decoded_text_chunks:
            self.decoded_text += "".join(self.decoded_text_chunks)
            self.decoded_text_chunks.clear()
        return self.decoded_text


class MirrorDetokenizer:
    """_decode_batch_token_id_output of the patched detokenizer_manager.py,
    with the batch-decode helper reduced to tokenizer.batch_decode. Keep in
    step with the hooks in patches/patch_escape_control_text.py."""

    def __init__(self, tokenizer, escaper_cls):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.decode_status = OrderedDict()
        self.is_tool_call_parser_gpt_oss = False
        self.control_tag_escaper = escaper_cls.create(tokenizer, self._grouped_batch_decode)

    def _grouped_batch_decode(self, ids_list, skip_list, space_list):
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip, spaces_between_special_tokens=space)
            if ids else ""
            for ids, skip, space in zip(ids_list, skip_list, space_list)
        ]

    def trim_matched_stop(self, output, finished_reason, no_stop_trim):
        if not finished_reason:
            return output
        matched = finished_reason.get("matched", None)
        if not matched:
            return output
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            return output[:-1]
        return output

    def _decode_batch_token_id_output(self, recv_obj):
        bs = len(recv_obj.rids)
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=list(recv_obj.decode_ids[i]),
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])
            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset:], recv_obj.finished_reasons[i], recv_obj.no_stop_trim[i]
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset: s.read_offset])

        surr_texts = self._grouped_batch_decode(
            surr_ids, recv_obj.skip_special_tokens, recv_obj.spaces_between_special_tokens
        )
        read_texts = self._grouped_batch_decode(
            read_ids, recv_obj.skip_special_tokens, recv_obj.spaces_between_special_tokens
        )

        # gfx1151 patch 33
        read_held = None
        if self.control_tag_escaper is not None:
            read_held = self.control_tag_escaper.rewrite(
                surr_ids, read_ids, surr_texts, read_texts,
                recv_obj.skip_special_tokens, recv_obj.spaces_between_special_tokens,
            )

        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            s = self.decode_status[rid]
            new_text = read_texts[i][len(surr_texts[i]):]
            if recv_obj.finished_reasons[i] is None:
                pending = s.sent_offset - s.decoded_text_len
                held = read_held[i] if read_held is not None else 0
                if new_text and not new_text.endswith("\ufffd") and not held:
                    s.append_decoded_text(new_text)
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    output_strs.append(new_text[pending:] if pending else new_text)
                else:
                    if held and not new_text.endswith("\ufffd"):
                        printable = new_text[: max(0, len(new_text) - held)]
                    else:
                        printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    output_strs.append(printable[pending:] if pending else printable)
                continue

            if rid in self.decode_status:
                del self.decode_status[rid]

            output_str = s.get_decoded_text() + new_text
            finished_reason = recv_obj.finished_reasons[i]
            if self.control_tag_escaper is not None:
                finished_reason = self.control_tag_escaper.escape_finished_reason(
                    finished_reason, output_str
                )
            output_str = self.trim_matched_stop(output_str, finished_reason, recv_obj.no_stop_trim[i])
            incremental_output = output_str[s.sent_offset:]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)
        return output_strs


def real_detokenizer(tokenizer):
    """The patched class from the image, or None when sglang does not import."""
    try:
        from sglang.srt.managers import detokenizer_manager as dm
        from sglang.srt.managers.control_tag_escape import ControlTagEscaper
    except Exception as e:  # noqa: BLE001
        print(f"  (sglang detokenizer not importable here: {type(e).__name__}; using the mirror)")
        return None
    d = dm.DetokenizerManager.__new__(dm.DetokenizerManager)
    d.tokenizer = tokenizer
    d.vocab_size = len(tokenizer)
    d.decode_status = dm.LimitedCapacityDict(capacity=1024)
    d.disable_tokenizer_batch_decode = False
    d.is_tool_call_parser_gpt_oss = False
    d.control_tag_escaper = ControlTagEscaper.create(tokenizer, d._grouped_batch_decode)
    return d


class Batch:
    def __init__(self, rids, decode_ids, read_offsets=None, finished=None, skip=True, space=True,
                 no_stop_trim=False, decoded_texts=None):
        n = len(rids)
        self.rids = rids
        self.decode_ids = decode_ids
        self.read_offsets = read_offsets or [0] * n
        self.finished_reasons = finished or [None] * n
        self.skip_special_tokens = [skip] * n
        self.spaces_between_special_tokens = [space] * n
        self.no_stop_trim = [no_stop_trim] * n
        self.decoded_texts = decoded_texts or [""] * n


def run_stream(detok, prompt_tail, output, finish, chunks, rid="r", **kw):
    """Feed prompt_tail (as the surrogate context, read_offset past it) then
    output in the given chunk sizes; the last chunk carries finish. Returns
    the list of emitted strings."""
    emitted = []
    pos = 0
    first = True
    sizes = list(chunks)
    assert sum(sizes) == len(output)
    for k, size in enumerate(sizes):
        ids = output[pos: pos + size]
        pos += size
        last = k == len(sizes) - 1
        if first:
            b = Batch([rid], [list(prompt_tail) + ids], read_offsets=[len(prompt_tail)],
                      finished=[finish if last else None], **kw)
            first = False
        else:
            b = Batch([rid], [ids], finished=[finish if last else None], **kw)
        emitted.extend(detok._decode_batch_token_id_output(b))
    return emitted


def random_chunks(rng, n):
    sizes, i = [], 0
    while i < n:
        j = min(n, i + rng.randint(1, 5))
        sizes.append(j - i)
        i = j
    return sizes


def main() -> int:
    os.environ["SGLANG_UNESCAPE_CONTROL_TAGS"] = "1"
    mod = load_module(PATCH, "control_tag_escape")
    mod32 = load_module(PATCH32, "control_tag_unescape") if PATCH32.exists() else None
    tags = mod.CONTROL_TAGS

    print("tag list:")
    jinja = TEMPLATE.read_text()
    m = re.search(r"\{%- for s in \[([^\]]*)\] %\}", jinja)
    template_tags = tuple(re.findall(r"'([^']+)'", m.group(1))) if m else ()
    check("CONTROL_TAGS == template escape list", template_tags == tags, f"{template_tags} vs {tags}")
    if mod32 is not None:
        check("CONTROL_TAGS == patch 32's", mod32.CONTROL_TAGS == tags)
        check("switch reads the same variable as patch 32",
              mod.enabled() == mod32.enabled() and mod.enabled())
        check("patch 32 decodes patch 33's output", all(
            mod32.unescape(mod.escape(f"a {t} b")) == f"a {t} b" for t in tags))

    print("escape:")
    for t in tags:
        check(f"{t} -> &lt;{t[1:-1]}&gt;", mod.escape(f"a {t} b") == f"a &lt;{t[1:-1]}&gt; b")
    for other in ["&lt;/think&gt;", "<thinking>", "<tool_calls>", "<div>", "1 < 2", "", "<", "</", "<|im_end"]:
        check(f"untouched: {other!r}", mod.escape(other) == other)
    check("nested brackets", mod.escape("<<think>>") == "<&lt;think&gt;>")
    check("adjacent tags", mod.escape("</think><tool_call>") == "&lt;/think&gt;&lt;tool_call&gt;")

    print("hold-back:")
    for text, want in [("", 0), ("a", 0), ("a<", 1), ("a</", 2), ("a</thi", 5), ("a</think>", 0),
                       ("a<|im_end|", 9), ("<x", 0), ("a<b<", 1), ("<|endoftext|", 12),
                       ("<|endoftext|>", 0), ("</think><", 1), ("<tool_respons", 13),
                       ("<tool_response", 14), ("x</thinky", 0), ("&lt;/thi", 0)]:
        check(f"held({text!r}) == {want}", mod.held_suffix_len(text) == want, str(mod.held_suffix_len(text)))
    check("_MAX_HOLD is longest tag - 1", mod._MAX_HOLD == len("</tool_response>") - 1)

    print("fake tokenizer:")
    tok = FakeTokenizer(PIECES, ADDED)
    ids = mod.control_token_ids(tok)
    check("all nine control tokens found", sorted(ids.values()) == sorted(tags))
    T = {t: tok.convert_tokens_to_ids(t) for t in tags}

    def decode_rows(ids_list, skips, spaces):
        return [tok.decode(i, skip_special_tokens=s, spaces_between_special_tokens=sp) if i else ""
                for i, s, sp in zip(ids_list, skips, spaces)]

    esc = mod.ControlTagEscaper(ids, decode_rows)
    # Pieces: 0 The 1 ' note' 2 ' says' 3 ' </' 4 think 5 > 6 ' and' 7 ' <|' 8 im 9 _end 10 | 11 >
    # 12 ' ok' 13 \n\n 14 Answer 15 ' <' 16 tool 17 _call 18 ' x' 19 assistant 20 \n 21/22 é
    prompt_tail = [T["<|im_start|>"], 19, 20, T["<think>"], 20]
    output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, T["</think>"], 13, 14, 15, 16, 17, 5, 18, 21, 22,
              T["<|im_end|>"]]
    expected = "The note says &lt;/think&gt; and &lt;|im_end|&gt; ok</think>\n\nAnswer &lt;tool_call&gt; xé"
    eos = {"type": "stop", "matched": T["<|im_end|>"]}

    texts, held = esc.decode([prompt_tail + output], [True], [True])
    check("run-wise decode escapes spelled-out strings, keeps real tokens",
          texts[0] == "assistant\n<think>\n" + expected, repr(texts[0]))
    check("no hold-back after a trailing real token", held == [0])
    texts, held = esc.decode([output[:9]], [True], [True])
    check("hold-back reported for a trailing partial", held == [4] and texts[0].endswith(" <|im"), repr((texts, held)))
    texts, _ = esc.decode([prompt_tail + output], [False], [True])
    check("skip_special_tokens=False keeps the real <|im_start|>/<|im_end|> literal",
          texts[0] == "<|im_start|>assistant\n<think>\n" + expected + "<|im_end|>", repr(texts[0]))

    plain = [23, 24, 20, 0]
    surr_t, read_t = decode_rows([plain[:2], [25]], [True, True], [True, True]), decode_rows([plain, [25, 26]], [True, True], [True, True])
    s0, r0 = list(surr_t), list(read_t)
    held = esc.rewrite([plain[:2], [25]], [plain, [25, 26]], surr_t, read_t, [True, True], [True, True])
    check("rewrite leaves rows without a control string unchanged (even with '<')",
          surr_t == s0 and read_t == r0 and held == [0, 0], repr((surr_t, read_t, held)))

    fr = {"type": "stop", "matched": "</think>"}
    check("finished_reason: raw match kept when present", esc.escape_finished_reason(fr, "a </think> b") is fr)
    out = esc.escape_finished_reason(fr, "a &lt;/think&gt; b")
    check("finished_reason: escaped copy when only the escaped form is present",
          out == {"type": "stop", "matched": "&lt;/think&gt;"} and fr["matched"] == "</think>")
    check("finished_reason: int / None / absent pass through",
          esc.escape_finished_reason(eos, "x") is eos and esc.escape_finished_reason(None, "x") is None
          and esc.escape_finished_reason({"type": "length"}, "x") == {"type": "length"})

    for label, make in (("mirror loop", lambda: MirrorDetokenizer(tok, mod.ControlTagEscaper)),
                        ("real detokenizer", lambda: real_detokenizer(tok))):
        detok = make()
        if detok is None:
            continue
        print(f"{label}:")
        one = run_stream(detok, prompt_tail, output, eos, [len(output)])
        check("one-shot output", "".join(one) == expected, repr(one))
        per = run_stream(detok, prompt_tail, output, eos, [1] * len(output))
        check("token-at-a-time stream == one-shot", "".join(per) == expected, repr("".join(per)))
        sofar = "".join(per[:4])
        check("partial ' </' held back, not emitted", sofar == "The note says ", repr(sofar))
        sofar = "".join(per[:6])
        check("complete spelled-out </think> emitted escaped", sofar == "The note says &lt;/think&gt;", repr(sofar))
        sofar = "".join(per[:14])
        check("real </think> token emitted literal", sofar.endswith(" ok</think>"), repr(sofar))
        sofar = "".join(per[:22])
        check("split UTF-8 still held", not sofar.endswith("\ufffd") and sofar.endswith(" x"), repr(sofar))
        check("no emitted chunk contains a raw control string",
              not any(t in "".join(per) for t in tags if t != "</think>"))
        rng = random.Random(33)
        ok = True
        for _ in range(300):
            got = "".join(run_stream(detok, prompt_tail, output, eos, random_chunks(rng, len(output))))
            if got != expected:
                ok, bad = False, got
                break
        check("300 random chunkings == one-shot", ok, "" if ok else repr(bad))
        got = "".join(run_stream(detok, prompt_tail, output, eos, [1] * len(output), skip=False))
        check("skip_special_tokens=False: same text, EOS trimmed", got == expected, repr(got))
        got = "".join(run_stream(detok, prompt_tail, output, eos, [len(output)], skip=False, no_stop_trim=True))
        check("no_stop_trim keeps the real <|im_end|>", got == expected + "<|im_end|>", repr(got))
        # The scheduler stops at the match, so the matched text ends the output.
        stop = {"type": "stop", "matched": "</think>"}
        got = "".join(run_stream(detok, prompt_tail, output[:14], stop, [1] * 14))
        check("stop string matched on the real token trims there",
              got == "The note says &lt;/think&gt; and &lt;|im_end|&gt; ok", repr(got))
        got = "".join(run_stream(detok, prompt_tail, output[:14], stop, [14], no_stop_trim=True))
        check("no_stop_trim keeps the real token literal", got == "The note says &lt;/think&gt; and &lt;|im_end|&gt; ok</think>", repr(got))
        got = "".join(run_stream(detok, prompt_tail, output[:6], stop, [1] * 6))
        check("stop string matched on a spelled-out copy trims there", got == "The note says ", repr(got))
        got = "".join(run_stream(detok, prompt_tail, output[:6], stop, [6], no_stop_trim=True))
        check("no_stop_trim keeps the spelled-out copy, escaped", got == "The note says &lt;/think&gt;", repr(got))
        plain_out = [23, 24, 25, 26, 13, 0, T["<|im_end|>"]]
        got = "".join(run_stream(detok, prompt_tail, plain_out, eos, [1] * len(plain_out)))
        check("text without control strings is untouched", got == " plain text 1 < 2 <div>\n\nThe", repr(got))
        # Two rows in one batch, one needing the rewrite.
        b = Batch(["a", "b"], [prompt_tail + output, prompt_tail + plain_out], read_offsets=[5, 5], finished=[eos, eos])
        got = detok._decode_batch_token_id_output(b)
        check("mixed batch: rows handled independently", got == [expected, " plain text 1 < 2 <div>\n\nThe"], repr(got))
        # Switch off: identical to upstream behaviour (raw strings pass through).
        os.environ["SGLANG_UNESCAPE_CONTROL_TAGS"] = "0"
        off = make()
        os.environ["SGLANG_UNESCAPE_CONTROL_TAGS"] = "1"
        check("escaper is None when the switch is off", off.control_tag_escaper is None)
        got = "".join(run_stream(off, prompt_tail, output, eos, [1] * len(output)))
        check("switch off: raw text", got == expected.replace("&lt;", "<").replace("&gt;", ">"), repr(got))

    model_dir = os.environ.get("MODEL_DIR") or (DEFAULT_MODEL if os.path.isdir(DEFAULT_MODEL) else None)
    if model_dir:
        print(f"real tokenizer ({model_dir}):")
        try:
            from transformers import AutoTokenizer
            rt = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  (skipped: {type(e).__name__}: {e})")
            rt = None
        if rt is not None:
            rids = mod.control_token_ids(rt)
            check("nine single-token control strings", sorted(rids.values()) == sorted(tags), repr(rids))
            RT = {v: k for k, v in rids.items()}

            def rdecode(ids_list, skips, spaces):
                return [rt.decode(i, skip_special_tokens=s, spaces_between_special_tokens=sp) if i else ""
                        for i, s, sp in zip(ids_list, skips, spaces)]

            resc = mod.ControlTagEscaper(rids, rdecode)
            rng = random.Random(7)
            vocab = [i for i in range(256, 20000)]
            ok = True
            for _ in range(200):
                seq = [rng.choice(vocab) for _ in range(rng.randint(1, 30))]
                for _ in range(rng.randint(0, 3)):
                    seq.insert(rng.randint(0, len(seq)), rng.choice(list(rids)))
                for skip in (True, False):
                    whole = rt.decode(seq, skip_special_tokens=skip)
                    runs, _ = resc.decode([seq], [skip], [True])
                    # Equal unless random tokens happened to spell a control
                    # string out, in which case one level of decode restores it.
                    if runs[0] != whole and (mod32 is None or mod32.unescape(runs[0]) != whole):
                        ok = False
                        bad = (seq, skip, whole, runs[0])
                        break
                if not ok:
                    break
            check("run-wise decode == whole decode on 200 random sequences with real control tokens", ok,
                  "" if ok else repr(bad))

            def spelled(s):
                # The model's own spelling: ordinary tokens, never the added token.
                out = []
                for piece in re.findall(r" ?<\||</|<| ?[A-Za-z_]+|\||>|[^<>|A-Za-z_ ]+| ", s):
                    out.extend(rt.encode(piece, add_special_tokens=False))
                return out

            text = "The note says </think> and <|im_end|> ok"
            sp_ids = spelled(text)
            check("spelled-out pieces hold no real control token", not any(i in rids for i in sp_ids))
            check("spelled-out pieces decode back to the text", rt.decode(sp_ids) == text, repr(rt.decode(sp_ids)))
            r_prompt_tail = [RT["<|im_start|>"]] + rt.encode("assistant\n", add_special_tokens=False) + [RT["<think>"]] + rt.encode("\n", add_special_tokens=False)
            r_output = sp_ids + [RT["</think>"]] + rt.encode("\n\nAnswer ", add_special_tokens=False) + spelled("<tool_call>") + rt.encode(" x é", add_special_tokens=False) + [RT["<|im_end|>"]]
            r_expected = "The note says &lt;/think&gt; and &lt;|im_end|&gt; ok</think>\n\nAnswer &lt;tool_call&gt; x é"
            r_eos = {"type": "stop", "matched": RT["<|im_end|>"]}
            for label, make in (("mirror loop", lambda: MirrorDetokenizer(rt, mod.ControlTagEscaper)),
                                ("real detokenizer", lambda: real_detokenizer(rt))):
                detok = make()
                if detok is None:
                    continue
                one = "".join(run_stream(detok, r_prompt_tail, r_output, r_eos, [len(r_output)]))
                check(f"{label}: one-shot on real tokens", one == r_expected, repr(one))
                per = "".join(run_stream(detok, r_prompt_tail, r_output, r_eos, [1] * len(r_output)))
                check(f"{label}: token-at-a-time on real tokens", per == r_expected, repr(per))
                rng = random.Random(1)
                ok = all(
                    "".join(run_stream(detok, r_prompt_tail, r_output, r_eos, random_chunks(rng, len(r_output)))) == r_expected
                    for _ in range(100)
                )
                check(f"{label}: 100 random chunkings on real tokens", ok)
    else:
        print("real tokenizer: skipped (set MODEL_DIR)")

    print(f"\n{checks - len(failures)}/{checks} checks passed.")
    if failures:
        print("FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
