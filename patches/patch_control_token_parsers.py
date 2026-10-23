#!/usr/bin/env python3
"""Reasoning and tool-call parsers match control tokens by id, not text
(patch 33).

The reasoning parser (--reasoning-parser qwen3) and the tool-call parser
(--tool-call-parser qwen3_coder) work on detokenized text. The model can
spell "</think>" or "<tool_call>" with ordinary tokens, quoting a file or
discussing the format, and the text looks the same as the control token:
the reasoning parser ends the thinking block at the quote, and the
tool-call parser opens a call at a quoted "<tool_call>".

The serving layer already receives the output token ids with every chunk.
This patch adds a ControlTokenStream per request that decodes those ids
beside the text the parsers receive and records where a real control token
begins. Parsers search a "view" of their text in which every marker
occurrence that does not begin at a control token id has its first
character replaced, so only genuine control tokens change their state. The
text returned to the client is never the view; nothing is rewritten.
Without output ids, or if the decoded ids ever disagree with the text, the
parser falls back to today's text matching with one warning.

Hooks: a new module parser/control_token_stream.py; BaseReasoningFormatDetector
and BaseFormatDetector search through _marker_view; Qwen3CoderDetector finds
<tool_call>/</tool_call> through the view; ReasoningParser and
FunctionCallParser gain attach_control_stream; serving_chat.py creates the
stream per choice, feeds ids and text, and attaches it to both parsers in
the streaming and non-streaming paths. Markers that are not single tokens
of the tokenizer are matched as text, as before. Anchors are asserted so an
upstream rewrite fails the build. See patches/33-control-token-parsers.md.
"""
import sys

MODULE_SRC = '''\
"""Tell a request\'s control tokens from the same strings spelled out.

gfx1151 patch 33. The reasoning and tool-call parsers receive detokenized
text, in which the control token </think> and the ordinary tokens "</",
"think", ">" look alike. A ControlTokenStream decodes the request\'s output
ids beside that text and records where each control token begins; view()
hands a parser its text with every look-alike defused, so only real control
tokens change its state.
"""
import logging
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Replaces the first character of a marker spelled with ordinary tokens. The
# view is only searched, never returned to a client.
_MASK = "\\x00"


def parser_markers(
    reasoning_parser: Optional[str], tool_call_parser: Optional[str]
) -> List[str]:
    """The marker strings the configured parsers switch state on."""
    markers: List[str] = []
    if reasoning_parser:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        detector_class = ReasoningParser.DetectorMap.get(reasoning_parser.lower())
        if detector_class is not None:
            try:
                detector = detector_class()
            except Exception:
                detector = None
            if detector is not None:
                markers += [
                    getattr(detector, "think_start_token", None),
                    getattr(detector, "think_end_token", None),
                    getattr(detector, "tool_start_token", None),
                ]
    if tool_call_parser:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        detector_class = FunctionCallParser.ToolCallParserEnum.get(tool_call_parser)
        if detector_class is not None:
            try:
                detector = detector_class()
            except Exception:
                detector = None
            if detector is not None:
                markers += [
                    getattr(detector, "tool_call_start_token", None),
                    getattr(detector, "tool_call_end_token", None),
                ]
    return sorted({m for m in markers if isinstance(m, str) and m})


def resolve_control_ids(tokenizer, markers: Iterable[str]) -> Dict[int, str]:
    """The markers that are single tokens of this tokenizer, by id. Others
    have no id to match and are left to text matching."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return {}
    out: Dict[int, str] = {}
    for marker in markers:
        token_id = backend.token_to_id(marker)
        if token_id is None:
            continue
        if backend.decode([token_id], skip_special_tokens=False) != marker:
            continue
        out[token_id] = marker
    return out


class ControlTokenStream:
    """One request\'s output ids, decoded beside the text its parsers see.

    feed() takes the output ids (every id so far, or only the new ones);
    consume() the text handed to the first parser. view(text), for text that
    ends the consumed text, returns it with each marker occurrence that does
    not begin at a control token defused. The view has the text\'s length,
    so an index found in it slices the text itself.

    The stream applies once ids have been fed. It stops applying, and view()
    returns the text unchanged, if the text ever disagrees with the decoded
    ids; that is logged once.
    """

    def __init__(
        self, backend, control_ids: Dict[int, str], skip_special_tokens: bool
    ):
        from tokenizers.decoders import DecodeStream

        self._backend = backend
        self._control_ids = dict(control_ids)
        self._markers = sorted(set(control_ids.values()), key=len, reverse=True)
        self._decoder = DecodeStream(skip_special_tokens=skip_special_tokens)
        self._decoded = ""
        self._control_starts = set()
        self._fed = 0
        self.consumed_len = 0
        # Chars at the end of the consumed text that the parser now searching
        # has not been handed yet (a reasoning parser holds back a partial
        # marker before passing the rest on to the tool-call parser).
        self._tail_lag = 0
        self._aligned = True

    @property
    def active(self) -> bool:
        return self._fed > 0 and self._aligned

    def feed(self, output_ids: Sequence[int], cumulative: bool) -> None:
        """Decode output ids: all of them so far when ``cumulative``."""
        if not self._aligned:
            return
        new_ids = output_ids[self._fed :] if cumulative else output_ids
        if not new_ids:
            return
        pieces = []
        decoded_len = len(self._decoded)
        try:
            for token_id in new_ids:
                piece = self._decoder.step(self._backend, token_id)
                if not piece:
                    continue
                marker = self._control_ids.get(token_id)
                if marker is not None and piece.endswith(marker):
                    self._control_starts.add(decoded_len + len(piece) - len(marker))
                pieces.append(piece)
                decoded_len += len(piece)
        except Exception as exc:  # an id the tokenizer cannot decode
            self._lose_alignment(f"output ids failed to decode: {exc}")
            return
        self._fed += len(new_ids)
        if pieces:
            self._decoded = "".join([self._decoded, *pieces])

    def consume(self, text: str) -> None:
        """Record ``text`` as the next text handed to the first parser."""
        self.consumed_len += len(text)
        self._tail_lag = 0

    def set_tail_lag(self, held_back: int) -> None:
        """The next parser\'s text ends ``held_back`` chars before the consumed
        text: what the parser before it is still holding."""
        self._tail_lag = max(0, held_back)

    def view(self, text: str) -> str:
        """``text``, which must end the consumed text, with every marker that
        does not begin at a control token defused."""
        if not text or not self.active:
            return text
        end = self.consumed_len - self._tail_lag
        start = end - len(text)
        if start < 0:
            self._lose_alignment("parser text is longer than the text consumed")
            return text
        # The decoded ids may run ahead of the text (ids arrive before the
        # detokenizer commits a character) but never behind it, so the text
        # must be a prefix of the decoded text wherever both exist.
        avail = min(end, len(self._decoded))
        if avail > start and self._decoded[start:avail] != text[: avail - start]:
            self._lose_alignment("parser text does not match the decoded output ids")
            return text
        view = None
        for marker in self._markers:
            i = text.find(marker)
            while i != -1:
                pos = start + i
                if pos < avail and pos not in self._control_starts:
                    if view is None:
                        view = list(text)
                    view[i] = _MASK
                i = text.find(marker, i + 1)
        return text if view is None else "".join(view)

    def _lose_alignment(self, why: str) -> None:
        if self._aligned:
            logger.warning(
                "%s; matching reasoning and tool-call markers as text for the "
                "rest of this output.",
                why,
            )
        self._aligned = False
'''


def patch(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old)
    return text.replace(old, new)


def patch_reasoning_parser(path):
    text = open(path).read()
    text = patch(
        text,
        "class BaseReasoningFormatDetector:\n"
        '    """Base class providing two sets of interfaces: one-time and streaming incremental."""\n'
        "\n"
        "    def __init__(\n",
        "class BaseReasoningFormatDetector:\n"
        '    """Base class providing two sets of interfaces: one-time and streaming incremental."""\n'
        "\n"
        "    # gfx1151 patch 33: a ControlTokenStream when markers are matched by\n"
        "    # token id; see _marker_view.\n"
        "    _control = None\n"
        "\n"
        "    def _marker_view(self, text: str) -> str:\n"
        '        """The string searched for markers in ``text``, which ends the text\n'
        "        parsed so far. Same length as ``text``, so its indices slice ``text``;\n"
        '        a marker spelled with ordinary tokens is defused in it."""\n'
        "        if self._control is None:\n"
        "            return text\n"
        "        return self._control.view(text)\n"
        "\n"
        "    def __init__(\n",
    )
    # one-shot parsing
    text = patch(
        text,
        "    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:\n"
        "        in_reasoning = self._in_reasoning or self.think_start_token in text\n",
        "    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:\n"
        "        in_reasoning = (\n"
        "            self._in_reasoning or self.think_start_token in self._marker_view(text)\n"
        "        )  # gfx1151 patch 33\n",
    )
    text = patch(
        text,
        "        processed_text = text\n"
        "        while processed_text.startswith(think_start_text):\n"
        "            processed_text = processed_text[len(think_start_text) :]\n"
        "\n"
        "        if (\n"
        "            self.think_end_token not in processed_text\n"
        "            and self.think_end_token not in self.previous_content\n"
        "        ):\n"
        "            # Check for tool_start_token interruption\n"
        "            if (\n"
        "                in_reasoning\n"
        "                and self.tool_start_token is not None\n"
        "                and self.tool_start_token in processed_text\n"
        "            ):\n"
        "                # Find the first occurrence of tool_start_token and split there\n"
        "                tool_idx = processed_text.find(self.tool_start_token)\n",
        "        processed_text = text\n"
        "        while self._marker_view(processed_text).startswith(think_start_text):\n"
        "            processed_text = processed_text[len(think_start_text) :]\n"
        "\n"
        "        # gfx1151 patch 33: markers are located in the view, sliced from the text.\n"
        "        end_idx = self._marker_view(processed_text).find(self.think_end_token)\n"
        "        if end_idx == -1 and self.think_end_token not in self.previous_content:\n"
        "            # Check for tool_start_token interruption\n"
        "            tool_idx = (\n"
        "                self._marker_view(processed_text).find(self.tool_start_token)\n"
        "                if in_reasoning and self.tool_start_token is not None\n"
        "                else -1\n"
        "            )\n"
        "            if tool_idx != -1:\n",
    )
    text = patch(
        text,
        "        # Extract reasoning content\n"
        "        if self.think_end_token in processed_text:\n"
        "            splits = processed_text.split(self.think_end_token, maxsplit=1)\n"
        "            reasoning_text = splits[0]\n"
        "            normal_text = splits[1]\n",
        "        # Extract reasoning content\n"
        "        if end_idx != -1:\n"
        "            reasoning_text = processed_text[:end_idx]\n"
        "            normal_text = processed_text[end_idx + len(self.think_end_token) :]\n",
    )
    # streaming
    text = patch(
        text,
        "        # Strip `<think>` token if present\n"
        "        if not self.stripped_think_start and think_start_text in current_text:\n"
        "            current_text = current_text.replace(think_start_text, \"\", 1)\n",
        "        # Strip `<think>` token if present\n"
        "        start_idx = (\n"
        "            -1\n"
        "            if self.stripped_think_start\n"
        "            else self._marker_view(current_text).find(think_start_text)\n"
        "        )  # gfx1151 patch 33\n"
        "        if start_idx != -1:\n"
        "            current_text = (\n"
        "                current_text[:start_idx]\n"
        "                + current_text[start_idx + len(think_start_text) :]\n"
        "            )\n",
    )
    text = patch(
        text,
        "        # Handle end of reasoning block\n"
        "        if self._in_reasoning and self.think_end_token in current_text:\n"
        "            end_idx = current_text.find(self.think_end_token)\n"
        "\n"
        "            reasoning_text = current_text[:end_idx]\n",
        "        # Handle end of reasoning block\n"
        "        end_idx = (\n"
        "            self._marker_view(current_text).find(self.think_end_token)\n"
        "            if self._in_reasoning\n"
        "            else -1\n"
        "        )  # gfx1151 patch 33\n"
        "        if end_idx != -1:\n"
        "            reasoning_text = current_text[:end_idx]\n",
    )
    text = patch(
        text,
        "            if self.tool_start_token and self.tool_start_token in current_text:\n"
        "                tool_idx = current_text.find(self.tool_start_token)\n"
        "                reasoning_text = current_text[:tool_idx]\n",
        "            tool_idx = (\n"
        "                self._marker_view(current_text).find(self.tool_start_token)\n"
        "                if self.tool_start_token\n"
        "                else -1\n"
        "            )  # gfx1151 patch 33\n"
        "            if tool_idx != -1:\n"
        "                reasoning_text = current_text[:tool_idx]\n",
    )
    text = patch(
        text,
        "        think_start_text = self.think_start_token + self.think_start_self_label\n"
        "        if text.startswith(think_start_text):\n"
        "            return text[len(think_start_text) :]\n",
        "        think_start_text = self.think_start_token + self.think_start_self_label\n"
        "        if self._marker_view(text).startswith(think_start_text):  # gfx1151 patch 33\n"
        "            return text[len(think_start_text) :]\n",
    )
    # ReasoningParser facade
    text = patch(
        text,
        "        self.detector = detector_class(**kwargs)\n"
        "\n"
        "    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:\n",
        "        self.detector = detector_class(**kwargs)\n"
        "\n"
        "    def attach_control_stream(self, stream) -> None:\n"
        '        """gfx1151 patch 33: match markers by token id through ``stream``\n'
        '        (a ControlTokenStream fed this request\'s output ids), or None."""\n'
        "        self.detector._control = stream\n"
        "\n"
        "    def held_back_len(self) -> int:\n"
        '        """gfx1151 patch 33: chars of streamed text the detector has consumed\n'
        '        but not yet passed on (a possible partial marker)."""\n'
        "        return len(getattr(self.detector, \"_buffer\", \"\"))\n"
        "\n"
        "    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:\n",
    )
    open(path, "w").write(text)


def patch_base_format_detector(path):
    text = open(path).read()
    text = patch(
        text,
        "class BaseFormatDetector(ABC):\n"
        '    """Base class providing two sets of interfaces: one-time and streaming incremental."""\n'
        "\n"
        "    def __init__(self):\n",
        "class BaseFormatDetector(ABC):\n"
        '    """Base class providing two sets of interfaces: one-time and streaming incremental."""\n'
        "\n"
        "    # gfx1151 patch 33: a ControlTokenStream when markers are matched by\n"
        "    # token id; see _marker_view.\n"
        "    _control = None\n"
        "\n"
        "    def _marker_view(self, text: str) -> str:\n"
        '        """The string searched for markers in ``text``, which ends the text\n'
        "        parsed so far. Same length as ``text``, so its indices slice ``text``;\n"
        '        a marker spelled with ordinary tokens is defused in it."""\n'
        "        if self._control is None:\n"
        "            return text\n"
        "        return self._control.view(text)\n"
        "\n"
        "    def __init__(self):\n",
    )
    open(path, "w").write(text)


def patch_function_call_parser(path):
    text = open(path).read()
    text = patch(
        text,
        "        self.detector = detector\n"
        "        self.tools = tools\n"
        "        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()\n",
        "        self.detector = detector\n"
        "        self.tools = tools\n"
        "        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()\n"
        "\n"
        "    def attach_control_stream(self, stream) -> None:\n"
        '        """gfx1151 patch 33: match markers by token id through ``stream``\n'
        '        (a ControlTokenStream fed this request\'s output ids), or None."""\n'
        "        self.detector._control = stream\n",
    )
    open(path, "w").write(text)


def patch_qwen3_coder_detector(path):
    text = open(path).read()
    text = patch(
        text,
        "    def has_tool_call(self, text: str) -> bool:\n"
        "        return self.tool_call_start_token in text\n",
        "    def has_tool_call(self, text: str) -> bool:\n"
        "        return self.tool_call_start_token in self._marker_view(text)  # gfx1151 patch 33\n",
    )
    text = patch(
        text,
        '        """One-shot parsing for non-streaming scenarios."""\n'
        "        if self.tool_call_start_token not in text:\n"
        "            return StreamingParseResult(normal_text=text)\n"
        "\n"
        "        calls = []\n"
        "        try:\n"
        "            # Simple cleanup of the text to find tool calls\n"
        "            # Note: This is a simplified regex approach consistent with vLLM\n"
        "            raw_tool_calls = self.tool_call_regex.findall(text)\n",
        '        """One-shot parsing for non-streaming scenarios."""\n'
        "        # gfx1151 patch 33: <tool_call> and </tool_call> are located in the\n"
        "        # view, so a spelled-out pair is text; bodies are sliced from the text.\n"
        "        view = self._marker_view(text)\n"
        "        if self.tool_call_start_token not in view:\n"
        "            return StreamingParseResult(normal_text=text)\n"
        "\n"
        "        calls = []\n"
        "        try:\n"
        "            # Simple cleanup of the text to find tool calls\n"
        "            # Note: This is a simplified regex approach consistent with vLLM\n"
        "            raw_tool_calls = [\n"
        "                text[m.start(1) : m.end(1)]\n"
        "                for m in self.tool_call_regex.finditer(view)\n"
        "            ]\n",
    )
    text = patch(
        text,
        "            # Determine normal text (text before the first tool call)\n"
        "            start_idx = text.find(self.tool_call_start_token)\n",
        "            # Determine normal text (text before the first tool call)\n"
        "            start_idx = view.find(self.tool_call_start_token)  # gfx1151 patch 33\n",
    )
    text = patch(
        text,
        "        calls = []\n"
        "        normal_text_chunks = []\n"
        "\n"
        "        while True:\n"
        "            # Working text slice\n"
        "            current_slice = self._buffer[self.parsed_pos :]\n",
        "        calls = []\n"
        "        normal_text_chunks = []\n"
        "        # gfx1151 patch 33: the buffer ends the text parsed so far, so its\n"
        "        # view defuses a spelled-out <tool_call> or </tool_call>.\n"
        "        buffer_view = self._marker_view(self._buffer)\n"
        "\n"
        "        while True:\n"
        "            # Working text slice\n"
        "            current_slice = self._buffer[self.parsed_pos :]\n"
        "            current_view = buffer_view[self.parsed_pos :]\n",
    )
    text = patch(
        text,
        "            if current_slice.startswith(self.tool_call_start_token):\n"
        "                self.parsed_pos += len(self.tool_call_start_token)\n",
        "            if current_view.startswith(self.tool_call_start_token):\n"
        "                self.parsed_pos += len(self.tool_call_start_token)\n",
    )
    text = patch(
        text,
        "            if current_slice.startswith(self.tool_call_end_token):\n"
        "                self.parsed_pos += len(self.tool_call_end_token)\n",
        "            if current_view.startswith(self.tool_call_end_token):\n"
        "                self.parsed_pos += len(self.tool_call_end_token)\n",
    )
    # A complete spelled-out marker at the very end of the buffer must not be
    # held as a potential tag forever; the view tells it from a real one.
    text = patch(
        text,
        "                is_potential_tag = False\n"
        "                for tag in possible_tags:\n"
        "                    if tag.startswith(current_slice):\n"
        "                        is_potential_tag = True\n"
        "                        break\n",
        "                is_potential_tag = False\n"
        "                for tag in possible_tags:\n"
        "                    probe = (\n"
        "                        current_view\n"
        "                        if tag in (self.tool_call_start_token, self.tool_call_end_token)\n"
        "                        else current_slice\n"
        "                    )  # gfx1151 patch 33\n"
        "                    if tag.startswith(probe):\n"
        "                        is_potential_tag = True\n"
        "                        break\n",
    )
    open(path, "w").write(text)


def patch_serving_chat(path):
    text = open(path).read()
    text = patch(
        text,
        "from sglang.srt.utils import control_token_encoder  # gfx1151 patch 32\n",
        "from sglang.srt.utils import control_token_encoder  # gfx1151 patch 32\n"
        "from sglang.srt.parser import control_token_stream  # gfx1151 patch 33\n",
    )
    text = patch(
        text,
        "        if self._control_token_encoder is not None:\n"
        "            logger.info(\n"
        "                \"Chat template marks control tokens; prompts are encoded with \"\n"
        "                \"content and control tokens kept apart (patch 32)\"\n"
        "            )\n",
        "        if self._control_token_encoder is not None:\n"
        "            logger.info(\n"
        "                \"Chat template marks control tokens; prompts are encoded with \"\n"
        "                \"content and control tokens kept apart (patch 32)\"\n"
        "            )\n"
        "        # gfx1151 patch 33: parser markers that are single tokens are matched\n"
        "        # by id in the output, so the same strings spelled out are text.\n"
        "        self._control_token_ids = control_token_stream.resolve_control_ids(\n"
        "            self.tokenizer_manager.tokenizer,\n"
        "            control_token_stream.parser_markers(\n"
        "                self.reasoning_parser, self.tool_call_parser\n"
        "            ),\n"
        "        )\n"
        "        if self._control_token_ids:\n"
        "            logger.info(\n"
        "                \"Reasoning and tool-call parsers match %s by token id (patch 33)\",\n"
        "                sorted(self._control_token_ids.values()),\n"
        "            )\n",
    )
    text = patch(
        text,
        "    def _handle_last_assistant_message(\n",
        "    def _new_control_stream(\n"
        "        self, request: ChatCompletionRequest\n"
        "    ) -> Optional[control_token_stream.ControlTokenStream]:\n"
        "        \"\"\"gfx1151 patch 33: a stream decoding one choice's output ids for\n"
        "        the parsers, or None when no parser marker is a single token.\"\"\"\n"
        "        if not self._control_token_ids:\n"
        "            return None\n"
        "        return control_token_stream.ControlTokenStream(\n"
        "            self.tokenizer_manager.tokenizer.backend_tokenizer,\n"
        "            self._control_token_ids,\n"
        "            skip_special_tokens=request.skip_special_tokens,\n"
        "        )\n"
        "\n"
        "    def _handle_last_assistant_message(\n",
    )
    # streaming: per-choice streams
    text = patch(
        text,
        "        reasoning_tokens: dict[int, int],\n"
        "        completion_tokens: dict[int, int],\n"
        "    ) -> AsyncGenerator[str, None]:\n"
        "        \"\"\"Generate SSE chunks for streaming content.\"\"\"\n"
        "        offset = stream_offsets.get(index, 0)\n"
        "        if get_serving().incremental_streaming_output:\n"
        "            delta = content[\"text\"]\n"
        "        else:\n"
        "            delta = content[\"text\"][offset:]\n"
        "            stream_offsets[index] = len(content[\"text\"])\n",
        "        reasoning_tokens: dict[int, int],\n"
        "        completion_tokens: dict[int, int],\n"
        "        control_streams: dict | None = None,  # gfx1151 patch 33\n"
        "    ) -> AsyncGenerator[str, None]:\n"
        "        \"\"\"Generate SSE chunks for streaming content.\"\"\"\n"
        "        offset = stream_offsets.get(index, 0)\n"
        "        if get_serving().incremental_streaming_output:\n"
        "            delta = content[\"text\"]\n"
        "        else:\n"
        "            delta = content[\"text\"][offset:]\n"
        "            stream_offsets[index] = len(content[\"text\"])\n"
        "\n"
        "        # gfx1151 patch 33: decode this chunk's ids beside its text so the\n"
        "        # parsers below match control tokens by id.\n"
        "        control_stream = None\n"
        "        if control_streams is not None:\n"
        "            if index not in control_streams:\n"
        "                control_streams[index] = self._new_control_stream(request)\n"
        "            control_stream = control_streams[index]\n"
        "            if control_stream is not None:\n"
        "                chunk_ids = content.get(\"output_ids\")\n"
        "                if chunk_ids is not None:\n"
        "                    control_stream.feed(\n"
        "                        chunk_ids,\n"
        "                        cumulative=not get_serving().incremental_streaming_output,\n"
        "                    )\n"
        "                control_stream.consume(delta)\n",
    )
    text = patch(
        text,
        "            reasoning_text, delta = self._process_reasoning_stream(\n"
        "                index,\n"
        "                delta,\n"
        "                reasoning_parser_dict,\n"
        "                content,\n"
        "                request,\n"
        "                finish_reason_type,\n"
        "            )\n",
        "            reasoning_text, delta = self._process_reasoning_stream(\n"
        "                index,\n"
        "                delta,\n"
        "                reasoning_parser_dict,\n"
        "                content,\n"
        "                request,\n"
        "                finish_reason_type,\n"
        "                control_stream=control_stream,  # gfx1151 patch 33\n"
        "            )\n"
        "            if control_stream is not None:\n"
        "                # The tool-call parser's text ends where the reasoning\n"
        "                # parser's held-back partial marker begins.\n"
        "                control_stream.set_tail_lag(\n"
        "                    reasoning_parser_dict[index].held_back_len()\n"
        "                )\n",
    )
    text = patch(
        text,
        "            async for chunk in self._process_tool_call_stream(\n"
        "                index,\n"
        "                delta,\n"
        "                parser_dict,\n"
        "                content,\n"
        "                request,\n"
        "                has_tool_calls,\n"
        "                continuous_usage_stats,\n"
        "                flush=finish_reason_type is not None and finish_reason_type != \"abort\",\n"
        "            ):\n",
        "            async for chunk in self._process_tool_call_stream(\n"
        "                index,\n"
        "                delta,\n"
        "                parser_dict,\n"
        "                content,\n"
        "                request,\n"
        "                has_tool_calls,\n"
        "                continuous_usage_stats,\n"
        "                flush=finish_reason_type is not None and finish_reason_type != \"abort\",\n"
        "                control_stream=control_stream,  # gfx1151 patch 33\n"
        "            ):\n",
    )
    text = patch(
        text,
        "        # Parsers for tool calls and reasoning\n"
        "        parser_dict = {}\n"
        "        reasoning_parser_dict = {}\n",
        "        # Parsers for tool calls and reasoning\n"
        "        parser_dict = {}\n"
        "        reasoning_parser_dict = {}\n"
        "        control_streams: dict[int, Any] = {}  # gfx1151 patch 33\n",
    )
    text = patch(
        text,
        "                    reasoning_tokens=reasoning_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                ):\n"
        "                    yield chunk\n",
        "                    reasoning_tokens=reasoning_tokens,\n"
        "                    completion_tokens=completion_tokens,\n"
        "                    control_streams=control_streams,  # gfx1151 patch 33\n"
        "                ):\n"
        "                    yield chunk\n",
    )
    text = patch(
        text,
        "        request: ChatCompletionRequest,\n"
        "        finish_reason_type: str | None = None,\n"
        "    ) -> tuple[str | None, str]:\n"
        "        \"\"\"Process reasoning content in streaming response\"\"\"\n",
        "        request: ChatCompletionRequest,\n"
        "        finish_reason_type: str | None = None,\n"
        "        control_stream=None,  # gfx1151 patch 33\n"
        "    ) -> tuple[str | None, str]:\n"
        "        \"\"\"Process reasoning content in streaming response\"\"\"\n",
    )
    text = patch(
        text,
        "                tokenizer=self.tokenizer_manager.tokenizer,\n"
        "                tool_call_parser_active=self._tool_call_parsing_active(request),\n"
        "            )\n"
        "        reasoning_parser = reasoning_parser_dict[index]\n",
        "                tokenizer=self.tokenizer_manager.tokenizer,\n"
        "                tool_call_parser_active=self._tool_call_parsing_active(request),\n"
        "            )\n"
        "            reasoning_parser_dict[index].attach_control_stream(control_stream)\n"
        "        reasoning_parser = reasoning_parser_dict[index]\n",
    )
    text = patch(
        text,
        "        continuous_usage_stats: bool = False,\n"
        "        flush: bool = False,\n"
        "    ):\n"
        "        \"\"\"Process tool calls in streaming response.\n",
        "        continuous_usage_stats: bool = False,\n"
        "        flush: bool = False,\n"
        "        control_stream=None,  # gfx1151 patch 33\n"
        "    ):\n"
        "        \"\"\"Process tool calls in streaming response.\n",
    )
    text = patch(
        text,
        "        parser = parser_dict[index]\n"
        "\n"
        "        # Handle both FunctionCallParser and JsonArrayParser\n"
        "        if isinstance(parser, JsonArrayParser):\n",
        "        parser = parser_dict[index]\n"
        "        if isinstance(parser, FunctionCallParser):\n"
        "            parser.attach_control_stream(control_stream)  # gfx1151 patch 33\n"
        "\n"
        "        # Handle both FunctionCallParser and JsonArrayParser\n"
        "        if isinstance(parser, JsonArrayParser):\n",
    )
    # non-streaming
    text = patch(
        text,
        "            text = self._decode_response(ret_item)\n"
        "            if isinstance(text, ErrorResponse):\n"
        "                return ORJSONResponse(content=text.model_dump(), status_code=text.code)\n",
        "            text = self._decode_response(ret_item)\n"
        "            if isinstance(text, ErrorResponse):\n"
        "                return ORJSONResponse(content=text.model_dump(), status_code=text.code)\n"
        "\n"
        "            # gfx1151 patch 33: the parsers match control tokens by id.\n"
        "            control_stream = self._new_control_stream(request)\n"
        "            if control_stream is not None:\n"
        "                choice_output_ids = ret_item.get(\"output_ids\")\n"
        "                if choice_output_ids is not None:\n"
        "                    control_stream.feed(choice_output_ids, cumulative=True)\n"
        "                control_stream.consume(text)\n",
    )
    text = patch(
        text,
        "                        tool_call_parser_active=self._tool_call_parsing_active(request),\n"
        "                    )\n"
        "                    reasoning_text, text = parser.parse_non_stream(text)\n",
        "                        tool_call_parser_active=self._tool_call_parsing_active(request),\n"
        "                    )\n"
        "                    parser.attach_control_stream(control_stream)  # gfx1151 patch 33\n"
        "                    reasoning_text, text = parser.parse_non_stream(text)\n",
    )
    text = patch(
        text,
        "                tool_calls, text, finish_reason = self._process_tool_calls(\n"
        "                    text,\n"
        "                    effective_tools,\n"
        "                    finish_reason,\n"
        "                    request.tool_choice,\n"
        "                    history_tool_calls_cnt,\n"
        "                )\n",
        "                tool_calls, text, finish_reason = self._process_tool_calls(\n"
        "                    text,\n"
        "                    effective_tools,\n"
        "                    finish_reason,\n"
        "                    request.tool_choice,\n"
        "                    history_tool_calls_cnt,\n"
        "                    control_stream=control_stream,  # gfx1151 patch 33\n"
        "                )\n",
    )
    text = patch(
        text,
        "        tool_choice: str | ToolChoice | None = None,\n"
        "        history_tool_calls_cnt: int = 0,\n"
        "    ) -> ToolCallProcessingResult:\n"
        "        \"\"\"Process tool calls in the response\"\"\"\n",
        "        tool_choice: str | ToolChoice | None = None,\n"
        "        history_tool_calls_cnt: int = 0,\n"
        "        control_stream=None,  # gfx1151 patch 33\n"
        "    ) -> ToolCallProcessingResult:\n"
        "        \"\"\"Process tool calls in the response\"\"\"\n",
    )
    text = patch(
        text,
        "            parser = FunctionCallParser(\n"
        "                tools, self.tool_call_parser, tokenizer=self.tokenizer_manager.tokenizer\n"
        "            )\n"
        "            detector_owns_format = (\n",
        "            parser = FunctionCallParser(\n"
        "                tools, self.tool_call_parser, tokenizer=self.tokenizer_manager.tokenizer\n"
        "            )\n"
        "            parser.attach_control_stream(control_stream)  # gfx1151 patch 33\n"
        "            detector_owns_format = (\n",
    )
    open(path, "w").write(text)


def main():
    tree = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    srt = f"{tree}/python/sglang/srt"
    with open(f"{srt}/parser/control_token_stream.py", "w") as f:
        f.write(MODULE_SRC)
    patch_reasoning_parser(f"{srt}/parser/reasoning_parser.py")
    patch_base_format_detector(f"{srt}/function_call/base_format_detector.py")
    patch_function_call_parser(f"{srt}/function_call/function_call_parser.py")
    patch_qwen3_coder_detector(f"{srt}/function_call/qwen3_coder_detector.py")
    patch_serving_chat(f"{srt}/entrypoints/openai/serving_chat.py")
    print(f"patched parsers and serving_chat.py; wrote {srt}/parser/control_token_stream.py")


if __name__ == "__main__":
    main()
