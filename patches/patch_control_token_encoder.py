#!/usr/bin/env python3
"""Encode chat prompts so only the template's control tokens are control
token ids (patch 32).

SGLang renders the chat template to one string and tokenizes it with the
added tokens enabled. Anything in that string that spells a control token
becomes the control token: a tool result quoting "<|im_end|>" is tokenized
as the real end-of-turn id, a quoted "</think>" as the real end-of-thinking
id. With --enable-strict-thinking the sampler then masks those ids and the
model's own turn structure falls apart; without it the reasoning parser ends
the thinking block at the quote. The model never sees the quoted text at
all.

The fix keeps structure and content apart at the tokenizer. The repo chat
template wraps every control token it emits in a pair of private-use
sentinels (U+E000 name U+E001; see configs/chat/qwen38.jinja). The encoder
installed here tokenizes the marked names as their added-token ids and
everything in between with a copy of the tokenizer whose added-token table
is empty, so a quoted "<|im_end|>" is the ordinary pieces "<", "|", "im",
"_end", "|", ">" and the model reads exactly the text the client sent.
Request strings are escaped before rendering (U+E000 -> U+E000 U+E000, same
for U+E001) and the encoder unescapes them, so no request content can
produce a control token id. For prompts whose content never spells a control
token the ids are identical to today's, so prefix caching is unaffected.

Hooks in serving_chat.py: the encoder is built in __init__ from the loaded
chat template (None when the template carries no marks, in which case
nothing changes), _render_and_encode_chat_template escapes, renders and
split-encodes, _append_assistant_prefix_to_prompt_ids encodes the
continue_final_message prefix as plain text, and _engine_prompt sends the
ids rather than the text so the tokenizer manager does not re-tokenize
(this model reports multimodal, which otherwise routes the rendered text
back through the tokenizer). Anchors are asserted so an upstream rewrite
fails the build. See patches/32-control-token-encoder.md.
"""
import sys

MODULE_SRC = '''\
"""Encode a rendered chat prompt so that only the chat template\'s control
tokens become control token ids.

gfx1151 patch 32. The chat template wraps each control token it emits in
MARK_START name MARK_END (private-use code points). Everything outside the
marks is content and is tokenized with a copy of the tokenizer that has no
added tokens, so a quoted "<|im_end|>" stays ordinary text. Sentinel code
points inside request strings are escaped by doubling before the template
runs and folded back here, so no request string can forge a mark.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MARK_START = "\\ue000"
MARK_END = "\\ue001"


def escape_text(text: str) -> str:
    """Double the sentinel code points so the encoder reads them as content."""
    if MARK_START in text or MARK_END in text:
        text = text.replace(MARK_START, MARK_START + MARK_START)
        text = text.replace(MARK_END, MARK_END + MARK_END)
    return text


def escape_request(obj: Any) -> Any:
    """escape_text over every string in a nested dict/list structure."""
    if isinstance(obj, str):
        return escape_text(obj)
    if isinstance(obj, dict):
        return {k: escape_request(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [escape_request(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(escape_request(v) for v in obj)
    return obj


class ControlTokenEncoder:
    """Split encoder for a chat template that marks its control tokens."""

    def __init__(self, control_ids: Dict[str, int], plain_tokenizer):
        if not control_ids:
            raise ValueError("tokenizer has no added tokens to mark")
        self.control_ids = dict(control_ids)
        self._plain = plain_tokenizer
        names = "|".join(
            re.escape(name) for name in sorted(control_ids, key=len, reverse=True)
        )
        # Doubled sentinels are escaped content; a mark must name a known
        # control token. Any other sentinel is ordinary text.
        self._scan = re.compile(
            MARK_START + MARK_START + "|" + MARK_END + MARK_END + "|"
            + MARK_START + "(" + names + ")" + MARK_END
        )

    @classmethod
    def from_tokenizer(cls, tokenizer) -> Optional["ControlTokenEncoder"]:
        """The encoder for this tokenizer\'s chat template, or None when the
        template does not mark its control tokens."""
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str):
            return None
        # The marks are written as Jinja string escapes in the template source.
        template = template.replace("\\\\ue000", MARK_START).replace(
            "\\\\ue001", MARK_END
        )
        if MARK_START not in template:
            return None
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "the chat template marks control tokens, which needs a fast "
                "(tokenizers-backed) tokenizer"
            )
        from tokenizers import Tokenizer

        spec = json.loads(backend.to_str())
        spec["added_tokens"] = []
        plain = Tokenizer.from_str(json.dumps(spec))
        control_ids = dict(tokenizer.get_added_vocab())
        encoder = cls(control_ids, plain)
        for name in cls.marked_names(template):
            if name not in control_ids:
                raise ValueError(
                    f"the chat template marks {name!r} as a control token but the "
                    "tokenizer has no such added token"
                )
        return encoder

    @staticmethod
    def marked_names(template: str) -> List[str]:
        """The control token names a (rendered or source) template marks."""
        return sorted(
            set(
                re.findall(
                    MARK_START + "([^\\\\s" + MARK_START + MARK_END + "]+)" + MARK_END,
                    template,
                )
            )
        )

    def encode_plain(self, text: str) -> List[int]:
        """Tokenize text as content: no added token is ever produced."""
        if not text:
            return []
        return self._plain.encode(text, add_special_tokens=False).ids

    def encode(self, marked: str) -> List[int]:
        """Tokenize a rendered, marked prompt."""
        ids: List[int] = []
        plain: List[str] = []
        pos = 0
        for m in self._scan.finditer(marked):
            plain.append(marked[pos : m.start()])
            pos = m.end()
            name = m.group(1)
            if name is None:
                plain.append(m.group(0)[0])
                continue
            ids.extend(self.encode_plain("".join(plain)))
            plain.clear()
            ids.append(self.control_ids[name])
        plain.append(marked[pos:])
        ids.extend(self.encode_plain("".join(plain)))
        return ids

    @staticmethod
    def strip_marks(marked: str) -> str:
        """The rendered text as the model reads it."""
        out = []
        pos = 0
        for m in re.finditer(
            MARK_START + MARK_START + "|" + MARK_END + MARK_END + "|" + MARK_START + "|" + MARK_END,
            marked,
        ):
            out.append(marked[pos : m.start()])
            pos = m.end()
            if len(m.group(0)) == 2:
                out.append(m.group(0)[0])
        out.append(marked[pos:])
        return "".join(out)
'''


def patch(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old)
    return text.replace(old, new)


def main():
    tree = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    srt = f"{tree}/python/sglang/srt"

    with open(f"{srt}/utils/control_token_encoder.py", "w") as f:
        f.write(MODULE_SRC)

    path = f"{srt}/entrypoints/openai/serving_chat.py"
    text = open(path).read()

    text = patch(
        text,
        "from sglang.srt.utils import ImageData\n",
        "from sglang.srt.utils import ImageData\n"
        "from sglang.srt.utils import control_token_encoder  # gfx1151 patch 32\n",
    )

    text = patch(
        text,
        "        self._prompt_text_round_trip_is_lossy = self._probe_prompt_text_round_trip()\n",
        "        self._prompt_text_round_trip_is_lossy = self._probe_prompt_text_round_trip()\n"
        "        # gfx1151 patch 32: a chat template that marks its control tokens is\n"
        "        # encoded piecewise, so content can never become a control token.\n"
        "        self._control_token_encoder = (\n"
        "            control_token_encoder.ControlTokenEncoder.from_tokenizer(\n"
        "                self.tokenizer_manager.tokenizer\n"
        "            )\n"
        "        )\n"
        "        if self._control_token_encoder is not None:\n"
        "            logger.info(\n"
        "                \"Chat template marks control tokens; prompts are encoded with \"\n"
        "                \"content and control tokens kept apart (patch 32)\"\n"
        "            )\n",
    )

    text = patch(
        text,
        "        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)\n"
        "        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:\n",
        "        if self._control_token_encoder is not None:  # gfx1151 patch 32\n"
        "            return prompt_ids + self._control_token_encoder.encode_plain(\n"
        "                assistant_prefix\n"
        "            )\n"
        "        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)\n"
        "        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:\n",
    )

    text = patch(
        text,
        "        prefers_prompt_ids = (\n"
        "            self._prompt_text_round_trip_is_lossy\n"
        "            and isinstance(processed_messages.prompt_ids, list)\n",
        "        prefers_prompt_ids = (\n"
        "            (\n"
        "                self._prompt_text_round_trip_is_lossy\n"
        "                or self._control_token_encoder is not None  # gfx1151 patch 32\n"
        "            )\n"
        "            and isinstance(processed_messages.prompt_ids, list)\n",
    )
    text = patch(
        text,
        "        if (\n"
        "            is_multimodal\n"
        "            and not chat_encoding.spec_renders_prompt_ids(self.chat_encoding_spec)\n"
        "            and not prefers_prompt_ids\n"
        "        ):\n"
        "            return \"text\", processed_messages.prompt\n",
        "        if (\n"
        "            is_multimodal\n"
        "            and not chat_encoding.spec_renders_prompt_ids(self.chat_encoding_spec)\n"
        "            and not prefers_prompt_ids\n"
        "        ):\n"
        "            if self._control_token_encoder is not None:  # gfx1151 patch 32\n"
        "                # The MM processor tokenizes the text itself, with the\n"
        "                # added tokens enabled; quoted control strings in this\n"
        "                # request's content become control tokens again.\n"
        "                logger.warning(\n"
        "                    \"Request carries media; the prompt text is re-tokenized \"\n"
        "                    \"and quoted control strings are not kept apart\"\n"
        "                )\n"
        "            return \"text\", processed_messages.prompt\n",
    )

    text = patch(
        text,
        "        if self._prompt_text_round_trip_is_lossy:\n"
        "            # Re-encoding rendered text would drop the template's control tokens.\n"
        "            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(\n",
        "        if self._control_token_encoder is not None:\n"
        "            # gfx1151 patch 32: the template marks the control tokens it\n"
        "            # emits; request strings are escaped first so none of them can\n"
        "            # forge a mark, then the marked render is encoded piecewise.\n"
        "            encoder = self._control_token_encoder\n"
        "            rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(\n"
        "                control_token_encoder.escape_request(messages),\n"
        "                tokenize=False,\n"
        "                add_generation_prompt=True,\n"
        "                tools=control_token_encoder.escape_request(tools),\n"
        "                return_dict=False,\n"
        "                **control_token_encoder.escape_request(template_kwargs),\n"
        "            )\n"
        "            prompt_ids = encoder.encode(rendered_prompt)\n"
        "        elif self._prompt_text_round_trip_is_lossy:\n"
        "            # Re-encoding rendered text would drop the template's control tokens.\n"
        "            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(\n",
    )

    with open(path, "w") as f:
        f.write(text)
    print(f"patched {path} and wrote {srt}/utils/control_token_encoder.py")


if __name__ == "__main__":
    main()
