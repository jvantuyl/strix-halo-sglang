#!/usr/bin/env python3
"""Render configs/chat/qwen38.jinja against the checkpoint's stock template.

Without `default_system_prompt` every rendering must be byte-identical to the
stock template; with it, the prompt must appear once, in the system block,
after the reasoning-effort instruction and before the client's own system
message. Runs inside the image (needs the tokenizer):

    docker run --rm -v <model>:/models/qwen38:ro -v <repo>:/repo:ro \
        strix-halo-sglang:dev python3 /repo/tests/check_chat_template.py
"""

import itertools
import os
import sys

from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL_DIR", "/models/qwen38")
TEMPLATE = os.environ.get(
    "CHAT_TEMPLATE", os.path.join(os.path.dirname(__file__), "..", "configs", "chat", "qwen38.jinja")
)
PROMPT = "If you are unsure or do not know something, say so plainly instead of guessing."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
CONVERSATIONS = {
    "user only": [{"role": "user", "content": "hi"}],
    "system + user": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ],
    "empty system": [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
    "multi-turn with reasoning": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "greet"},
        {"role": "user", "content": "again"},
    ],
    "tool call round trip": [
        {"role": "user", "content": "look up x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
        },
        {"role": "tool", "content": "found x"},
    ],
}
KWARG_SETS = [
    {},
    {"enable_thinking": False},
    {"reasoning_effort": "medium"},
    {"reasoning_effort": "low"},
    {"preserve_thinking": False},
]


def load(path):
    # Same transformation as sglang's TemplateManager._load_jinja_template.
    with open(path) as f:
        return "".join(f.readlines()).strip("\n").replace("\\n", "\n")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    stock = tok.chat_template
    ours = load(TEMPLATE)
    failures = 0
    cases = 0
    for (name, msgs), kwargs, tools in itertools.product(
        CONVERSATIONS.items(), KWARG_SETS, (None, TOOLS)
    ):
        common = dict(tokenize=False, add_generation_prompt=True, tools=tools, **kwargs)
        tok.chat_template = stock
        want = tok.apply_chat_template(msgs, **common)
        tok.chat_template = ours
        got = tok.apply_chat_template(msgs, **common)
        got_empty = tok.apply_chat_template(msgs, default_system_prompt="", **common)
        got_prompt = tok.apply_chat_template(msgs, default_system_prompt=PROMPT, **common)
        label = f"{name} | {kwargs or 'defaults'} | tools={'yes' if tools else 'no'}"
        cases += 1
        if got != want or got_empty != want:
            failures += 1
            print(f"FAIL {label}: differs from stock without a prompt")
            continue
        if got_prompt.count(PROMPT) != 1:
            failures += 1
            print(f"FAIL {label}: prompt appears {got_prompt.count(PROMPT)} times")
            continue
        head, _, _ = got_prompt.partition("<|im_start|>user")
        if not head.startswith("<|im_start|>system\n") or "<|im_end|>" not in head:
            failures += 1
            print(f"FAIL {label}: prompt not inside the leading system block")
            continue
        sys_block = head.split("<|im_end|>")[0]
        p = sys_block.index(PROMPT)
        if "Reasoning effort" in sys_block and sys_block.index("Reasoning effort") > p:
            failures += 1
            print(f"FAIL {label}: prompt precedes the reasoning instruction")
            continue
        if "You are terse." in sys_block and sys_block.index("You are terse.") < p:
            failures += 1
            print(f"FAIL {label}: client system message precedes the prompt")
            continue
        if tools and sys_block.index("# Tools") < p:
            failures += 1
            print(f"FAIL {label}: tool list precedes the prompt")
            continue
        # Removing the prompt paragraph must give the stock rendering back: the
        # prompt is either a whole system block of its own, or one paragraph
        # of a larger one (followed or preceded by a blank line).
        alone = "<|im_start|>system\n" + PROMPT + "<|im_end|>\n"
        if alone in got_prompt:
            stripped = got_prompt.replace(alone, "", 1)
        elif PROMPT + "\n\n" in got_prompt:
            stripped = got_prompt.replace(PROMPT + "\n\n", "", 1)
        else:
            stripped = got_prompt.replace("\n\n" + PROMPT, "", 1)
        if stripped != want:
            failures += 1
            print(f"FAIL {label}: prompt changes more than its own paragraph")
            continue
    tok.chat_template = ours
    sample = tok.apply_chat_template(
        CONVERSATIONS["system + user"],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="medium",
        default_system_prompt=PROMPT,
    )
    print("sample (medium effort, prompt, client system message):")
    print(sample)
    print(f"{cases - failures}/{cases} cases identical to stock modulo the prompt paragraph")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
