#!/usr/bin/env python3
"""Render configs/chat/qwen38.jinja against the checkpoint's stock template.

Without `default_system_prompt` every rendering must be byte-identical to the
stock template; with it, the prompt must appear once, in the system block,
after the reasoning-effort instruction and before the client's own system
message. Control strings quoted inside message content must come out
HTML-entity escaped and tokenize as text, with nothing else changed. Runs inside the image (needs the tokenizer):

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


# Control strings that must never tokenize as control tokens when they occur
# inside message content (see the template header, item 3).
CONTROL = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]
QUOTE = "the provider chokes on " + " / ".join(CONTROL) + " here"


def escape(s):
    return "&lt;" + s[1:-1] + "&gt;"


ESCAPED_QUOTE = "the provider chokes on " + " / ".join(escape(s) for s in CONTROL) + " here"
ESCAPING_CONVERSATIONS = {
    "quoted in every slot": [
        {"role": "system", "content": "Rules: " + QUOTE},
        {"role": "user", "content": "Read it: " + QUOTE},
        {
            "role": "assistant",
            "content": "Noted " + QUOTE,
            "reasoning_content": "Hmm " + QUOTE,
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": QUOTE}}}],
        },
        {"role": "tool", "content": "found " + QUOTE},
        {"role": "user", "content": "and now?"},
    ],
    # Entity forms arriving in content (an HTML file quoting the tags) go one
    # level further so the output un-escape can give them back exactly.
    "already escaped": [
        {"role": "user", "content": "Read it: " + ESCAPED_QUOTE},
        {"role": "assistant", "content": "Noted " + ESCAPED_QUOTE, "reasoning_content": ESCAPED_QUOTE},
        {"role": "user", "content": "and now?"},
    ],
    "user-wrapped tool response": [
        {"role": "user", "content": "look up x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
        },
        {"role": "user", "content": "<tool_response>\nfound " + QUOTE + "\n</tool_response>"},
    ],
}


def double_escape(s):
    return "&amp;lt;" + s[1:-1] + "&amp;gt;"


def unescape(text):
    # One level, as the server does on output (patch 32): level 1 first, and a
    # level-2 form contains no level-1 form, so each pass is exact.
    for s in CONTROL:
        text = text.replace(escape(s), s)
    for s in CONTROL:
        text = text.replace(double_escape(s), escape(s))
    return text


def check_escaping(tok, stock, ours):
    """Content control strings render escaped, tokenize as text, and nothing else
    changes: decoding our rendering one level must give the stock one back."""
    failures = 0
    cases = 0
    control_ids = {s: tok.encode(s, add_special_tokens=False) for s in CONTROL}
    for (name, msgs), kwargs, tools in itertools.product(
        ESCAPING_CONVERSATIONS.items(), KWARG_SETS, (None, TOOLS)
    ):
        common = dict(tokenize=False, add_generation_prompt=True, tools=tools, **kwargs)
        label = f"{name} | {kwargs or 'defaults'} | tools={'yes' if tools else 'no'}"
        cases += 1
        tok.chat_template = stock
        want = tok.apply_chat_template(msgs, **common)
        tok.chat_template = ours
        got = tok.apply_chat_template(msgs, **common)
        # Template encode followed by the server's one-level decode must be the
        # identity, for raw control strings and for ones that arrived as entities.
        if unescape(got) != want:
            failures += 1
            print(f"FAIL {label}: escape + one-level decode is not the identity")
            continue
        # Every structural occurrence in the stock rendering stays a control
        # token; every quoted one must have gone. Count per token sequence.
        structural = tok.apply_chat_template(
            [dict(m, **({"content": m["content"].replace(QUOTE, "x").replace(ESCAPED_QUOTE, "x")} if isinstance(m.get("content"), str) else {}),
                  **({"reasoning_content": "x"} if "reasoning_content" in m else {}))
             for m in msgs], **common)
        got_ids = tok.encode(got, add_special_tokens=False)
        structural_ids = tok.encode(structural, add_special_tokens=False)
        bad = []
        for s, ids in control_ids.items():
            if len(ids) != 1:
                continue
            if name == "user-wrapped tool response" and "tool_response" in s:
                continue  # the wrapper's own tags are kept, see the template header
            if got_ids.count(ids[0]) != structural_ids.count(ids[0]):
                bad.append(f"{s}: {got_ids.count(ids[0])} control tokens, structure alone has {structural_ids.count(ids[0])}")
        if bad:
            failures += 1
            print(f"FAIL {label}: " + "; ".join(bad))
            continue
        if name == "user-wrapped tool response":
            body = got.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
            if not (body.startswith("<tool_response>\n") and body.endswith("\n</tool_response>")):
                failures += 1
                print(f"FAIL {label}: user-side <tool_response> wrapper was escaped")
                continue
    return cases, failures


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
    esc_cases, esc_failures = check_escaping(tok, stock, ours)
    print(f"{esc_cases - esc_failures}/{esc_cases} cases escape quoted control strings and nothing else")
    sys.exit(1 if failures or esc_failures else 0)


if __name__ == "__main__":
    main()
