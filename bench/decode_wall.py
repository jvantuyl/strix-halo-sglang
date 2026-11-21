"""Single-stream decode wall benchmark for the qwen38 SGLang server.

Canonical probe for the performance ledger. Two modes:

* default: one fixed short prompt (Suez Canal analysis), the shape of the
  historical "220-token wall" numbers.
* --context-tokens N --corpus FILE: builds an ~N-token prompt from real
  corpus text (assistant preference pairs), appends a fixed comprehension
  task, and measures the long prefill (TTFT) and the long generation
  (decode wall and effective tok/s) separately. This is the shape the
  decode work is judged on: a few tens of thousands of prefill tokens,
  then ~1000 generated tokens.

Usage:
    python3 bench/decode_wall.py [--url http://localhost:8000] \
        [--max-tokens 220] [--runs 3]
    python3 bench/decode_wall.py --context-tokens 30000 --max-tokens 1000 \
        --corpus ~/compiled_corpus.jsonl --runs 2
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

PROMPT = (
    "Explain how the Suez Canal changed global trade. Cover: the volume of "
    "traffic it carries today as a share of world maritime trade, the 2021 "
    "Ever Given grounding and what it revealed about supply-chain fragility, "
    "the alternate Cape of Good Hope route and its added cost and time, and "
    "how the 2023-2024 Red Sea security situation shifted routing again. "
    "Be specific with numbers where you can, and keep the focus on trade "
    "economics rather than politics."
)

LONG_TASK = (
    "\n\n---\n\nThe text above is a corpus of assistant-response preference "
    "pairs. After reading it, do the following, in order:\n"
    "1. State the five most common failure patterns in the rejected "
    "responses, with a short quoted fragment as evidence for each (at "
    "least 150 words per pattern).\n"
    "2. Write two new preference pairs of your own in the same style "
    "(prompt, chosen, rejected), on the topics of benchmarking LLM "
    "servers and of NVMe I/O paths. At least 250 words each.\n"
    "3. Estimate what fraction of the corpus concerns coding tasks, with "
    "your reasoning (at least 200 words)."
)

LONG_SYSTEM = (
    "You are a careful analyst. Read the entire provided text before "
    "answering. Answer in English."
)


def build_paragraph_prompt(corpus_path, n_paragraphs, shuffle_seed):
    """Paragraph-shuffle probe: same paragraph multiset, randomized order.

    The radix cache kills any repeated prompt, and fully different text
    changes the n-gram (PLE row) working set. Shuffling a fixed set of
    independent paragraphs defeats the radix prefix (the order differs
    from the very first paragraph) while keeping the overall n-gram
    distribution identical across runs — so run-to-run TTFT deltas
    measure PLE row-cache warmth, not content differences.

    Paragraph selection is deterministic (selection seed 0); only the
    order varies with shuffle_seed.
    """
    rng = random.Random(0)
    with open(corpus_path) as f:
        lines = f.readlines()
    rng.shuffle(lines)
    paragraphs = []
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = d.get("chosen", "").strip()
        # Independent significant paragraphs: long enough to carry real
        # n-gram content, short enough to be one coherent unit.
        if 400 <= len(text) <= 4000:
            paragraphs.append(text)
        if len(paragraphs) >= n_paragraphs:
            break
    if len(paragraphs) < n_paragraphs:
        raise SystemExit(f"corpus only yielded {len(paragraphs)} paragraphs")
    random.Random(shuffle_seed).shuffle(paragraphs)
    return "\n\n".join(paragraphs) + LONG_TASK


def build_long_prompt(corpus_path, target_tokens, seed=0):
    """Concatenate corpus 'prompt'/'chosen' fields up to ~target_tokens."""
    rng = random.Random(seed)
    pieces = []
    total = 0
    # ~4.2 chars per token target; the task adds a fixed tail.
    target_chars = int(target_tokens * 4.2)
    with open(corpus_path) as f:
        lines = f.readlines()
    rng.shuffle(lines)
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = (d.get("prompt", "") + "\n" + d.get("chosen", "")).strip()
        if not text:
            continue
        pieces.append(text)
        total += len(text) + 2
        if total >= target_chars:
            break
    return "\n\n".join(pieces) + LONG_TASK


def run_once(url, model, prompt, max_tokens, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    tokens = 0
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            usage = d.get("usage") or {}
            if usage.get("completion_tokens"):
                tokens = usage["completion_tokens"]
            if usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
            elif d.get("choices"):
                # One token per choices-carrying delta chunk in SGLang
                # streaming; the usage chunk (if any) overrides above.
                tokens += 1
                if ttft is None:
                    delta = d["choices"][0].get("delta", {})
                    if delta.get("reasoning_content") or delta.get("content"):
                        ttft = time.time() - t0
    wall = time.time() - t0
    return wall, ttft, tokens, prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen38-flash-next")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--context-tokens", type=int, default=None,
                    help="build an ~N-token prompt from --corpus")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--paragraphs", type=int, default=None,
                    help="paragraph-shuffle mode: N independent paragraphs, "
                         "same multiset every run, order shuffled per run "
                         "(defeats the radix cache, keeps the PLE row "
                         "working set identical)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-file", default=None,
                    help="override the built-in short prompt")
    args = ap.parse_args()

    if args.paragraphs:
        if not args.corpus:
            ap.error("--paragraphs requires --corpus")
        mode = "paragraphs"
        n_par = args.paragraphs
        system = LONG_SYSTEM
    elif args.context_tokens:
        if not args.corpus:
            ap.error("--context-tokens requires --corpus")
        mode = "chars"
        system = LONG_SYSTEM
    else:
        mode = None
    system = LONG_SYSTEM if mode else None
    url = args.url.rstrip("/") + "/v1/chat/completions"
    results = []
    for i in range(args.runs):
        if mode == "paragraphs":
            prompt = build_paragraph_prompt(args.corpus, n_par, args.seed + i)
        elif mode == "chars":
            # Vary the seed per run: an identical repeat prompt hits the
            # radix prefix cache and the TTFT stops measuring prefill.
            prompt = build_long_prompt(args.corpus, args.context_tokens,
                                       args.seed + i)
        else:
            prompt = PROMPT
        wall, ttft, tokens, prompt_tokens = run_once(
            url, args.model, prompt, args.max_tokens, system)
        results.append((wall, ttft, tokens))
        gen_tok = max(tokens - 0, 1)
        extra = ""
        if prompt_tokens and ttft:
            extra = f"  prefill {prompt_tokens / ttft:6.0f} tok/s ({prompt_tokens} tok)"
        print(
            f"run {i + 1}: wall {wall:7.2f} s  ttft {ttft:6.2f} s  "
            f"gen {tokens:5d} tok  decode {gen_tok / max(wall - ttft, 1e-9):6.2f} tok/s"
            + extra
        )
    walls = [w for w, _, _ in results]
    ttfts = [t for _, t, _ in results]
    toks = [t for _, _, t in results]
    print(
        f"\nmedian: ttft {statistics.median(ttfts):.2f} s  "
        f"gen {statistics.median(toks):.0f} tok  "
        f"decode {statistics.median(toks) / (statistics.median(walls) - statistics.median(ttfts)):.2f} tok/s  "
        f"wall {statistics.median(walls):.2f} s"
    )


if __name__ == "__main__":
    main()
