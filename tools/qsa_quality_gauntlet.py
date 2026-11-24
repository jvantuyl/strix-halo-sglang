#!/usr/bin/env python3
"""Paired, objective QSA quantization smoke tests against an SGLang server."""
import argparse
import json
import math
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean


def request(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def cases():
    rng = random.Random(71403)
    for i in range(8):
        a, b, c = rng.sample(range(12, 87), 3)
        yield f"arithmetic-{i}", "reasoning", f"What is {a} × {b} minus {c}? Reply with only the integer.", a*b-c
    for i in range(8):
        values = rng.sample(range(3, 60), 7)
        prompt = (
            f"What does this Python expression return? Reply with only the integer.\n"
            f"sum((i+1)*v for i,v in enumerate({values!r}[::-2]) if v % 3 != 0)"
        )
        result = sum((i+1)*v for i,v in enumerate(values[::-2]) if v % 3 != 0)
        yield f"code-{i}", "code", prompt, result
    for i in range(8):
        left, right = rng.sample(range(10, 999), 2)
        yield f"json-{i}", "structured", (
            f"Station Birch has {left} crates. Station Alder has {right}. "
            "Return only a JSON object with keys birch and alder and integer counts."
        ), {"birch": left, "alder": right}
    for i in range(8):
        a, b = rng.sample(range(100, 999), 2)
        yield f"reading-{i}", "reading", (
            f"The inventory listed {a} disks for Rami and {b} for Neve. "
            "After noticing that the names, but not the counts, had been swapped, "
            "the clerk corrected the report. How many disks belong to Rami now? Only the integer."
        ), b
    for n in (320, 1280):
        for frac in (0.15, 0.75):
            code = f"V{rng.randrange(10000, 99999)}"
            rows = [f"Entry {i:04}: depot {rng.randrange(100,999)}, lamp {rng.randrange(100,999)}, status pending."
                    for i in range(n)]
            rows.insert(int(n*frac), f"Entry X: station ORIOLE, access identifier {code}, status verified.")
            yield f"needle-{n}-{frac}", "retrieval", (
                "\n".join(rows) + "\nWhat is the access identifier for ORIOLE? Reply with only the identifier."
            ), code


PASSAGES = {
    "prose": (
        "At the end of the pier stood a wooden cabinet with a brass latch.",
        " Mara opened it and found a dry map folded beneath a stack of damp newspapers. The map marked a footpath along the eastern shore, but there was no sign of a bridge where the path crossed the river. She copied the route into her notebook and left the cabinet as she had found it."
    ),
    "technical": (
        "A durable queue treats an acknowledged write as a promise to recover the record after a crash.",
        " One implementation appends each record to a log and flushes the log before acknowledging the request. A separate index can be reconstructed by scanning the log. If the last record is incomplete, recovery discards it instead of treating its partial contents as a valid message."
    ),
    "code": (
        "def count_words(lines):\n    counts = {}\n    for line in lines:\n",
        "        for word in line.split():\n            word = word.lower()\n            counts[word] = counts.get(word, 0) + 1\n    return counts\n"
    ),
    "structured": (
        '{"source":"station-oriole","readings":[',
        '{"minute":0,"temperature":17.2,"valid":true},{"minute":5,"temperature":17.8,"valid":true},{"minute":10,"temperature":null,"valid":false}],"units":"C"}'
    ),
}


def score(text, expected):
    text = text.strip()
    if isinstance(expected, dict):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        return (isinstance(parsed, dict) and parsed == expected
                and all(type(v) is int for v in parsed.values()))
    if isinstance(expected, int):
        return bool(re.fullmatch(r"-?[0-9]+", text)) and int(text) == expected
    return text == expected


def tokenize(base, **kwargs):
    return request(base, "/tokenize", kwargs)["tokens"]


def generate_case(base, case):
    name, group, prompt, expected = case
    ids = tokenize(base, messages=[{"role": "user", "content": prompt}],
                   chat_template_kwargs={"enable_thinking": False})
    result = request(base, "/generate", {
        "input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 64}
    })
    text = result["text"]
    return dict(kind="task", name=name, group=group, expected=expected, text=text,
                correct=score(text, expected), input_tokens=len(ids))


def logprob_case(base, name, prefix, suffix):
    ids = tokenize(base, prompt=prefix+suffix, add_special_tokens=False)
    start_ids = tokenize(base, prompt=prefix, add_special_tokens=False)
    start = next((i for i, (a,b) in enumerate(zip(ids, start_ids)) if a != b), len(start_ids))
    assert 0 < start < len(ids)
    result = request(base, "/generate", {
        "input_ids": ids, "return_logprob": True, "logprob_start_len": start - 1,
        "top_logprobs_num": 1, "sampling_params": {"temperature": 0, "max_new_tokens": 0}
    })
    meta = result["meta_info"]
    # SGLang returns a null sentinel for the first requested position.
    rows = meta["input_token_logprobs"][1:]
    assert [x[1] for x in rows] == ids[start:] and all(math.isfinite(x[0]) for x in rows)
    tops = meta["input_top_logprobs"][1:]
    assert len(tops) == len(rows) and all(x and math.isfinite(x[0][0]) for x in tops)
    return dict(kind="logprobs", name=name, input_tokens=len(ids), token_ids=ids[start:],
                logprobs=[x[0] for x in rows], top1=[x[0][1] for x in tops])


def run(base, label, out, smoke=False):
    with out.open("x") as f:
        task_cases = list(cases())
        selected = task_cases[:1] if smoke else task_cases
        for row in (generate_case(base, case) for case in selected):
            print(row["name"], row["input_tokens"], row["correct"], flush=True)
            f.write(json.dumps(dict(variant=label, **row)) + "\n")
            f.flush()
        for name, (prefix, suffix) in PASSAGES.items():
            row = logprob_case(base, name, prefix, suffix)
            print(name, "scored", len(row["token_ids"]), "tokens", flush=True)
            f.write(json.dumps(dict(variant=label, **row)) + "\n")
            f.flush()


def compare(baseline, candidate):
    a, b = ({r["name"]: r for r in map(json.loads, p.read_text().splitlines())}
            for p in (baseline, candidate))
    if a.keys() != b.keys() or len(a) != len(list(cases())) + len(PASSAGES):
        raise ValueError("Incomplete or unmatched paired gauntlet")
    groups = {}
    log_a, log_b, top_agree = [], [], []
    for name, x in a.items():
        y = b[name]
        if x["kind"] != y["kind"] or x["input_tokens"] != y["input_tokens"]:
            raise ValueError(f"Different request: {name}")
        if x["kind"] == "task":
            assert x["expected"] == y["expected"]
            c = groups.setdefault(x["group"], Counter())
            c["total"] += 1
            c["baseline"] += x["correct"]
            c["candidate"] += y["correct"]
            c["lost"] += x["correct"] and not y["correct"]
            c["gained"] += y["correct"] and not x["correct"]
            c["same_text"] += x["text"] == y["text"]
        else:
            assert x["token_ids"] == y["token_ids"]
            log_a.extend(x["logprobs"])
            log_b.extend(y["logprobs"])
            top_agree.extend(a == b for a,b in zip(x["top1"], y["top1"]))
    return dict(groups=groups, scored_tokens=len(log_a), baseline_nll=-mean(log_a),
                candidate_nll=-mean(log_b), delta_nll=mean(a-b for a,b in zip(log_a,log_b)),
                top1_agreement=mean(top_agree))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    collect = sub.add_parser("run")
    collect.add_argument("--base", default="http://127.0.0.1:8000")
    collect.add_argument("--label", required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--smoke", action="store_true")
    paired = sub.add_parser("compare")
    paired.add_argument("baseline", type=Path)
    paired.add_argument("candidate", type=Path)
    args = ap.parse_args()
    if args.action == "run":
        run(args.base, args.label, args.out, args.smoke)
    else:
        print(json.dumps(compare(args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()
