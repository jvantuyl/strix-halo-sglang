#!/usr/bin/env python3
"""Patch 60: backport SGLang #40554's two GigaChat 3.5 CLI choices."""

import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    srt = root / "python/sglang/srt"
    for filename, before, after in (
        ("function_call/parser_names.py", '    "gigachat3",\n', '    "gemma4",\n'),
        ("parser/reasoning_parser_names.py", '    "cohere_command4",\n', "]\n"),
    ):
        path = srt / filename
        source = path.read_text()
        entry = '    "gigachat35",\n'
        assert entry not in source and source.count(before + after) == 1, filename
        path.write_text(source.replace(before + after, before + entry + after))


if __name__ == "__main__":
    main()
