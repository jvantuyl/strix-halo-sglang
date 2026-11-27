# 60: Accept `gigachat35` in the parser CLI choices

The pinned tree already registers the GigaChat 3.5 tool-call and reasoning
detectors. When those modules have not yet been imported, `ServerArgs` uses
lightweight lists of parser names, and neither list included `gigachat35`.
That made a valid flag value fail depending on import order.

The [patch script](patch_gigachat35_parser_names.py) backports both one-line
additions from [upstream #40554](https://github.com/sgl-project/sglang/pull/40554).
It adds no import, changes no detector and leaves the Qwen settings untouched.

In a CPU-only copy of the patched ROCm image, run these checks from
`/sgl-workspace/sglang` with the fork mounted at `/repo`. The `torch.version.hip`
override only avoids the import-time hardware probe in this test process.

```sh
python3 -c 'import sys, torch; torch.version.hip=None; import pytest; sys.exit(pytest.main(sys.argv[1:]))' \
    -q -p no:cacheprovider /repo/tests/check_gigachat35_parser_names.py \
    test/registered/unit/function_call/test_function_call_parser.py::TestToolCallParserNames \
    test/registered/unit/parser/test_reasoning_parser.py::TestReasoningParserNames \
    test/registered/unit/server_args/test_server_args.py::TestParserChoices
```

**Verification:** Through patch 59, both registry parity tests failed and the
CLI rejected `gigachat35` for each flag before the detectors loaded. With
patch 60, the broad CPU run (all tool detectors, forward timing, cache and
scheduler regression cases, plus both registries and CLI choices) passed:
**326 tests, 198 subtests, 0 failures**. Clean replay through patch 60 passed
**119/119** checks and compiled **70** changed Python files. No model server
was started or restarted for these tests.
