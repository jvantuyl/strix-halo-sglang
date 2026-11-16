# Patch 36: tool-call parameters survive a missing closer and an unread schema

**Files:** `python/sglang/srt/function_call/utils.py`,
`python/sglang/srt/function_call/qwen3_coder_detector.py`
**Script:** [`patch_tool_param_recovery.py`](patch_tool_param_recovery.py)
**Check:** [`tests/check_tool_param_recovery.py`](../tests/check_tool_param_recovery.py)
(137 checks, streaming equals one-shot under four chunkings)
**Switch:** none; always on

## Why

The `qwen3_coder` tool-call parser receives each parameter as text between
`<parameter=name>` and `</parameter>` and converts it by the type the tool
schema gives it. Two gaps in that conversion turned a model slip into a
client-side schema error. Both showed up in the journal as the agent's
`handoff: Expected object, received string` and `questions: Expected array,
received string`.

**A closer short.** When the schema says array or object, `json.loads` is
tried and, failing that, `ast.literal_eval`; if both fail the raw text is
handed on as a string. Of the 18 "cannot be parsed with json.loads" warnings
since 2026-11-01 (`handoff`, `calls`, `operations`, `questions`), 14 were
complete JSON that stopped one closer short (one of them two): `...}}` where
`...}}}` was due, on payloads of 300 bytes to 8 KB. The model loses the
bracket count at the end of a long nested value. The client then sees a
string where it declared an object and reports the shape, not the bracket,
and the agent retries the same content.

**A property the parser could not see.** `get_schema_properties()` returned
the top-level `properties` and descended into `anyOf`/`oneOf`/`allOf` only
when the top level declared none, and followed no `$ref`. A parameter
declared in a branch beside top-level properties, or typed through
`#/$defs/...` (what pydantic and zod-to-json-schema emit), was "not defined
in the tool parameters" and returned as a string even when its value was a
well-formed JSON array. The 2026-11-12 `AskUser` failures carried that
warning while the value parsed.

## What changes

`utils.py`

- `get_schema_properties(schema, root=None)` merges the top-level
  `properties` with those of every `anyOf`/`oneOf`/`allOf` branch (the top
  level wins on a name) and resolves local `$ref` pointers, for the schema
  itself and for each property, through the new `resolve_local_ref()`.
  Sibling keys beside a `$ref` are kept; a ref that is not local, does not
  resolve, or cycles is left as it is. A schema with top-level properties and
  no branches returns exactly what it did before.
- `close_unterminated_json(text)` walks a value string-aware (escapes
  honoured) and returns it with the missing closers appended, plus the count,
  only when the text starts a container, ends outside a string, closes
  nothing it never opened and still has containers open. Everything else
  returns `None`: a repair there would have to guess at content.

`qwen3_coder_detector.py`, `_convert_param_value` (shared by the one-shot and
streaming paths)

- For an array/object parameter whose value fails `json.loads`, the value is
  closed with `close_unterminated_json` and accepted only if the result has
  the schema's type (list for array, dict for object); one warning names the
  parameter and how many closers were added. Otherwise the previous
  behaviour (literal_eval, then string) runs unchanged.
- For a parameter the schema does not list, the value is returned as the
  JSON array or object it spells when `json.loads` yields one; otherwise as
  the string it was. No closing is attempted here. This is the one judgement
  call in the patch: upstream deliberately returns a string for undeclared
  parameters, and a string-typed parameter that merely looks like JSON is
  still never converted (the schema is consulted first). Drop
  `_undeclared_structure` if the behaviour is unwanted.

## What stays as it was

The four journal values broken in the middle (a `}` dropped before a `]`
inside an array, a missing `,` between fields) still come through as strings.
A value cut off inside a string is not completed. Nothing is changed in the
model, template, sampler or request path; the closer the model forgot is
still forgotten, the server only stops mislabelling it. The warning keeps the
event visible in the journal.

Not addressed here, same area: in the streaming path the arguments object is
closed only on `</function>`, so a call the model ends with `</tool_call>`
alone (or with EOS) streams out as `{"k": v` (sgl-project/sglang#43150,
open upstream). Not seen in this journal.

## Check

```
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -v $PWD:/repo:ro --entrypoint python3 \
  strix-halo-sglang:dev /repo/tests/check_tool_param_recovery.py
```

(The device flags are only so that importing `sglang` can initialise Triton;
the check itself runs on the CPU.) Run against the 18 journal values, the
patched converter returns 14 lists/dicts and 4 strings;
`tests/check_control_token_parsers.py` still passes 148/148.
