#!/usr/bin/env python3
"""Tool-call parameters survive a missing closing bracket and a schema the
parser could not read (patch 36).

The qwen3_coder tool-call parser receives every parameter value as text and
converts it by the type the tool schema gives it. Two gaps turned a model slip
into a client-side schema error ("handoff: Expected object, received string"):

  * An array or object value the model ended one closer short ("...}}" for
    "...}}}") fails json.loads, and the parser hands the client the raw text
    as a string. In the journal 14 of 18 such values were complete JSON
    missing one or two closers at the very end. The parser now closes a value
    that ends outside a string with containers still open (never one that is
    cut off inside a string or closes something it never opened), accepts the
    result only when its type matches the schema, and says so in one warning.
  * get_schema_properties() returned the top-level "properties" alone,
    ignoring anyOf/oneOf/allOf branches whenever the top level declared any,
    and followed no "$ref". A parameter it could not see was returned as a
    string even when its value was a well-formed JSON array. Branch properties
    are now merged with the top-level ones, local "$ref" pointers ("#/$defs/x",
    "#/definitions/x") are resolved for the schema and for each property, and a
    parameter the schema still does not list is returned as the JSON array or
    object it spells when it spells one.

Hooks: function_call/utils.py (get_schema_properties, two new helpers) and
function_call/qwen3_coder_detector.py (_convert_param_value, two new
methods). Anchors are asserted so an upstream rewrite fails the build. See
patches/36-tool-param-recovery.md.
"""
import sys


def patch(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old)
    return text.replace(old, new)


def patch_utils(path):
    text = open(path).read()
    text = patch(
        text,
        "def get_schema_properties(schema: Any) -> Dict[str, Any]:\n"
        '    """Top-level ``properties`` of a tool ``parameters`` schema, descending\n'
        "    into ``anyOf``/``oneOf``/``allOf`` branches when the top level declares\n"
        '    none (legal JSON Schema, e.g. discriminated-union arguments)."""\n'
        "    if not isinstance(schema, dict):\n"
        "        return {}\n"
        "    properties = schema.get(\"properties\")\n"
        "    if isinstance(properties, dict):\n"
        "        return properties\n"
        "    merged: Dict[str, Any] = {}\n"
        "    for keyword in (\"anyOf\", \"oneOf\", \"allOf\"):\n"
        "        branches = schema.get(keyword)\n"
        "        if isinstance(branches, list):\n"
        "            for branch in branches:\n"
        "                for key, value in get_schema_properties(branch).items():\n"
        "                    merged.setdefault(key, value)\n"
        "    return merged\n",
        "def resolve_local_ref(schema: Any, root: Any) -> Any:\n"
        '    """gfx1151 patch 36: ``schema`` with a local ``$ref`` (``#/$defs/x``,\n'
        "    ``#/definitions/x``) replaced by its target in ``root``, sibling keys\n"
        "    kept. A ref that is not local, does not resolve or cycles is left as\n"
        '    it is."""\n'
        "    seen = set()\n"
        "    while isinstance(schema, dict) and isinstance(schema.get(\"$ref\"), str):\n"
        "        ref = schema[\"$ref\"]\n"
        "        if not ref.startswith(\"#/\") or ref in seen:\n"
        "            break\n"
        "        seen.add(ref)\n"
        "        target = root\n"
        "        for part in ref[2:].split(\"/\"):\n"
        "            part = part.replace(\"~1\", \"/\").replace(\"~0\", \"~\")\n"
        "            if not isinstance(target, dict) or part not in target:\n"
        "                return schema\n"
        "            target = target[part]\n"
        "        if not isinstance(target, dict):\n"
        "            return schema\n"
        "        schema = {**target, **{k: v for k, v in schema.items() if k != \"$ref\"}}\n"
        "    return schema\n"
        "\n"
        "\n"
        "def get_schema_properties(schema: Any, root: Any = None) -> Dict[str, Any]:\n"
        '    """The parameters a tool ``parameters`` schema declares: its top-level\n'
        "    ``properties`` merged with those of its ``anyOf``/``oneOf``/``allOf``\n"
        "    branches (the top level wins on a name), with local ``$ref`` pointers\n"
        '    resolved for the schema and for every property (gfx1151 patch 36)."""\n'
        "    if root is None:\n"
        "        root = schema\n"
        "    schema = resolve_local_ref(schema, root)\n"
        "    if not isinstance(schema, dict):\n"
        "        return {}\n"
        "    merged: Dict[str, Any] = {}\n"
        "    properties = schema.get(\"properties\")\n"
        "    if isinstance(properties, dict):\n"
        "        for key, value in properties.items():\n"
        "            merged[key] = resolve_local_ref(value, root)\n"
        "    for keyword in (\"anyOf\", \"oneOf\", \"allOf\"):\n"
        "        branches = schema.get(keyword)\n"
        "        if isinstance(branches, list):\n"
        "            for branch in branches:\n"
        "                for key, value in get_schema_properties(branch, root).items():\n"
        "                    merged.setdefault(key, value)\n"
        "    return merged\n"
        "\n"
        "\n"
        "def close_unterminated_json(text: str) -> Optional[Tuple[str, int]]:\n"
        '    """gfx1151 patch 36: ``text``, a JSON array or object that stopped short\n'
        "    of its last closers, with them appended, and how many were added. None\n"
        "    when the text is not a container, is cut off inside a string, closes\n"
        "    something it never opened, or is already balanced: a repair there\n"
        '    would have to guess at content."""\n'
        "    text = text.strip()\n"
        "    if not text or text[0] not in \"[{\":\n"
        "        return None\n"
        "    stack = []\n"
        "    in_string = False\n"
        "    escaped = False\n"
        "    for ch in text:\n"
        "        if in_string:\n"
        "            if escaped:\n"
        "                escaped = False\n"
        "            elif ch == \"\\\\\":\n"
        "                escaped = True\n"
        "            elif ch == '\"':\n"
        "                in_string = False\n"
        "            continue\n"
        "        if ch == '\"':\n"
        "            in_string = True\n"
        "        elif ch in \"[{\":\n"
        "            stack.append(ch)\n"
        "        elif ch in \"]}\":\n"
        "            if not stack or stack[-1] != (\"[\" if ch == \"]\" else \"{\"):\n"
        "                return None\n"
        "            stack.pop()\n"
        "    if in_string or not stack:\n"
        "        return None\n"
        "    closers = \"\".join(\"]\" if c == \"[\" else \"}\" for c in reversed(stack))\n"
        "    return text + closers, len(stack)\n",
    )
    open(path, "w").write(text)


def patch_detector(path):
    text = open(path).read()
    text = patch(
        text,
        "from sglang.srt.function_call.utils import (\n"
        "    get_schema_properties,\n"
        "    infer_type_from_json_schema,\n"
        "    safe_literal_eval,\n"
        ")\n",
        "from sglang.srt.function_call.utils import (\n"
        "    close_unterminated_json,  # gfx1151 patch 36\n"
        "    get_schema_properties,\n"
        "    infer_type_from_json_schema,\n"
        "    safe_literal_eval,\n"
        ")\n",
    )
    text = patch(
        text,
        "    def _convert_param_value(\n"
        "        self, param_value: str, param_name: str, param_config: dict, func_name: str\n"
        "    ) -> Any:\n"
        '        """Convert parameter value based on its type in the schema."""\n'
        "        # Handle null value for any type\n"
        "        if param_value.lower() == \"null\":\n"
        "            return None\n"
        "\n"
        "        if param_name not in param_config:\n"
        "            if param_config != {}:\n"
        "                logger.warning(\n"
        "                    f\"Parsed parameter '{param_name}' is not defined in the tool \"\n"
        "                    f\"parameters for tool '{func_name}', directly returning the string value.\"\n"
        "                )\n"
        "            return param_value\n",
        "    def _close_unterminated(\n"
        "        self, param_value: str, param_name: str, param_type: str, func_name: str\n"
        "    ) -> Any:\n"
        '        """gfx1151 patch 36: the array or object ``param_value`` spells when it\n'
        "        only lacks its final closers and the result has the schema\'s type;\n"
        '        None otherwise."""\n'
        "        closed = close_unterminated_json(param_value)\n"
        "        if closed is None:\n"
        "            return None\n"
        "        closed_text, added = closed\n"
        "        try:\n"
        "            value = json.loads(closed_text)\n"
        "        except Exception:\n"
        "            return None\n"
        "        wants_list = param_type in [\"array\", \"arr\"] or param_type.startswith(\"list\")\n"
        "        if not isinstance(value, list if wants_list else dict):\n"
        "            return None\n"
        "        logger.warning(\n"
        "            f\"Parsed value of parameter '{param_name}' in tool '{func_name}' ended \"\n"
        "            f\"{added} closing bracket(s) short; parsed after closing it.\"\n"
        "        )\n"
        "        return value\n"
        "\n"
        "    @staticmethod\n"
        "    def _undeclared_structure(param_value: str) -> Any:\n"
        '        """gfx1151 patch 36: the JSON array or object an undeclared parameter\n'
        '        value spells, or None when it spells neither."""\n'
        "        stripped = param_value.strip()\n"
        "        if not stripped or stripped[0] not in \"[{\":\n"
        "            return None\n"
        "        try:\n"
        "            value = json.loads(stripped)\n"
        "        except Exception:\n"
        "            return None\n"
        "        return value if isinstance(value, (list, dict)) else None\n"
        "\n"
        "    def _convert_param_value(\n"
        "        self, param_value: str, param_name: str, param_config: dict, func_name: str\n"
        "    ) -> Any:\n"
        '        """Convert parameter value based on its type in the schema."""\n'
        "        # Handle null value for any type\n"
        "        if param_value.lower() == \"null\":\n"
        "            return None\n"
        "\n"
        "        if param_name not in param_config:\n"
        "            # gfx1151 patch 36: a parameter the schema does not list is handed\n"
        "            # over as the array or object it spells, if it spells one.\n"
        "            structure = self._undeclared_structure(param_value)\n"
        "            if param_config != {}:\n"
        "                logger.warning(\n"
        "                    f\"Parsed parameter '{param_name}' is not defined in the tool \"\n"
        "                    f\"parameters for tool '{func_name}', directly returning the \"\n"
        "                    + (\"JSON value.\" if structure is not None else \"string value.\")\n"
        "                )\n"
        "            return param_value if structure is None else structure\n",
    )
    text = patch(
        text,
        "                try:\n"
        "                    param_value = json.loads(param_value)\n"
        "                    return param_value\n"
        "                except Exception:\n"
        "                    logger.warning(\n"
        "                        f\"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool \"\n"
        "                        f\"'{func_name}', will try other methods to parse it.\"\n"
        "                    )\n",
        "                try:\n"
        "                    param_value = json.loads(param_value)\n"
        "                    return param_value\n"
        "                except Exception:\n"
        "                    closed = self._close_unterminated(\n"
        "                        param_value, param_name, param_type, func_name\n"
        "                    )  # gfx1151 patch 36\n"
        "                    if closed is not None:\n"
        "                        return closed\n"
        "                    logger.warning(\n"
        "                        f\"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool \"\n"
        "                        f\"'{func_name}', will try other methods to parse it.\"\n"
        "                    )\n",
    )
    open(path, "w").write(text)


def main():
    tree = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    srt = f"{tree}/python/sglang/srt"
    patch_utils(f"{srt}/function_call/utils.py")
    patch_detector(f"{srt}/function_call/qwen3_coder_detector.py")
    print("patched function_call/utils.py and function_call/qwen3_coder_detector.py")


if __name__ == "__main__":
    main()
