#!/usr/bin/env python3
"""Make the idle SGLang scheduler sleep instead of busy-polling (patch 10).

Upstream's scheduler polls its ZMQ sockets with NOBLOCK in a tight loop, so an
idle server pins one CPU core at 100% forever. `--sleep-on-idle` replaces that
with a blocking zmq.Poller wait (1 s timeout, returns as soon as a request
arrives), but it is off by default. On a Strix Halo APU the CPU and GPU share
one cooler, and the spinning core alone took Tctl from 38 to 71 C at idle.

This flips the default in the pinned SGLang (re-anchored for the arg_groups
restructure):
  1. ServerArgs.sleep_on_idle defaults to True -- the field moved from
     server_args.py to arg_groups/fields/device.py. Engine / Python API and the
     generated CLI default both follow the field.
  2. The auto-generated CLI uses store_true for bools, which cannot unset a
     True default. The bool branch in arg_groups/arg_utils.py special-cases
     sleep_on_idle with BooleanOptionalAction, so `--sleep-on-idle` still
     parses and `--no-sleep-on-idle` restores the upstream busy-poll.

Every anchor is asserted, so an upstream move fails the build instead of
silently shipping the spin again. See patches/10-sleep-on-idle-default.md.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# 1. field default: device.py
p = f"{path}/python/sglang/srt/arg_groups/fields/device.py"
text = open(p).read()
old = '    sleep_on_idle: A[bool, "Reduce CPU usage when sglang is idle."] = False\n'
assert text.count(old) == 1, "device.py: sleep_on_idle field anchor not found"
new = (
    '    sleep_on_idle: A[bool, "Reduce CPU usage when sglang is idle."] = True'
    "  # gfx1151 patch 10: don't spin a core when idle\n"
)
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

# 2. CLI: arg_utils.py bool branch
p = f"{path}/python/sglang/srt/arg_groups/arg_utils.py"
text = open(p).read()
old = """        # Bool → store_true
        if inner_type is bool:
            kwargs = dict(action="store_true", help=arg_meta.help, **dest_kwarg)
            if default is not _MISSING:
                kwargs["default"] = default
            parser.add_argument(*names, **kwargs)
            continue
"""
assert text.count(old) == 1, "arg_utils.py: bool branch anchor not found"
new = """        # Bool → store_true
        if inner_type is bool:
            # gfx1151 patch 10: sleep_on_idle defaults to True in the fork;
            # BooleanOptionalAction lets --no-sleep-on-idle restore the
            # upstream busy-poll (plain store_true cannot unset a True
            # default).
            if dest_kwarg.get("dest") == "sleep_on_idle" or any(
                "sleep-on-idle" in name for name in names
            ):
                parser.add_argument(
                    *names,
                    action=argparse.BooleanOptionalAction,
                    default=bool(default) if default is not _MISSING else False,
                    help=arg_meta.help,
                    **dest_kwarg,
                )
                continue
            kwargs = dict(action="store_true", help=arg_meta.help, **dest_kwarg)
            if default is not _MISSING:
                kwargs["default"] = default
            parser.add_argument(*names, **kwargs)
            continue
"""
text = text.replace(old, new, 1)
if "import argparse" not in text:
    marker = "from __future__ import annotations\n"
    assert marker in text, "arg_utils.py: no __future__ import anchor"
    text = text.replace(marker, marker + "\nimport argparse\n", 1)
open(p, "w").write(text)
print("patched", p)

print("patch 10 (sleep-on-idle default) applied")
