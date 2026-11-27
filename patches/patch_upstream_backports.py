#!/usr/bin/env python3
"""Patch 59: apply the reviewed upstream runtime and regression-test diffs."""

import os
import subprocess
import sys
from pathlib import Path

PRODUCTION = (
    "43290-mixed-kv.patch",
    "44124-qwen-parser.patch",
    "38711-forward-metrics.patch",
)
HICACHE_RESTORE = (
    "39862-ple-checkpoints.patch",
    "39893-qsa-checkpoints.patch",
)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    directory = Path(__file__).resolve().with_name("backports59")
    profiles = {"production": PRODUCTION, "hicache-trial": PRODUCTION + HICACHE_RESTORE}
    profile = os.environ.get("SGLANG_BACKPORT_PROFILE", "production")
    if profile not in profiles:
        raise SystemExit(f"unknown SGLANG_BACKPORT_PROFILE: {profile!r}")
    print(f"patch 59 build profile: {profile}", flush=True)
    for name in profiles[profile]:
        # git apply checks every context hunk and refuses partial application.
        # Do not touch the index: earlier build steps leave unstaged changes.
        subprocess.run(["git", "apply", str(directory / name)], cwd=root, check=True)
        print(f"patch 59: applied {name}")


if __name__ == "__main__":
    main()
