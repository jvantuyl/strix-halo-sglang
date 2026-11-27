#!/usr/bin/env python3
"""Patch 59: apply the reviewed upstream runtime and regression-test diffs."""

import subprocess
import sys
from pathlib import Path

PRODUCTION = (
    "43290-mixed-kv.patch",
)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    directory = Path(__file__).resolve().with_name("backports59")
    for name in PRODUCTION:
        # git apply checks every context hunk and refuses partial application.
        # Do not touch the index: earlier build steps leave unstaged changes.
        subprocess.run(["git", "apply", str(directory / name)], cwd=root, check=True)
        print(f"patch 59: applied {name}")


if __name__ == "__main__":
    main()
