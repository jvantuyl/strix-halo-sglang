#!/usr/bin/env python3
"""Guard against the failure modes behind issue #5, without an image build.

A fresh build broke because the Dockerfile tracked an unpinned upstream `main`,
and because a transitive dependency silently downgraded the base image's
gfx1151 ROCm torch. This script makes drift loud and deterministic in CI, with
no GPU and no 50 GB image build:

  1. The Dockerfile pins SGLang to an immutable ref (not a moving branch), and
     still freezes torch/torchvision through a pip constraint.
  2. Every patch step the Dockerfile runs is applied, in Dockerfile order, to a
     shallow clone of that pinned revision: the vendored sgl-kernel script, the
     inline heredoc patch and every patches/*.py it invokes. The scripts assert
     their own anchors, so a step that no longer applies fails here first. The
     files they touched must still compile.
  3. Bookkeeping: every script the Dockerfile references exists, every script in
     patches/ is either referenced or listed as retired below, and the pinned
     pyproject_other.toml still has the dependency groups the install flattens.

Run locally with: python3 tests/check_patches.py
"""
from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
PATCHES = ROOT / "patches"
IMAGE_TREE = "/sgl-workspace/sglang"

# Scripts kept for their write-ups but no longer run by the Dockerfile. A new
# orphan fails the bookkeeping check until it is wired in or listed here.
RETIRED = {
    # Superseded by patch 16: the dense WNA16 scheme is dequantized to bf16 at
    # load instead of routed to gptq_gemm.
    "patch_wna16_rocm.py": "superseded by patch 16 (patches/16-wna16-rocm-dense.md)",
    # Fixed upstream: compressed_tensors.py dispatches ParallelLMHead itself.
    "patch_lmhead_rocm.py": "fixed upstream (ParallelLMHead dispatch at the pin)",
}
# Patches the base image, not the SGLang tree; only an image build can check it.
BASE_IMAGE_PATCHES = {"fix_aiter_gfx1151_mxfp4.py"}
# The dependency groups the install step flattens out of pyproject_other.toml.
DEP_GROUPS = ("runtime_base", "runtime_common", "srt_hip")

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def arg_default(name: str) -> str | None:
    m = re.search(rf"^ARG {re.escape(name)}=(.+)$", DOCKERFILE, re.MULTILINE)
    return m.group(1).strip() if m else None


def clone_pinned(repo: str, ref: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="sglang-src-"))
    run = lambda *a: subprocess.run(a, check=True, capture_output=True)
    run("git", "init", "-q", str(d))
    run("git", "-C", str(d), "remote", "add", "origin", repo)
    run("git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", ref)
    run("git", "-C", str(d), "checkout", "-q", "FETCH_HEAD")
    return d


def dockerfile_steps() -> list[tuple[str, str]]:
    """The patch steps in Dockerfile order: ("script", name) for
    `RUN python3 /tmp/<name>`, ("heredoc", body) for `RUN python3 - <<'PYEOF'`,
    ("kernel", args) for the vendored sgl-kernel script."""
    steps: list[tuple[str, str]] = []
    lines = DOCKERFILE.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"RUN python3 /tmp/([A-Za-z0-9_]+\.py)", line)
        if m:
            steps.append(("script", m.group(1)))
        elif re.match(r"RUN python3 - <<'(\w+)'\s*$", line):
            tag = re.match(r"RUN python3 - <<'(\w+)'\s*$", line).group(1)
            body = []
            i += 1
            while i < len(lines) and lines[i] != tag:
                body.append(lines[i])
                i += 1
            steps.append(("heredoc", "\n".join(body) + "\n"))
        elif re.search(r"(^RUN|&&)\s+sh /tmp/sgl-kernel-gfx1151\.sh", line):
            # The COPY line also mentions the script; only the invocation counts.
            steps.append(("kernel", line.strip()))
        i += 1
    return steps


def run_step(cmd: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return r.returncode == 0, " | ".join(tail)


def main() -> int:
    print("Dockerfile invariants:")
    repo = arg_default("SGL_REPO") or ""
    ref = arg_default("SGL_BRANCH") or ""
    # A pinned ref is an immutable tag or 40-char SHA — never a moving branch.
    pinned = bool(re.fullmatch(r"[0-9a-f]{40}", ref)) or ref.startswith("v")
    check("SGLang pinned to an immutable ref (issue #5)", pinned, f"SGL_BRANCH={ref!r}")
    check("build freezes torch/torchvision via PIP_CONSTRAINT",
          "PIP_CONSTRAINT" in DOCKERFILE and "rocm-constraints.txt" in DOCKERFILE)
    check("build installs from pyproject_other.toml, flattening " + ", ".join(DEP_GROUPS),
          "cp python/pyproject_other.toml python/pyproject.toml" in DOCKERFILE
          and all(f'"{g}"' in DOCKERFILE for g in DEP_GROUPS))

    steps = dockerfile_steps()
    referenced = {name for kind, name in steps if kind == "script"} | {"sgl-kernel-gfx1151.sh"}
    for name in sorted(referenced):
        check(f"referenced script exists: patches/{name}", (PATCHES / name).is_file())
    present = {p.name for p in PATCHES.iterdir() if p.suffix in (".py", ".sh")}
    for name in sorted(present - referenced):
        check(f"unreferenced script is listed as retired: {name}", name in RETIRED,
              "add it to the Dockerfile or to RETIRED with the reason")
    for name in sorted(RETIRED):
        check(f"retired script is not also run by the build: {name}", name not in referenced)
    check("Dockerfile runs the vendored sgl-kernel script in the aot tree",
          any(kind == "kernel" for kind, _ in steps)
          and "WORKDIR /sgl-workspace/sglang/python/sglang/kernels/aot" in DOCKERFILE)

    if not repo or not ref:
        print("\nCould not read SGL_REPO/SGL_BRANCH from Dockerfile — aborting.")
        return 1

    print(f"\nCloning pinned SGLang {ref[:12]}…")
    try:
        src = clone_pinned(repo, ref)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ fetch failed: {e.stderr.decode()[:300]}")
        return 1

    print("Dependency groups at the pin:")
    pyproject = src / "python" / "pyproject_other.toml"
    text = pyproject.read_text() if pyproject.is_file() else ""
    try:
        import tomllib
        extras = tomllib.loads(text)["project"]["optional-dependencies"] if text else {}
        for g in DEP_GROUPS:
            check(f"pyproject_other.toml has [{g}]", g in extras)
    except ImportError:  # python < 3.11: a textual check is still better than none
        for g in DEP_GROUPS:
            check(f"pyproject_other.toml has [{g}]", re.search(rf"^{g} = \[", text, re.M) is not None)

    print("Patch steps applied to the pin, in Dockerfile order:")
    for kind, arg in steps:
        if kind == "kernel":
            setup = src / "python/sglang/kernels/aot/setup_rocm.py"
            ok, tail = run_step(["sh", str(PATCHES / "sgl-kernel-gfx1151.sh"), str(setup)])
            check("sgl-kernel-gfx1151.sh (arch gate + wave32 WARP_SIZE)", ok, tail)
        elif kind == "heredoc":
            if "pip" in arg:
                continue  # the install step; its inputs are checked above
            body = arg.replace(IMAGE_TREE, str(src))
            # Patch 2 and any future inline step name their file in the body.
            label = re.search(r"/([\w/.-]+\.py)'", arg)
            ok, tail = run_step([sys.executable, "-c", body])
            check(f"inline patch: {label.group(1) if label else 'heredoc'}", ok, tail)
        elif arg in BASE_IMAGE_PATCHES:
            print(f"  - {arg}: patches the base image's site-packages, checked at image build only")
        else:
            ok, tail = run_step([sys.executable, str(PATCHES / arg), str(src)])
            check(f"patches/{arg}", ok, tail)

    print("Patched files compile:")
    changed = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True
    ).stdout.splitlines()
    py_files = [l[3:].strip() for l in changed if l[3:].strip().endswith(".py")]
    check("patches touched the tree", bool(changed))
    bad = []
    for rel in py_files:
        try:
            py_compile.compile(str(src / rel), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{rel}: {e.msg.splitlines()[-1]}")
    check(f"{len(py_files)} patched .py files compile", not bad, "; ".join(bad))

    print(f"\n{checks - len(failures)}/{checks} checks passed.")
    if failures:
        print("FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("All patch steps apply to the pinned SGLang; invariants intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
