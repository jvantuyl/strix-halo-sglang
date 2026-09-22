#!/usr/bin/env python3
"""Tune the Triton fused-MoE kernel for gfx1151, with an early bail-out.

Runs upstream's ``benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py``
with a gfx1151 search space and two small source edits applied to a temporary
copy:

1. Search space. RDNA 3.5 has wave32 SIMDs, 64 KB of LDS per workgroup and
   16x16 WMMA tiles, so the stock ROCm space (CDNA-oriented, BLOCK_M >= 32) is
   replaced: BLOCK_M includes 16 (what decode needs: at M <= 8 with top-k 10
   every expert sees a handful of rows), waves_per_eu in {0, 2}, and configs
   whose A+B tiles exceed 64 KB or give a wave less than one 16x16 tile are
   pruned before compilation. The space is written to a JSON file and passed
   through upstream's own ``--search-space-file``.

2. Early bail-out. The stock loop pays a graph capture plus 100 timed replays
   for every config. Here a single eager run after the JIT warmup drops a
   config that is already 2x over budget (grossly slow ones, e.g. BLOCK_M 256
   padding every expert at small M, never reach capture), then one replay is
   timed right after capture and the config is dropped if it is
   ``MOE_TUNE_BAIL_FACTOR`` (default 3) times slower than the best so far.
   35-60% of configs bail at every M; the sweep runs ~4x faster.

3. BLOCK_K filter. For int4 group quant the kernel derives the scale index per
   element, so BLOCK_K may be any multiple of the group size (the runtime
   default uses 64 with group 32). Upstream's filter only allows divisors.

4. Resilience. Each batch size runs in its own process and writes its result
   to ``MOE_TUNE_PARTIAL_DIR`` (upstream tunes every size in one Ray worker
   that keeps all compiled kernels resident and only writes the combined file
   at the end; on this box the kernel OOM-killed it part way through). Ray's
   memory monitor is disabled too: it counts page cache and killed the worker
   at "95%" with 10 GB genuinely free. Sizes already present in the partial
   directory are skipped, so a run can be resumed by pointing
   ``MOE_TUNE_PARTIAL_DIR`` at the previous one. The worker also appends every
   config to ``<partial dir>/trace-M=<m>.log`` (epoch time, index, best so
   far) before running it, so a config that kills the worker (GPU page fault)
   is identified by the last line and can be matched against ``dmesg``.

Environment: ``MOE_TUNE_BAIL_FACTOR`` (default 3.0), ``MOE_TUNE_PARTIAL_DIR``,
``SGLANG_MOE_TUNER_DIR``. Compiled kernels land in the normal Triton cache, so
an interrupted run resumes at replay speed.

Usage (inside the image, GPU attached):

    python3 tune_moe_gfx1151.py --model /models/qwen38 --dtype int4_w4a16 \
        --disable-shared-experts-fusion --tune [--batch-sizes 1 2 4 8 ...]

Writes E=...,N=...,device_name=...,dtype=int4_w4a16.json into the current
directory; copy it to python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/<triton_ver>/.
"""
import json
import os
import subprocess
import sys
import tempfile

TUNER_DIR = os.environ.get(
    "SGLANG_MOE_TUNER_DIR", "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton"
)
BAIL_FACTOR = float(os.environ.get("MOE_TUNE_BAIL_FACTOR", "3.0"))


def gfx1151_configs():
    configs = []
    for block_m in (16, 32, 64, 128, 256):
        for block_n in (16, 32, 64, 128, 256):
            for block_k in (16, 32, 64, 128):
                # Two stages of a bf16 A tile and a (dequantized) B tile.
                if (block_m + block_n) * block_k * 2 > 65536:
                    continue
                for num_warps in (1, 2, 4, 8):
                    # Each wave needs at least one 16x16 output tile, and more
                    # than 128 fp32 accumulators per lane (4096 per wave32)
                    # spills; those configs only cost compile time (~10 s each).
                    per_wave = block_m * block_n // num_warps
                    if per_wave < 256 or per_wave > 4096:
                        continue
                    for group_m in (1, 8, 32):
                        for waves_per_eu in (0, 2):
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_m,
                                    "num_warps": num_warps,
                                    "num_stages": 2,
                                    "waves_per_eu": waves_per_eu,
                                }
                            )
    return configs


src_path = os.path.join(TUNER_DIR, "tuning_fused_moe_triton.py")
src = open(src_path).read()


def replace(old, new):
    global src
    assert src.count(old) == 1, f"tuner anchor not found: {old[:60]!r}"
    src = src.replace(old, new, 1)


# Int4 group quant: BLOCK_K may also be a multiple of the group size.
replace(
    "                if block_k % config[\"BLOCK_SIZE_K\"] == 0\n",
    "                if block_k % config[\"BLOCK_SIZE_K\"] == 0\n"
    "                or (use_int4_w4a16 and config[\"BLOCK_SIZE_K\"] % block_k == 0)\n",
)

# benchmark_config: optional budget. A single eager run after the JIT warmup
# rejects grossly slow configs (2x the budget, so launch overhead cannot
# trigger it at small M) before paying for graph capture; the first replay
# after capture applies the exact budget.
replace(
    "    block_shape: List[int] = None,\n    num_iters: int = 100,\n) -> float:\n",
    "    block_shape: List[int] = None,\n    num_iters: int = 100,\n    bail_after_us: float = float('inf'),\n) -> float:\n",
)
replace(
    "    # JIT compilation & warmup\n    run()\n    torch.cuda.synchronize()\n",
    "    # JIT compilation & warmup\n    run()\n    torch.cuda.synchronize()\n"
    "    if bail_after_us != float('inf'):\n"
    "        eager_start = torch.cuda.Event(enable_timing=True)\n"
    "        eager_end = torch.cuda.Event(enable_timing=True)\n"
    "        eager_start.record()\n        run()\n        eager_end.record()\n"
    "        torch.cuda.synchronize()\n"
    "        eager_us = eager_start.elapsed_time(eager_end) * 1000\n"
    "        if eager_us > 2 * bail_after_us:\n            return eager_us\n",
)
replace(
    "    # Warmup\n    for _ in range(5):\n        graph.replay()\n    torch.cuda.synchronize()\n",
    "    # Warmup; the first timed replay doubles as the bail-out probe.\n"
    "    graph.replay()\n"
    "    probe_start = torch.cuda.Event(enable_timing=True)\n"
    "    probe_end = torch.cuda.Event(enable_timing=True)\n"
    "    probe_start.record()\n    graph.replay()\n    probe_end.record()\n"
    "    torch.cuda.synchronize()\n"
    "    probe_us = probe_start.elapsed_time(probe_end) / 10 * 1000\n"
    "    if probe_us > bail_after_us:\n        graph.reset()\n        return probe_us\n"
    "    for _ in range(3):\n        graph.replay()\n    torch.cuda.synchronize()\n",
)

# tune(): pass the budget, count how many configs bailed. Trace every config
# to <partial dir>/trace-M=<m>.log from the worker before it runs, so a
# config that kills the worker (GPU page fault) can be identified by index
# and matched against the kernel log by epoch time.
replace(
    "        best_config = None\n        best_time = float(\"inf\")\n",
    "        best_config = None\n        best_time = float(\"inf\")\n        bailed = 0\n"
    "        _trace = None\n"
    "        if os.environ.get('MOE_TUNE_PARTIAL_DIR'):\n"
    "            os.makedirs(os.environ['MOE_TUNE_PARTIAL_DIR'], exist_ok=True)\n"
    "            _trace = open(os.path.join(os.environ['MOE_TUNE_PARTIAL_DIR'], f'trace-M={num_tokens}.log'), 'a')\n",
)
replace(
    "            for config in tqdm(search_space):\n                try:\n",
    "            for _idx, config in enumerate(tqdm(search_space)):\n"
    "                if _trace is not None:\n"
    "                    _trace.write(f'{time.time():.3f} idx={_idx} best={best_time:.1f} {config}\\n')\n"
    "                    _trace.flush()\n"
    "                try:\n",
)
replace("import json\n", "import json\nimport time\n")
replace(
    "                        block_shape,\n                        num_iters=10,\n                    )\n",
    "                        block_shape,\n                        num_iters=10,\n"
    f"                        bail_after_us=best_time * {BAIL_FACTOR},\n"
    "                    )\n",
)
replace(
    "                if kernel_time < best_time:\n",
    f"                if kernel_time > best_time * {BAIL_FACTOR}:\n"
    "                    bailed += 1\n                    continue\n"
    "                if kernel_time < best_time:\n",
)
replace(
    '        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")\n',
    '        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}: '
    'best {best_time:.1f} us {best_config}, {bailed}/{len(search_space)} bailed", flush=True)\n'
    "        if os.environ.get('MOE_TUNE_PARTIAL_DIR'):\n"
    "            os.makedirs(os.environ['MOE_TUNE_PARTIAL_DIR'], exist_ok=True)\n"
    "            with open(os.path.join(os.environ['MOE_TUNE_PARTIAL_DIR'], f'M={num_tokens}.json'), 'w') as f:\n"
    "                json.dump({'M': num_tokens, 'time_us': best_time, 'config': best_config}, f)\n",
)
replace("import json\n", "import json\nimport os\n")

# One GPU, one worker: the default pool of one idle worker per host CPU costs
# ~2.5 GB of host RAM here for nothing.
replace("    ray.init()\n", "    ray.init(num_cpus=2)\n")

work_dir = tempfile.mkdtemp(prefix="tune_moe_gfx1151_")
patched_path = os.path.join(work_dir, "tuning_fused_moe_triton_gfx1151.py")
with open(patched_path, "w") as f:
    f.write(src)

argv = sys.argv[1:]
if "--search-space-file" not in argv:
    space_path = os.path.join(work_dir, "gfx1151_search_space.json")
    with open(space_path, "w") as f:
        json.dump(gfx1151_configs(), f)
    argv += ["--search-space-file", space_path]

# common_utils lives next to the upstream tuner; Ray workers inherit this env.
env = dict(os.environ)
env["PYTHONPATH"] = TUNER_DIR + os.pathsep + env.get("PYTHONPATH", "")
# Ray's memory monitor counts page cache and kills the worker at 95% of host
# RAM, which a 51B PLE table on the same box reaches long before real pressure.
env.setdefault("RAY_memory_monitor_refresh_ms", "0")
# Finished batch sizes are saved here as they complete; upstream only writes
# the combined file at the very end.
partial_dir = env.setdefault("MOE_TUNE_PARTIAL_DIR", os.path.join(work_dir, "partial"))
print(f"tune_moe_gfx1151: work dir {work_dir}, partials in {partial_dir}", flush=True)

if "--tune" not in argv:
    os.execve(sys.executable, [sys.executable, patched_path] + argv, env)

# Tuning: one process per batch size. The Ray worker keeps every compiled
# kernel resident and grew past what this box's host RAM allows over a full
# sweep (the kernel OOM-killed it at M=64 with no traceback); a fresh process
# per M holds one size's worth and resumes compiles from the Triton cache.
# Already-finished sizes in the partial directory are skipped.
sys.path.insert(0, TUNER_DIR)
import common_utils  # noqa: E402

if "--batch-sizes" in argv:
    i = argv.index("--batch-sizes") + 1
    j = i
    while j < len(argv) and not argv[j].startswith("--"):
        j += 1
    batch_sizes = [int(x) for x in argv[i:j]]
    argv = argv[: i - 1] + argv[j:]
elif "--batch-size" in argv:
    i = argv.index("--batch-size") + 1
    batch_sizes = [int(argv[i])]
    argv = argv[: i - 1] + argv[i + 1 :]
else:
    batch_sizes = common_utils.get_default_batch_sizes()

for m in batch_sizes:
    partial_path = os.path.join(partial_dir, f"M={m}.json")
    if os.path.exists(partial_path):
        print(f"tune_moe_gfx1151: M={m} already done, skipping", flush=True)
        continue
    for attempt in (1, 2):
        rc = subprocess.call(
            [sys.executable, patched_path] + argv + ["--batch-sizes", str(m)], env=env
        )
        if rc == 0 and os.path.exists(partial_path):
            break
        print(f"tune_moe_gfx1151: M={m} attempt {attempt} failed (rc={rc})", flush=True)
    else:
        sys.exit(f"tune_moe_gfx1151: giving up on M={m}")

# Merge the partials into the single file upstream's runtime loader expects.
# The last per-M run left a one-key file with the right name in the cwd.
merged = {}
for m in batch_sizes:
    with open(os.path.join(partial_dir, f"M={m}.json")) as f:
        entry = json.load(f)
    merged[str(m)] = common_utils.sort_config(entry["config"])
    print(f"M={m}: {entry['time_us']:.1f} us {entry['config']}")
outputs = [p for p in os.listdir(".") if p.startswith("E=") and p.endswith(".json")]
assert len(outputs) == 1, f"expected one E=*.json in {os.getcwd()}, found {outputs}"
with open(outputs[0], "w") as f:
    json.dump(merged, f, indent=4)
    f.write("\n")
print(f"tune_moe_gfx1151: wrote {outputs[0]} with {len(merged)} batch sizes", flush=True)
