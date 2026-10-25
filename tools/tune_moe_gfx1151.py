#!/usr/bin/env python3
"""Tune the Triton fused-MoE kernel for gfx1151, with early bail-outs,
known-winner seeding and crash-safe checkpoints.

Runs upstream's ``benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py``
with a gfx1151 search space and a handful of source edits applied to a
temporary copy:

1. Search space. RDNA 3.5 has wave32 SIMDs, 64 KB of LDS per workgroup and
   16x16 WMMA tiles, so the stock ROCm space (CDNA-oriented, BLOCK_M >= 32) is
   replaced: BLOCK_M includes 16 (what decode needs: at M <= 8 with top-k 10
   every expert sees a handful of rows), waves_per_eu in {0, 2}, and configs
   whose A+B tiles exceed 64 KB or give a wave less than one 16x16 tile are
   pruned before compilation. The space is written to a JSON file and passed
   through upstream's own ``--search-space-file``.

2. Bail-outs. The stock loop pays a graph capture, 5 warmup replays and 10
   timed replays (160 kernel invocations) for every config, however slow.
   Here, with ``best`` the best per-invocation time so far:
     a. one eager run after the JIT warmup drops a config whose eager time
        is already 2 x MOE_TUNE_BAIL_FACTOR x the winner's eager time (grossly
        slow ones, e.g. BLOCK_M 256 padding every expert at small M, never
        reach graph capture). Eager is compared with eager because at small M
        both are mostly CPU launch overhead;
     b. the first replay after capture is timed and the config is dropped if
        it is MOE_TUNE_BAIL_FACTOR (default 3) x best;
     c. the timed loop keeps a running total and stops the moment it reaches
        the total the current winner needed (best x 10 x num_iters): the
        average can no longer beat the winner, so the remaining replays are
        pointless. This is exact (no measurement slack), and it is where the
        time goes at large M, where a single replay is 100+ ms and compile is
        already cached.
   The tighter ``best`` is from config #1, the more every rule cuts, hence:

3. Known-winner seeding. Before the sweep for M, the winners of the nearest
   finished batch sizes (from ``MOE_TUNE_PARTIAL_DIR``) and the nearest entry
   of ``MOE_TUNE_SEED_FILE`` (an existing runtime config JSON, e.g. the file
   being re-tuned) are moved to the front of the search space, so ``best`` is
   realistic from the first config instead of after a lucky early one. Seeds
   that the search space does not contain are ignored.

4. Crash safety. Each batch size runs in its own process (upstream tunes every
   size in one Ray worker that keeps all compiled kernels resident and only
   writes the combined file at the end; on this box the kernel OOM-killed it
   part way through). Results are written to ``MOE_TUNE_PARTIAL_DIR`` as they
   happen, fsync'd and atomically renamed:
     - ``M=<m>.json`` when a batch size finishes (sizes already present are
       skipped, so a run resumes by pointing at the same directory);
     - ``progress-M=<m>.json`` after every single config (index, best so far),
       so a crash mid-sweep resumes at the next config, not at the start of
       that M. The checkpoint carries a hash of the (seeded) search-space
       order and is ignored if the order changed.
     - ``trace-M=<m>.log`` gets a line per config before it runs, so a config
       that kills the worker (GPU page fault) is identified by the last line.
   The directory defaults to ``~/.cache/sglang/moe-tune/<tag>`` (persisted
   across container restarts by the launcher's cache mount; the tag is a hash
   of the model/dtype arguments) and is printed at startup. Compiled kernels
   go to the same persisted Triton cache the server uses
   (``TRITON_CACHE_DIR``), so an interrupted run resumes at replay speed.

5. Bisection over batch sizes (``MOE_TUNE_BISECT``, default on). Sizes are
   visited smallest, largest, then the midpoint of every unresolved interval.
   When the nearest finished sizes below and above an M have the same winner
   and are at most ``MOE_TUNE_BISECT_MAX_GAP`` positions apart in the size
   list (default 2: the M is adjacent to both, so no bracket wider than 4x is
   ever skipped; a pocket where another tile wins and then loses again would
   otherwise hide between two agreeing endpoints), that M only gets a quick
   check (the neighbours' winners benchmarked at this M, seconds) instead of
   the full sweep (an hour at large M); the runtime picks the nearest tuned M
   anyway. Where the neighbours disagree the regime is changing and the
   midpoint is swept in full. ``MOE_TUNE_BISECT=0`` gives the plain ascending
   sweep.

6. BLOCK_K filter. For int4 group quant the kernel derives the scale index per   element, so BLOCK_K may be any multiple of the group size. Upstream's
   filter only allows divisors.

7. Ray's memory monitor is disabled: it counts page cache and killed the
   worker at "95%" with 10 GB genuinely free.

Environment: ``MOE_TUNE_BAIL_FACTOR`` (3.0), ``MOE_TUNE_WARMUP_REPLAYS`` (5),
``MOE_TUNE_BISECT`` (1), ``MOE_TUNE_BISECT_MAX_GAP`` (2), ``MOE_TUNE_PARTIAL_DIR``,
``MOE_TUNE_SEED_FILE``, ``SGLANG_MOE_TUNER_DIR``, ``TRITON_CACHE_DIR``.

Usage (inside the image, GPU attached):

    python3 tune_moe_gfx1151.py --model /models/qwen38 --dtype int4_w4a16 \
        --disable-shared-experts-fusion --tune [--batch-sizes 1 2 4 8 ...]

Writes E=...,N=...,device_name=...[,dtype=...].json into the current
directory; copy it to the profile directory mounted at /moe-configs.

Note on ``--tp-size``: the runtime keys the int4 file on the packed width
(intermediate 640 stored as 320 int32 columns -> ``N=320``), while upstream's
tuner uses ``intermediate / tp_size``. The default ``--tp-size 2`` therefore
produces the file name the runtime reads but benchmarks a half-width shard;
``--tp-size 1`` measures the true shape and writes an ``N=640`` file that
must be renamed to ``N=320``. For the bf16 draft MoE (``--dtype auto``) the
runtime keys on the real width, so ``--tp-size 1`` gives the right name.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

TUNER_DIR = os.environ.get(
    "SGLANG_MOE_TUNER_DIR", "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton"
)
BAIL_FACTOR = float(os.environ.get("MOE_TUNE_BAIL_FACTOR", "3.0"))
WARMUP_REPLAYS = int(os.environ.get("MOE_TUNE_WARMUP_REPLAYS", "5"))


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


replace(
    "import json\n",
    "import json\nimport hashlib\nimport math\nimport os\nimport re\nimport time\n",
)

# Int4 group quant: BLOCK_K may also be a multiple of the group size.
replace(
    "                if block_k % config[\"BLOCK_SIZE_K\"] == 0\n",
    "                if block_k % config[\"BLOCK_SIZE_K\"] == 0\n"
    "                or (use_int4_w4a16 and config[\"BLOCK_SIZE_K\"] % block_k == 0)\n",
)

# Helpers used by the patched tune(): config normalisation for seed matching,
# fsync'd atomic JSON writes, and seed lookup.
replace(
    "@ray.remote(num_gpus=1)\nclass BenchmarkWorker:\n",
    '''_GFX_KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
             "num_warps", "num_stages", "waves_per_eu")
_GFX_DEFAULTS = {"num_stages": 2, "waves_per_eu": 0}
# Eager time of the config benchmark_config measured last; tune() keeps the
# winner's so the pre-capture check compares eager against eager (both carry
# the same CPU launch overhead, which dwarfs the kernel at small M).
_gfx_eager_last = [float("inf")]
# Outcome of the last config ("eager", "probe=<us>", "cum@<replay>=<ms>", "ok=<us>")
# for the trace log.
_gfx_last_note = [""]


def _gfx_norm(config):
    return {k: config.get(k, _GFX_DEFAULTS.get(k)) for k in _GFX_KEYS}


def _gfx_atomic_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _gfx_seeds(num_tokens, partial_dir, max_seeds=2):
    """Winners of the nearest finished batch sizes (log distance), from the
    partial directory and from MOE_TUNE_SEED_FILE, deduplicated."""
    cands = []
    if partial_dir and os.path.isdir(partial_dir):
        for name in os.listdir(partial_dir):
            m = re.match(r"M=(\\d+)\\.json$", name)
            if not m:
                continue
            try:
                with open(os.path.join(partial_dir, name)) as f:
                    entry = json.load(f)
                cands.append((abs(math.log(int(m.group(1)) / num_tokens)), entry["config"]))
            except (OSError, ValueError, KeyError):
                continue
    seed_file = os.environ.get("MOE_TUNE_SEED_FILE")
    if seed_file and os.path.exists(seed_file):
        with open(seed_file) as f:
            data = json.load(f)
        for key, cfg in data.items():
            if key.isdigit() and isinstance(cfg, dict):
                cands.append((abs(math.log(int(key) / num_tokens)), cfg))
    cands.sort(key=lambda t: t[0])
    out = []
    for _, cfg in cands:
        n = _gfx_norm(cfg)
        if n not in out:
            out.append(n)
    return out[:max_seeds]


@ray.remote(num_gpus=1)
class BenchmarkWorker:
''',
)

# benchmark_config: ``best_us`` is the best per-invocation time so far (inf
# when nothing has finished yet). Eager pre-capture check, probe replay, and a
# cumulative cutoff in the timed loop; bails return inf.
replace(
    "    block_shape: List[int] = None,\n    num_iters: int = 100,\n) -> float:\n",
    "    block_shape: List[int] = None,\n    num_iters: int = 100,\n    best_us: float = float('inf'),\n    best_eager_us: float = float('inf'),\n) -> float:\n",
)
replace(
    "    # JIT compilation & warmup\n    run()\n    torch.cuda.synchronize()\n",
    "    # JIT compilation & warmup\n    run()\n    torch.cuda.synchronize()\n"
    "    eager_start = torch.cuda.Event(enable_timing=True)\n"
    "    eager_end = torch.cuda.Event(enable_timing=True)\n"
    "    eager_start.record()\n    run()\n    eager_end.record()\n"
    "    torch.cuda.synchronize()\n"
    "    eager_us = eager_start.elapsed_time(eager_end) * 1000\n"
    "    _gfx_eager_last[0] = eager_us\n"
    f"    if eager_us > 2 * {BAIL_FACTOR} * best_eager_us:\n"
    "        _gfx_last_note[0] = f'eager={eager_us:.1f}'\n        return float('inf')\n",
)
replace(
    "    # Warmup\n    for _ in range(5):\n        graph.replay()\n    torch.cuda.synchronize()\n",
    "    # Warmup; the first timed replay doubles as the bail-out probe.\n"
    f"    for _ in range({WARMUP_REPLAYS}):\n        graph.replay()\n"
    "    probe_start = torch.cuda.Event(enable_timing=True)\n"
    "    probe_end = torch.cuda.Event(enable_timing=True)\n"
    "    probe_start.record()\n    graph.replay()\n    probe_end.record()\n"
    "    torch.cuda.synchronize()\n"
    "    probe_us = probe_start.elapsed_time(probe_end) / 10 * 1000\n"
    f"    if probe_us > {BAIL_FACTOR} * best_us:\n"
    "        _gfx_last_note[0] = f'probe={probe_us:.1f}'\n        graph.reset()\n        return float('inf')\n",
)
replace(
    "    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]\n"
    "    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]\n"
    "\n"
    "    for i in range(num_iters):\n"
    "        prepare(i)\n"
    "        start_events[i].record()\n"
    "        graph.replay()\n"
    "        end_events[i].record()\n"
    "    torch.cuda.synchronize()\n"
    "\n"
    "    latencies: List[float] = []\n"
    "    for i in range(num_iters):\n"
    "        latencies.append(start_events[i].elapsed_time(end_events[i]))\n"
    "    avg = sum(latencies) / (num_iters * 10) * 1000  # us\n"
    "    graph.reset()\n"
    "    return avg\n",
    "    # Cumulative cutoff: once this config has used as much time as the\n"
    "    # winner needed for the whole loop, its average cannot be lower.\n"
    "    budget_ms = best_us * num_iters * 10 / 1000\n"
    "    start_event = torch.cuda.Event(enable_timing=True)\n"
    "    end_event = torch.cuda.Event(enable_timing=True)\n"
    "    total_ms = 0.0\n"
    "    for i in range(num_iters):\n"
    "        prepare(i)\n"
    "        start_event.record()\n"
    "        graph.replay()\n"
    "        end_event.record()\n"
    "        end_event.synchronize()\n"
    "        total_ms += start_event.elapsed_time(end_event)\n"
    "        if total_ms >= budget_ms:\n"
    "            _gfx_last_note[0] = f'cum@{i + 1}={total_ms:.2f}ms'\n"
    "            graph.reset()\n"
    "            return float('inf')\n"
    "    avg = total_ms / (num_iters * 10) * 1000  # us\n"
    "    _gfx_last_note[0] = f'ok={avg:.1f}'\n"
    "    graph.reset()\n"
    "    return avg\n",
)

# tune(): seed the search space, resume from the per-config checkpoint, trace
# every config before it runs, checkpoint after it, save the result per M.
replace(
    "        best_config = None\n        best_time = float(\"inf\")\n",
    "        best_config = None\n        best_time = float(\"inf\")\n        bailed = 0\n"
    "        best_eager = float('inf')\n"
    "        _pd = os.environ.get('MOE_TUNE_PARTIAL_DIR')\n"
    "        _trace = None\n        _progress_path = None\n        _start_idx = 0\n"
    "        _seeds = [c for c in _gfx_seeds(num_tokens, _pd) if c in search_space]\n"
    "        if _seeds:\n"
    "            search_space = _seeds + [c for c in search_space if c not in _seeds]\n"
    "            print(f'M={num_tokens}: seeding with {_seeds}', flush=True)\n"
    "        _space_hash = hashlib.sha1(json.dumps(search_space, sort_keys=True).encode()).hexdigest()\n"
    "        if _pd:\n"
    "            os.makedirs(_pd, exist_ok=True)\n"
    "            _trace = open(os.path.join(_pd, f'trace-M={num_tokens}.log'), 'a')\n"
    "            _progress_path = os.path.join(_pd, f'progress-M={num_tokens}.json')\n"
    "            if os.path.exists(_progress_path):\n"
    "                with open(_progress_path) as f:\n"
    "                    _p = json.load(f)\n"
    "                if _p.get('space_hash') == _space_hash:\n"
    "                    _start_idx = _p['idx'] + 1\n"
    "                    best_time = _p['best_time'] if _p['best_time'] is not None else float('inf')\n"
    "                    best_config = _p['best_config']\n"
    "                    best_eager = _p['best_eager'] if _p.get('best_eager') is not None else float('inf')\n"
    "                    bailed = _p.get('bailed', 0)\n"
    "                    # The trace has a line for the config that was running when the\n"
    "                    # worker died; if it is the next one, it probably killed the worker.\n"
    "                    try:\n"
    "                        with open(os.path.join(_pd, f'trace-M={num_tokens}.log')) as f:\n"
    "                            _last = f.read().rstrip('\\n').rsplit('\\n', 1)[-1]\n"
    "                        _crashed = int(re.search(r'idx=(\\d+)', _last).group(1))\n"
    "                    except (OSError, AttributeError, ValueError):\n"
    "                        _crashed = None\n"
    "                    if _crashed == _start_idx:\n"
    "                        print(f'M={num_tokens}: config {_crashed} did not finish last time, skipping it: {search_space[_crashed]}', flush=True)\n"
    "                        _trace.write(f'{time.time():.3f} idx={_crashed} skipped (killed the worker last time)\\n')\n"
    "                        _start_idx += 1\n"
    "                    print(f'M={num_tokens}: resuming at config {_start_idx}/{len(search_space)}, best so far {best_time:.1f} us', flush=True)\n"
    "                else:\n"
    "                    print(f'M={num_tokens}: search space changed, ignoring checkpoint', flush=True)\n"
    "        def _checkpoint(idx):\n"
    "            if _progress_path is not None:\n"
    "                _gfx_atomic_json(_progress_path, {'idx': idx, 'space_hash': _space_hash,\n"
    "                    'best_time': None if best_time == float('inf') else best_time,\n"
    "                    'best_eager': None if best_eager == float('inf') else best_eager,\n"
    "                    'best_config': best_config, 'bailed': bailed})\n",
)
replace(
    "            for config in tqdm(search_space):\n                try:\n",
    "            for _idx, config in enumerate(tqdm(search_space)):\n"
    "                if _idx < _start_idx:\n                    continue\n"
    "                if _trace is not None:\n"
    "                    _trace.write(f'{time.time():.3f} idx={_idx} best={best_time:.1f} best_eager={best_eager:.1f} last_eager={_gfx_eager_last[0]:.1f} last={_gfx_last_note[0]} {config}\\n')\n"
    "                    _trace.flush()\n"
    "                try:\n",
)
replace(
    "                        block_shape,\n                        num_iters=10,\n                    )\n",
    "                        block_shape,\n                        num_iters=10,\n"
    "                        best_us=best_time,\n"
    "                        best_eager_us=best_eager,\n"
    "                    )\n",
)
replace(
    "                if kernel_time < best_time:\n"
    "                    best_time = kernel_time\n"
    "                    best_config = config\n",
    "                if kernel_time == float('inf'):\n"
    "                    bailed += 1\n"
    "                elif kernel_time < best_time:\n"
    "                    best_time = kernel_time\n"
    "                    best_config = config\n"
    "                    best_eager = _gfx_eager_last[0]\n"
    "                _checkpoint(_idx)\n",
)
replace(
    '        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")\n',
    '        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}: '
    'best {best_time:.1f} us {best_config}, {bailed}/{len(search_space)} bailed", flush=True)\n'
    "        if _pd:\n"
    "            _gfx_atomic_json(os.path.join(_pd, f'M={num_tokens}.json'),\n"
    "                {'M': num_tokens, 'time_us': best_time, 'config': best_config})\n"
    "            if _progress_path is not None and os.path.exists(_progress_path):\n"
    "                os.remove(_progress_path)\n",
)

# main(): record the runtime file name so the driver can merge even when every
# size was already finished in an earlier run.
replace(
    '        print(\n            f"Start tuning over {len(search_space)} configurations to create {filename}..."\n        )\n',
    '        print(\n            f"Start tuning over {len(search_space)} configurations to create {filename}..."\n        )\n'
    "        if os.environ.get('MOE_TUNE_PARTIAL_DIR'):\n"
    "            os.makedirs(os.environ['MOE_TUNE_PARTIAL_DIR'], exist_ok=True)\n"
    "            with open(os.path.join(os.environ['MOE_TUNE_PARTIAL_DIR'], 'filename.txt'), 'w') as f:\n"
    "                f.write(filename)\n",
)

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
# Compile into the server's persisted Triton cache (the launcher mounts
# ~/.cache/sglang), not a per-container directory that dies with it.
sglang_cache = os.path.expanduser("~/.cache/sglang")
if os.path.isdir(sglang_cache):
    env.setdefault("TRITON_CACHE_DIR", os.path.join(sglang_cache, "triton"))

if "--tune" not in argv:
    os.execve(sys.executable, [sys.executable, patched_path] + argv, env)

# Tuning: one process per batch size. The Ray worker keeps every compiled
# kernel resident and grew past what this box's host RAM allows over a full
# sweep (the kernel OOM-killed it at M=64 with no traceback); a fresh process
# per M holds one size's worth and resumes compiles from the Triton cache.
# Already-finished sizes in the partial directory are skipped; a size that
# was interrupted resumes at its next config.
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

# Results are keyed by everything but the batch sizes, so a later run with
# more sizes (or the same ones after a crash) lands in the same directory.
if "MOE_TUNE_PARTIAL_DIR" not in env:
    tag_args = [a for a in argv if not a.startswith(work_dir)]
    tag = hashlib.sha1(" ".join(tag_args).encode()).hexdigest()[:10]
    base = sglang_cache if os.path.isdir(sglang_cache) else work_dir
    env["MOE_TUNE_PARTIAL_DIR"] = os.path.join(base, "moe-tune", tag)
partial_dir = env["MOE_TUNE_PARTIAL_DIR"]
os.makedirs(partial_dir, exist_ok=True)
print(f"tune_moe_gfx1151: work dir {work_dir}, partials in {partial_dir}", flush=True)
if env.get("MOE_TUNE_SEED_FILE"):
    print(f"tune_moe_gfx1151: seeding from {env['MOE_TUNE_SEED_FILE']}", flush=True)

BISECT = os.environ.get("MOE_TUNE_BISECT", "1") != "0"
# Widest bracket (in list positions) a quick check may fill in. Adjacent sizes
# are at most 2x apart, so 2 means nothing wider than 4x is ever skipped.
BISECT_MAX_GAP = int(os.environ.get("MOE_TUNE_BISECT_MAX_GAP", "2"))


def partial_path(m):
    return os.path.join(partial_dir, f"M={m}.json")


def winner(m):
    with open(partial_path(m)) as f:
        return json.load(f)["config"]


def run_size(m, space_path=None, label="sweep"):
    """One batch size in its own process, retried from the checkpoint."""
    args = list(argv)
    if space_path is not None:
        args[args.index("--search-space-file") + 1] = space_path
    for attempt in (1, 2, 3, 4, 5):
        rc = subprocess.call(
            [sys.executable, patched_path] + args + ["--batch-sizes", str(m)], env=env
        )
        if rc == 0 and os.path.exists(partial_path(m)):
            return
        print(f"tune_moe_gfx1151: M={m} {label} attempt {attempt} failed (rc={rc}), resuming from checkpoint", flush=True)
    sys.exit(f"tune_moe_gfx1151: giving up on M={m}")


def bisect_order(sizes):
    """Endpoints first, then the midpoint of every unresolved interval, widest
    intervals first, so neighbouring winners are known as early as possible."""
    if len(sizes) <= 2:
        return list(sizes)
    order = [sizes[0], sizes[-1]]
    intervals = [(0, len(sizes) - 1)]
    while intervals:
        lo, hi = intervals.pop(0)
        if hi - lo < 2:
            continue
        mid = (lo + hi) // 2
        order.append(sizes[mid])
        intervals += [(lo, mid), (mid, hi)]
    return order


sizes = sorted(set(batch_sizes))
for m in bisect_order(sizes) if BISECT else sizes:
    if os.path.exists(partial_path(m)):
        print(f"tune_moe_gfx1151: M={m} already done, skipping", flush=True)
        continue
    done = [s for s in sizes if os.path.exists(partial_path(s))]
    below = [s for s in done if s < m]
    above = [s for s in done if s > m]
    close = (
        below and above
        and sizes.index(above[0]) - sizes.index(below[-1]) <= BISECT_MAX_GAP
    )
    if BISECT and close and winner(below[-1]) == winner(above[0]):
        # Same tile wins on both sides: the kernel is in one regime across
        # this interval. Measure just the neighbours' winners here instead of
        # the full space (a sweep at large M costs an hour; this costs seconds).
        cands = []
        for s in below[-2:] + above[:2]:
            if winner(s) not in cands:
                cands.append(winner(s))
        space_path = os.path.join(work_dir, f"quick_space_M={m}.json")
        with open(space_path, "w") as f:
            json.dump(cands, f)
        print(f"tune_moe_gfx1151: M={m}: neighbours M={below[-1]} and M={above[0]} agree, "
              f"quick check of {len(cands)} candidate(s) instead of a sweep", flush=True)
        run_size(m, space_path, label="quick check")
        with open(partial_path(m)) as f:
            entry = json.load(f)
        entry["quick_check"] = True
        with open(partial_path(m) + f".{os.getpid()}.tmp", "w") as f:
            json.dump(entry, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial_path(m) + f".{os.getpid()}.tmp", partial_path(m))
    else:
        if BISECT and below and above:
            why = "disagree" if close else f"are more than {BISECT_MAX_GAP} sizes apart"
            print(f"tune_moe_gfx1151: M={m}: neighbours M={below[-1]} and M={above[0]} {why}, full sweep", flush=True)
        run_size(m)

# Merge every finished size in the partial directory (not just this run's)
# into the single file upstream's runtime loader expects.
merged = {}
done = sorted(
    int(p[2:-5]) for p in os.listdir(partial_dir) if p.startswith("M=") and p.endswith(".json")
)
for m in done:
    with open(os.path.join(partial_dir, f"M={m}.json")) as f:
        entry = json.load(f)
    merged[str(m)] = common_utils.sort_config(entry["config"])
    how = " (quick check)" if entry.get("quick_check") else ""
    print(f"M={m}: {entry['time_us']:.1f} us {entry['config']}{how}")
with open(os.path.join(partial_dir, "filename.txt")) as f:
    out_name = os.path.basename(f.read().strip())
tmp_name = out_name + ".tmp"
with open(tmp_name, "w") as f:
    json.dump(merged, f, indent=4)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_name, out_name)
print(f"tune_moe_gfx1151: wrote {out_name} with {len(merged)} batch sizes", flush=True)
