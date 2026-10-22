#!/usr/bin/env python3
"""Presharded dump/reload with a host-resident PLE table (patch 31).

`--load-format presharded` dumps the post-processed model state on the
first boot and copies it straight back on later ones, which turns the
Qwen3.8 load (222k small tensors, 152 s) into a few thousand full-layer
copies bounded by NVMe speed. Three things on this model break it:

  1. The PLE n-gram table is a `Parameter` (patch 11, `Qwen4ExpPinnedHost
     Embedding.weight`) backed by a 47.7 GB file mmap on the host. The dump
     would hash it, write it a second time and, on reload, copy it back
     into the mmap. Tensors that do not live on the target device are left
     out of the dump and of the reload's missing-key check; the table has
     its own machinery (patch 13).
  2. That machinery only runs from `load_weights`, which the reload never
     calls, so a missing or incomplete table file would be served as
     garbage. Before reloading, the model's `weight_files_to_skip` hook is
     invoked with the checkpoint's files (this is where patch 13 checks the
     completion marker); if the model has file-backed tables and did not
     confirm them complete, the loader falls back to the normal load, which
     fills the table and refreshes the dump.
  3. `load_weights` ends with `finalize_fused_in_proj()` on every GDN
     module (in_proj_qkvz + in_proj_ba stacked into one GEMM weight; the
     parameters become row views). The reload runs it after the copy so
     the served kernels and numerics match the normal path.

And two things about the cache directory itself:

  4. The dump's key covers quantization, dtype, parallel layout and the
     parameters' shapes, not the code that produced the values. A rebuilt
     image whose patches change a weight's post-processing (patch 12's zero
     points, say) would reuse the old dump and serve wrong numerics without
     a word. `SGLANG_PRESHARDED_STAMP` (the launcher passes the image ID)
     joins the key, so every rebuild gets its own subfolder.
  5. Before choosing a path, the cache root is audited: the current
     subfolder without its READY marker is an interrupted dump and is
     removed so the redo starts clean (the writer would otherwise leave
     orphans from a differently planned file layout); any other subfolder,
     stale or another deployment's, is reported with its size and
     completeness and left alone.
  6. The load walks 222k tensors, the dump hashes every tensor and writes
     ~70 GB of files, and a later boot copies them back, all without a line
     in the log. Each phase now logs its size up front, a progress line
     every 15 s (items, GiB, rate, estimated time left) and a summary at the
     end. The load wrapper is in DefaultModelLoader._get_weights_iterator,
     so every load format gets it. Upstream's tqdm shard bars, which redraw
     with carriage returns and turn the surrounding log into "[N blob
     data]" under journald, are kept only when stderr is a TTY.

Every anchor is asserted; an upstream rewrite fails the build.
See patches/31-presharded-host-tables.md.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/model_loader/loader.py"
text = open(p).read()

# 1a. dump: leave host-resident tensors out
old = """            state_dict = dict(model.state_dict())
            extras = self._collect_extra_tensors(model)
"""
assert text.count(old) == 1, "loader.py: presharded dump state_dict anchor not found"
new = """            state_dict = dict(model.state_dict())
            extras = self._collect_extra_tensors(model)
            # gfx1151 patch 31: tensors off the target device (the file-backed
            # PLE table, 47.7 GB on the host) are not dumped; they are rebuilt
            # or reused by their own machinery.
            state_dict = {
                k: t for k, t in state_dict.items() if t.device.type == target_device.type
            }
            extras = {
                k: t for k, t in extras.items() if t.device.type == target_device.type
            }
"""
text = text.replace(old, new, 1)

# 1b. reload: host-resident tensors are not missing
old = """            for k, t in state_dict.items():
                if k in loaded_param_keys:
                    continue
                if t.numel() == 0:
                    continue
                storage_key = (t.device, t.untyped_storage().data_ptr())
"""
assert text.count(old) == 1, "loader.py: presharded missing-key anchor not found"
new = """            for k, t in state_dict.items():
                if k in loaded_param_keys:
                    continue
                if t.numel() == 0:
                    continue
                if t.device.type != target_device.type:
                    continue  # gfx1151 patch 31: host tables are not dumped
                storage_key = (t.device, t.untyped_storage().data_ptr())
"""
text = text.replace(old, new, 1)

# 2. reload: confirm host tables before trusting the dump; 3. post-load fusion
old = """            rank, _ = self._world_rank_and_size()
            with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME)) as f:
                plan = json.load(f)
"""
assert text.count(old) == 1, "loader.py: presharded reload plan anchor not found"
new = """            # gfx1151 patch 31: the dump holds no host tables, so the model's
            # own check (patch 13's completion marker, run from the
            # weight_files_to_skip hook) has to pass before the dump is used.
            if not self._host_tables_ready(model_config, model):
                logger.warning(
                    "Presharded reload at %s: the model's host table is not "
                    "complete on disk; loading the checkpoint normally and "
                    "refreshing the dump.",
                    presharded_dir,
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._first_time_load_and_dump(
                    model_config,
                    device_config,
                    presharded_dir,
                    self._collect_shard_config(model_config),
                )

            rank, _ = self._world_rank_and_size()
            with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME)) as f:
                plan = json.load(f)
"""
text = text.replace(old, new, 1)

old = """            self._rebind_parameter_aliases(model)
"""
assert text.count(old) == 1, "loader.py: presharded _rebind_parameter_aliases anchor not found"
new = """            self._rebind_parameter_aliases(model)
            # gfx1151 patch 31: load_weights is not called on this path; run
            # the fusion it would have ended with (Qwen3.5-family GDN).
            for module in model.modules():
                finalize = getattr(module, "finalize_fused_in_proj", None)
                if finalize is not None:
                    finalize()
"""
text = text.replace(old, new, 1)

# helper next to _presharded_ready
old = """    @classmethod
    def _presharded_ready(cls, presharded_dir: str) -> bool:
        return os.path.isfile(os.path.join(presharded_dir, cls.READY_FILENAME))
"""
assert text.count(old) == 1, "loader.py: _presharded_ready anchor not found"
new = old + """
    def _host_tables_ready(self, model_config: ModelConfig, model: nn.Module) -> bool:
        \"\"\"gfx1151 patch 31: True unless the model keeps file-backed host
        tables that its weight_files_to_skip hook did not confirm complete.\"\"\"
        hook = getattr(model, "weight_files_to_skip", None)
        tables = getattr(model, "_ple_file_tables", None)
        if hook is None or tables is None or not tables():
            return True
        source = DefaultModelLoader.Source.init_new(model_config, model)
        hf_folder, weight_files, _ = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        hook(hf_folder, list(weight_files))
        return bool(getattr(model, "_ple_table_reused", False))

    def _audit_presharded_root(self, presharded_dir: str) -> None:
        \"\"\"gfx1151 patch 31: drop an interrupted dump of this configuration;
        report, but keep, every other subfolder under the same root.\"\"\"
        rank, _ = self._world_rank_and_size()
        root, current = os.path.split(presharded_dir)
        if os.path.isdir(presharded_dir) and not self._presharded_ready(presharded_dir):
            if rank == 0:
                logger.warning(
                    "Presharded dump at %s has no %s marker (interrupted dump); "
                    "removing it before dumping again.",
                    presharded_dir,
                    self.READY_FILENAME,
                )
                shutil.rmtree(presharded_dir, ignore_errors=True)
            self._world_barrier()
        if rank != 0 or not os.path.isdir(root):
            return
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name == current or not os.path.isdir(path):
                continue
            size = 0
            for dirpath, _, files in os.walk(path):
                for fn in files:
                    with suppress(OSError):
                        size += os.path.getsize(os.path.join(dirpath, fn))
            logger.warning(
                "Presharded cache root %s also holds %s (%s, %.1f GiB). It does "
                "not match this model/image/configuration and is left alone; "
                "remove it by hand if no other deployment uses it.",
                root,
                name,
                "complete" if self._presharded_ready(path) else "incomplete",
                size / 2**30,
            )
"""
text = text.replace(old, new, 1)

# 4. code stamp in the cache key (only when the launcher sets it)
old = """            "structural_signature": self._compute_structural_signature(model_config),
        }
"""
assert text.count(old) == 1, "loader.py: _collect_shard_config anchor not found"
new = """            "structural_signature": self._compute_structural_signature(model_config),
            # gfx1151 patch 31: the code that post-processed the weights is
            # part of the key; the launcher passes the image ID.
            **(
                {"code_stamp": os.environ["SGLANG_PRESHARDED_STAMP"]}
                if os.environ.get("SGLANG_PRESHARDED_STAMP")
                else {}
            ),
        }
"""
text = text.replace(old, new, 1)

# 5. audit the cache root before choosing dump or reload
old = """        shard_config = self._collect_shard_config(model_config)
        presharded_dir = self._presharded_dir(model_config, shard_config)
        if self._presharded_ready(presharded_dir) and self._shard_config_matches(
"""
assert text.count(old) == 1, "loader.py: presharded load_model anchor not found"
new = """        shard_config = self._collect_shard_config(model_config)
        presharded_dir = self._presharded_dir(model_config, shard_config)
        self._audit_presharded_root(presharded_dir)  # gfx1151 patch 31
        if self._presharded_ready(presharded_dir) and self._shard_config_matches(
"""
text = text.replace(old, new, 1)

# 6. progress lines for the dump's hash and write phases and the reload's copy
old = """class PreshardedModelLoader(DefaultModelLoader):
    \"\"\"Dump/reload post-process weights under ``<model_path>/presharded/<subdir>/``."""
assert text.count(old) == 1, "loader.py: PreshardedModelLoader class anchor not found"
new = """class _LoadProgress:
    \"\"\"gfx1151 patch 31: periodic progress for the weight load, the presharded
    dump and the reload, which otherwise run for minutes (222k tensors,
    hashing and writing ~70 GB) without a line in the log. One line at the
    start, one every `every_s` seconds with rate and estimated time left, one
    at the end. total_items is None when only the byte total is known.\"\"\"

    def __init__(
        self,
        phase: str,
        total_bytes: int,
        total_items: Optional[int],
        unit: str = "items",
        every_s: float = 15.0,
    ):
        self.phase = phase
        self.total_bytes = total_bytes
        self.total_items = total_items
        self.unit = unit
        self.every_s = every_s
        self.started = self.last = time.perf_counter()
        self.done_bytes = 0
        self.done_items = 0
        logger.info(
            "%s: %s%.1f GiB",
            phase,
            f"{total_items} {unit}, " if total_items is not None else "",
            total_bytes / 2**30,
        )

    def _items(self) -> str:
        if self.total_items is None:
            return f"{self.done_items} {self.unit}"
        return f"{self.done_items}/{self.total_items} {self.unit}"

    def step(self, nbytes: int) -> None:
        self.done_bytes += nbytes
        self.done_items += 1
        now = time.perf_counter()
        if now - self.last < self.every_s:
            return
        if self.total_items is not None and self.done_items >= self.total_items:
            return
        self.last = now
        elapsed = now - self.started
        rate = self.done_bytes / elapsed if elapsed > 0 else 0.0
        left = (self.total_bytes - self.done_bytes) / rate if rate > 0 else float("inf")
        logger.info(
            "%s: %s, %.1f/%.1f GiB (%.0f%%), %.0f MiB/s, %.0f s elapsed, ~%.0f s left",
            self.phase,
            self._items(),
            self.done_bytes / 2**30,
            self.total_bytes / 2**30,
            100.0 * self.done_bytes / self.total_bytes if self.total_bytes else 100.0,
            rate / 2**20,
            elapsed,
            max(left, 0.0),
        )

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.started
        rate = self.done_bytes / elapsed if elapsed > 0 else 0.0
        logger.info(
            "%s: done, %s, %.1f GiB in %.0f s (%.0f MiB/s)",
            self.phase,
            self._items(),
            self.done_bytes / 2**30,
            elapsed,
            rate / 2**20,
        )


def _progress_weights_iterator(weights_iterator, weight_files):
    \"\"\"gfx1151 patch 31: progress lines while the model consumes the weights.
    The byte total is the shard files' size; a filtered iterator (the MTP
    draft keeps a few tensors of each file) ends below 100%.\"\"\"
    total = 0
    for f in weight_files:
        with suppress(OSError):
            total += os.path.getsize(f)
    progress = _LoadProgress(
        f"Weights load ({len(weight_files)} shard files)", total, None, unit="tensors"
    )
    for name, tensor in weights_iterator:
        progress.step(tensor.numel() * tensor.element_size())
        yield name, tensor
    progress.finish()


class PreshardedModelLoader(DefaultModelLoader):
    \"\"\"Dump/reload post-process weights under ``<model_path>/presharded/<subdir>/``."""
text = text.replace(old, new, 1)

old = """        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
"""
assert text.count(old) == 1, "loader.py: _get_weights_iterator return anchor not found"
new = """        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        weights_iterator = (
            (source.prefix + name, tensor) for (name, tensor) in weights_iterator
        )
        return _progress_weights_iterator(weights_iterator, hf_weights_files)  # gfx1151 patch 31
"""
text = text.replace(old, new, 1)

old = """        manifest: Dict[str, Dict[str, Any]] = {}
        num_workers = min(max(1, len(items)), self._hash_num_threads)
        if num_workers <= 1:
            for it in items:
                name, info = _entry(it)
                manifest[name] = info
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="presharded-hash",
            ) as ex:
                for name, info in ex.map(_entry, items):
                    manifest[name] = info
"""
assert text.count(old) == 1, "loader.py: presharded dump hash loop anchor not found"
new = """        manifest: Dict[str, Dict[str, Any]] = {}
        num_workers = min(max(1, len(items)), self._hash_num_threads)
        progress = _LoadProgress(  # gfx1151 patch 31
            f"Presharded dump rank {rank}, hashing ({num_workers} threads)",
            sum(t.numel() * t.element_size() for _, t, _ in items),
            len(items),
        )
        if num_workers <= 1:
            for it in items:
                name, info = _entry(it)
                manifest[name] = info
                progress.step(info["size"])
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="presharded-hash",
            ) as ex:
                for name, info in ex.map(_entry, items):
                    manifest[name] = info
                    progress.step(info["size"])
        progress.finish()
"""
text = text.replace(old, new, 1)

old = """        from safetensors.torch import save_file

        for f in plan["files"]:
            if f["writer_rank"] != rank:
                continue
            tensors_to_save: Dict[str, torch.Tensor] = {}
"""
assert text.count(old) == 1, "loader.py: _dump_files_for_rank loop anchor not found"
new = """        from safetensors.torch import save_file

        mine = [f for f in plan["files"] if f["writer_rank"] == rank]
        progress = _LoadProgress(  # gfx1151 patch 31
            f"Presharded dump rank {rank}, writing to {presharded_dir}",
            sum(t["size"] for f in mine for t in f["tensors"]),
            len(mine),
        )
        for f in mine:
            tensors_to_save: Dict[str, torch.Tensor] = {}
"""
text = text.replace(old, new, 1)

old = """                tensors_to_save[t["stored_key"]] = tensor
            save_file(tensors_to_save, os.path.join(presharded_dir, f["filename"]))
"""
assert text.count(old) == 1, "loader.py: _dump_files_for_rank save_file anchor not found"
new = """                tensors_to_save[t["stored_key"]] = tensor
            save_file(tensors_to_save, os.path.join(presharded_dir, f["filename"]))
            progress.step(sum(t["size"] for t in f["tensors"]))
        progress.finish()
"""
text = text.replace(old, new, 1)

old = """            loaded_param_keys: set = set()
            verify_hashes: List[Tuple[str, str]] = []
            for filename, items in by_file.items():
                stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                cached = self._read_presharded_file(
                    os.path.join(presharded_dir, filename), stored_keys
                )
                self._apply_presharded_file(
                    items=items,
                    cached=cached,
                    model=model,
                    state_dict=state_dict,
                    target_device=target_device,
                    loaded_param_keys=loaded_param_keys,
                    verify_hashes=verify_hashes,
                )
"""
assert text.count(old) == 1, "loader.py: _load_from_presharded file loop anchor not found"
new = """            loaded_param_keys: set = set()
            verify_hashes: List[Tuple[str, str]] = []
            file_bytes = {  # gfx1151 patch 31
                fn: os.path.getsize(os.path.join(presharded_dir, fn)) for fn in by_file
            }
            progress = _LoadProgress(
                f"Presharded reload rank {rank}, copying from {presharded_dir}"
                + (" with verification" if self._verify_on_load else ""),
                sum(file_bytes.values()),
                len(by_file),
            )
            for filename, items in by_file.items():
                stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                cached = self._read_presharded_file(
                    os.path.join(presharded_dir, filename), stored_keys
                )
                self._apply_presharded_file(
                    items=items,
                    cached=cached,
                    model=model,
                    state_dict=state_dict,
                    target_device=target_device,
                    loaded_param_keys=loaded_param_keys,
                    verify_hashes=verify_hashes,
                )
                progress.step(file_bytes[filename])
            progress.finish()
"""
text = text.replace(old, new, 1)

open(p, "w").write(text)
print("patched", p)

# 7. weight_utils.py: upstream's tqdm shard bars redraw with carriage returns
# and never end their line. Under journald that whole stretch of the log,
# including the progress lines above, shows as "[N blob data]". Keep the bars
# for interactive runs only; the line-based progress covers the rest.
p = f"{path}/python/sglang/srt/model_loader/weight_utils.py"
text = open(p).read()

old = "import struct\n"
assert text.count(old) == 1, "weight_utils.py: import anchor not found"
text = text.replace(old, "import struct\nimport sys\n", 1)

old = """    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
"""
assert text.count(old) == 6, "weight_utils.py: enable_tqdm blocks changed"
new = """    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ) and sys.stderr.isatty()  # gfx1151 patch 31: no CR-redrawn bars in a log pipe
"""
text = text.replace(old, new)

open(p, "w").write(text)
print("patched", p)
