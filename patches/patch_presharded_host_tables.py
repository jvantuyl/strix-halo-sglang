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

open(p, "w").write(text)
print("patched", p)
