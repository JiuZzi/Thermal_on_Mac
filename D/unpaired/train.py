"""Formal unpaired training: actual TIR, independent RGB, cycles and structure."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from importlib import metadata
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import torch

from D.common import ROOT, DEFAULT_CACHE, EDGE_SOURCES, read_json, write_json, sha256
from D.models import load_pipeline
from .protocol import (DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_RUNS, DEFAULT_PRIOR, DEFAULT_MANIFEST,
                       config_from, model_config, seed_runtime, IndexStream)
from .semantics import ConditionBuilder, RGBTeacher, default_semantic_checkpoint, legacy_models
from .data import ConditionDataset, data_identity, on_device
from .model import FormalGenerator
from .engine import UnpairedEngine, NumericalFailure, auxiliary_networks
from . import checkpoint as checkpoints
from .progress import TrainingLog, TrainingTimer, cuda_memory_snapshot, duration, event


NUMERICAL_REVISION = "20261029-finite-edge-logging-v2"
RECOVERY_SOURCES = "D/unpaired/recovery_sources.json"
AUDITED_PATCH_FILES = {
    "D/adaptation.py", "D/unpaired/checkpoint.py", "D/unpaired/data.py",
    "D/unpaired/engine.py", "D/unpaired/losses.py", "D/unpaired/model.py",
    "D/unpaired/protocol.py", "D/unpaired/report.py", "D/unpaired/semantics.py",
    "D/unpaired/train.py", "pytorch-CycleGAN-and-pix2pix/models/networks.py",
}
# Only this audited legacy implementation may migrate across changed source.
# Every other file, data/teacher/prior identity and optimization horizon stays fixed.
LEGACY_NUMERICAL_FILES = {
    "D/unpaired/checkpoint.py": "09c11b3bc022efbbb01f6d48c0b304fc345cc87927f4cc512dd675a96265c7aa",
    "D/unpaired/engine.py": "28af6e1f9d5e0d16b0e2745d02ca0bdeb36b0cccb9dc7356c11b28155da06451",
    "D/unpaired/losses.py": "35c5fe70d8e32ef23daf7b7ddc00d28e89b7fc05720b5bf32df73bc3bacf39f9",
    "D/unpaired/model.py": "7eee1b1982832eb362269b53a2e3b8ad1f5524c5826b5d7dd8bad66df1e8afd6",
    "D/unpaired/train.py": "13a307d36431061fe95cea1319bd33ed948460e83795bb59fb799b82d078c598",
}


def validate_recovery(parent, identity):
    if parent.get("format") != "D_unpaired_run_v1":
        raise ValueError("Full recovery requires a formal D run")
    for key in ("config", "data", "prior_sha256", "versions", "initialization"):
        if parent[key] != identity[key]:
            raise ValueError(f"Full recovery must preserve {key}; use a new warm-start experiment for intentional changes")
    for key in ("steps", "save_every", "log_every", "preview_every"):
        if parent["schedule"][key] != identity["schedule"][key]:
            raise ValueError(f"Full recovery must preserve original {key}")
    old, current = parent["code_sha256"], identity["code_sha256"]
    added = set(current) - set(old)
    if set(old) - set(current) or added - {RECOVERY_SOURCES}:
        raise ValueError("Unexpected source files in recovery")
    # Match a whole audited revision, never arbitrary per-file combinations.
    # Legacy runs may contain Windows CRLF hashes; new runs normalize line endings.
    audited = read_json(ROOT / RECOVERY_SOURCES)
    for revision, manifest in audited["revisions"].items():
        files = manifest["files"]
        if set(old) == set(files) and all(old[name] in files[name].values() for name in old):
            for name in old:
                if current[name] != files[name]["lf"] and name not in AUDITED_PATCH_FILES:
                    raise ValueError(f"Unaudited source change during full recovery: {name}")
            if (parent["config"].get("format") == "D_unpaired_v2" and
                parent["config"].get("chroma_smooth_policy") == "quality_soft"):
                raise ValueError("This legacy V2 run did not apply quality_soft; use an explicit new weight-only warm start for the changed objective")
            return
    for name in old:
        if old[name] != current[name] and LEGACY_NUMERICAL_FILES.get(name) != old[name]:
            raise ValueError(f"Unaudited source change during full recovery: {name}")


def _copy_recovery_baseline(parent, folder, cache_sha):
    source = parent / "previews/step_000000"
    target = folder / "previews/step_000000"
    if target.exists() or not (source / "status.json").is_file():
        return
    if (read_json(source / "status.json").get("phase") != "completed" or
        read_json(source / "prediction.json")["condition_manifest_sha256"] != cache_sha):
        raise ValueError("Recovery baseline is incomplete or uses different conditions")
    for name, record in read_json(source / "records.json").items():
        if sha256(source / "images" / f"{name}_fake_B.png") != record["sha256"]:
            raise ValueError("Original baseline image changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    write_json(folder / "baseline_origin.json", dict(run_sha256=sha256(parent / "run.json"),
               source=Path(os.path.relpath(source, ROOT)).as_posix()))


def run_updates(args, folder, engine, streams, tir_data, rgb_data, step, val_data=None, *, log=None):
    """Retry the same batch; only committed updates advance the counter."""
    if log is None:
        with TrainingLog(folder, details=getattr(args, "log_details", False)) as owned_log:
            return run_updates(args, folder, engine, streams, tir_data, rgb_data, step, val_data, log=owned_log)
    from .report import preview
    baseline = folder / "previews/step_000000"
    start_step = step
    timer = TrainingTimer(start_step, args.steps)
    stop_at = min(args.steps, step + args.verify_updates) if args.verify_updates else args.steps
    pending = None
    last_event = None
    try:
        while step < stop_at:
            pending = dict(streams={k:v.state() for k,v in streams.items()},
                           rng=checkpoints.capture_rng(), engine=engine.retry_state())
            tir = on_device(tir_data[streams["TIR"].next()], engine.device)
            rgb = on_device(rgb_data[streams["RGB"].next()], engine.device)
            logging = step == start_step or (step+1) % args.log_every == 0 or step+1 == stop_at
            # A rejected attempt restores engine RNG/pools and halves AMP scale.
            # Do not sample another image or another noise realization here.
            while True:
                applied, values = engine.update(tir, rgb, step, args.steps, diagnostics=logging)
                if applied:
                    break
                last_event = dict(step=step, TIR=tir["frame"]["filename"], RGB=rgb["frame"]["filename"], **values)
                with (folder / "numerical_events.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(last_event, allow_nan=False)+"\n")
                log.event("RETRY", f"update {step+1} | {values.get('stage', 'unknown')}"
                          f" | attempt {values.get('amp_overflow')}"
                          f" | AMP {values.get('amp_scale_before')} -> {values.get('amp_scale_after')}"
                          " | retrying the same batch/noise; details: numerical_events.jsonl")
            step += 1
            pending = None
            engine.phase = "ready"
            write_json(folder / "status.json", dict(phase="training", step=step, total=args.steps))
            if logging:
                row = {key:float(value) if torch.is_tensor(value) else value for key,value in values.items()}
                row.update(timer.snapshot(step))
                if str(engine.device).startswith("cuda"):
                    row.update(cuda_memory_snapshot(engine.device))
                row.update(step=step, total_steps=args.steps, epoch_equivalent=step/max(len(tir_data),len(rgb_data)),
                           lr_diffusion=engine.opt_g.param_groups[0]["lr"])
                log.metrics(row)
            if step % args.save_every == 0 or step == args.steps:
                tick = time.perf_counter()
                log.event("SAVE", f"Writing complete checkpoint at update {step}...")
                log.event("SAVED", f"{checkpoints.save(folder, step, engine, streams)} | {duration(time.perf_counter()-tick)}")
            elif step % args.recovery_every == 0 or step == stop_at:
                tick = time.perf_counter()
                log.event("SAVE", f"Writing rolling recovery at update {step}...")
                log.event("RECOVER", f"{checkpoints.save(folder, step, engine, streams, kind='recovery')} | {duration(time.perf_counter()-tick)}")
            if val_data is not None and args.preview_every and (step % args.preview_every == 0 or step == args.steps):
                preview(engine.generator, val_data, folder / "previews" / f"step_{step:06d}",
                        baseline=baseline if (baseline / "status.json").is_file() else None,
                        metadata=dict(run_sha256=sha256(folder / "run.json"), step=step), log=log)
    except (NumericalFailure, KeyboardInterrupt) as error:
        report = dict(step=step, total=args.steps, reason=str(error),
                      numerical=getattr(error,"report",None), previous_event=last_event)
        safe = engine.phase not in ("commit", "committed")
        if pending is not None and safe:
            for key, state in pending["streams"].items():
                streams[key].restore(state)
            engine.restore_retry(pending["engine"])
            checkpoints.restore_rng(pending["rng"])
            engine.opt_g.zero_grad(set_to_none=True); engine.opt_d.zero_grad(set_to_none=True)
        if safe:
            # NumericalFailure is raised before BOTH optimizer commits.
            try:
                saved = checkpoints.save(folder, step, engine, streams, kind="emergency", metadata=report)
                report["emergency_checkpoint"] = saved.relative_to(folder).as_posix()
                log.event("RECOVER", f"Emergency recovery saved: {saved}")
            except Exception as saving_error:
                report["recovery_save_error"] = f"{type(saving_error).__name__}: {saving_error}"
        try:
            write_json(folder / "failure.json", report)
            write_json(folder / "status.json", dict(phase="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                       step=step, total=args.steps, recovery=report.get("emergency_checkpoint")))
        except OSError:
            pass  # A full disk must not mask the original error.
        raise
    except Exception as error:
        # Unknown errors may occur during an optimizer commit. Keep the last
        # verified periodic checkpoint instead of serializing partial state.
        try:
            write_json(folder / "status.json", dict(phase="failed", step=step, total=args.steps, reason=str(error)))
        except OSError:
            pass
        raise
    phase = "completed" if step == args.steps else "verified"
    write_json(folder / "status.json", dict(phase=phase, step=step, total=args.steps))
    if phase == "verified":
        log.event("VERIFIED", f"Bounded recovery verification finished at {step}/{args.steps}; --resume continues")
    else:
        log.event("DONE", f"{step}/{args.steps} updates | training session {duration(timer.snapshot(step)['elapsed'])}")
    log.event("OUTPUT", folder / "previews" / f"step_{step:06d}" / "index.html"
              if val_data is not None and step == args.steps else folder / "latest.json")
    return step


def resolve_rgb_init(path):
    if path is None:
        return None
    if (path / "latest.json").is_file():
        path = path / "checkpoints" / f"step_{read_json(path / 'latest.json')['step']:06d}"
    header = read_json(path / "checkpoint.json")
    if sha256(path / "adapter.safetensors") != header["adapter_sha256"]:
        raise ValueError("RGB initialization changed")
    original = path.parent.parent / "run.json"
    if sha256(original) != header["run_sha256"] or read_json(original)["rgb_source"]["contour_variant"] != "clarity_v1":
        raise ValueError("RGB initialization must be a verified clarity_v1 warm-up")
    return path.resolve()


def source_hashes():
    names = ("__init__.py", "protocol.py", "model.py", "losses.py", "data.py", "semantics.py",
             "engine.py", "checkpoint.py", "report.py", "train.py")
    paths = [*[ROOT / "D/unpaired" / name for name in names], ROOT / "D/common.py",
             ROOT / "D/models.py", ROOT / "D/adaptation.py", ROOT / "D/prepare_rgb.py"]
    paths += [ROOT / name for name in (*EDGE_SOURCES,
              "pytorch-CycleGAN-and-pix2pix/models/learned_color_prior.py",
              "pytorch-CycleGAN-and-pix2pix/models/networks.py",
              "pytorch-CycleGAN-and-pix2pix/data/grouped_unaligned_dataset.py")]
    paths.append(ROOT / RECOVERY_SOURCES)
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(paths)}


def execute(args):
    config = config_from(args.config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in (".", ".."):
        raise ValueError("Use one safe run name")
    if min(args.steps, args.save_every, args.log_every, args.recovery_every) < 1 or min(args.preview_every,args.verify_updates) < 0:
        raise ValueError("Invalid update/preview intervals")
    tir_data = ConditionDataset(args.data_cache, "TIR", "train")
    rgb_data = ConditionDataset(args.data_cache, "RGB", "train")
    val_data = ConditionDataset(args.data_cache, "TIR", "val") if args.preview_every else None
    data = data_identity(args.data_cache)
    recipe = data["source"]["recipe"]
    if recipe["size"] != [config["width"], config["height"]] or recipe["flip_reliability"] != config["flip_reliability"]:
        raise ValueError("Cache geometry/reliability differs from the model config")
    if sha256(args.semantic_checkpoint) != recipe["tir_semantic_sha256"]:
        raise ValueError("Use the same FMB weights as condition preparation")
    if sha256(args.data_manifest) != data["source"]["source_manifest_sha256"]:
        raise ValueError("Original data manifest differs from cache")
    legacy_models()
    from data.grouped_unaligned_dataset import validate_prior_partitions
    validate_prior_partitions(read_json(args.data_manifest), args.data_manifest.parent, args.prior)
    folder = DEFAULT_RUNS / args.name
    previous = read_json(folder / "run.json") if args.resume else None
    if args.resume and (args.init_rgb_run or args.init_tir_checkpoint or args.recover_from_checkpoint):
        raise ValueError("Resume restores this run; do not supply a new initialization")
    rgb_init = resolve_rgb_init(args.init_rgb_run)
    initialization = dict(kind="pretrained edge-control conversion")
    if rgb_init is not None:
        initialization = dict(kind="RGB E-only conversion", checkpoint=Path(os.path.relpath(rgb_init, ROOT)).as_posix(),
                              sha256=sha256(rgb_init / "adapter.safetensors"))
    if args.init_tir_checkpoint:
        if args.init_rgb_run:
            raise ValueError("Choose one initialization")
        old_folder = args.init_tir_checkpoint.parent.parent
        header = checkpoints.verify(old_folder, args.init_tir_checkpoint)
        initialization = dict(kind="formal D warm start; optimizer/RNG/pools reset",
                              checkpoint=Path(os.path.relpath(args.init_tir_checkpoint.resolve(),ROOT)).as_posix(), files=header["files"])
    if previous:
        initialization = previous["initialization"]
    recovery_path = args.recover_from_checkpoint
    parent = None
    recovery = previous.get("recovery") if previous else None
    if recovery_path is not None:
        if args.init_rgb_run or args.init_tir_checkpoint:
            raise ValueError("Full-state recovery and weight-only initialization are different; choose one")
        recovery_path = recovery_path.resolve()
        parent = recovery_path.parent.parent
        header = checkpoints.verify(parent,recovery_path)
        parent_identity = read_json(parent / "run.json")
        initialization = parent_identity["initialization"]
        recovery = dict(checkpoint=Path(os.path.relpath(recovery_path,ROOT)).as_posix(),
                        parent_run_sha256=sha256(parent / "run.json"), files=header["files"],
                        step=header["step"], revision=NUMERICAL_REVISION,
                        restored="networks,optimizers,AMP,samplers,replay pools,all RNG; global step retained")
    identity = dict(format="D_unpaired_run_v1", config=config, data=data,
                    prior_sha256=sha256(args.prior), initialization=initialization,
                    schedule=dict(steps=args.steps, save_every=args.save_every, log_every=args.log_every,
                                  preview_every=args.preview_every, recovery_every=args.recovery_every), code_sha256=source_hashes(),
                    numerical_revision=NUMERICAL_REVISION, recovery=recovery,
                    versions={k:metadata.version(k) for k in ("torch", "torchvision", "diffusers", "transformers",
                              "numpy", "scipy", "scikit-image", "Pillow", "safetensors")},
                    limits="Unpaired research model; no guarantee of true RGB colors or exact geometry; "
                           "seed42; unsupported deterministic CUDA kernels warn; pretrained RGB data is an extra prior")
    if recovery_path is not None:
        validate_recovery(parent_identity,identity)
        if recovery["step"] > args.steps:
            raise ValueError("Recovery checkpoint exceeds the original training horizon")
    if args.check_only:
        event("CHECK", f"passed | TIR={len(tir_data)} RGB={len(rgb_data)} | input=[TIR,E,0]")
        event("CONFIG", f"{config['format']} | {config['width']}x{config['height']} | seed=42"
              f" | DDIM={int(config['sampling_steps']*config['strength'])} actual steps | target={args.steps} updates")
        event("INIT", initialization["kind"])
        if getattr(args, "log_details", False):
            event("DETAIL", json.dumps(dict(config=config, initialization=initialization), ensure_ascii=True))
        return
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    if args.resume:
        if previous != identity:
            raise ValueError("Resume must preserve data, code, priors, model and schedule; use an explicit new warm-start run for changes")
    else:
        if folder.exists():
            raise ValueError("Preserve existing run; choose another name or --resume")
        folder.mkdir(parents=True)
        write_json(folder / "run.json", identity)
    from filelock import FileLock
    with FileLock(str(folder / "training.lock"), timeout=0), TrainingLog(folder, details=getattr(args, "log_details", False)) as log:
        seed_runtime()
        latest = folder / "latest.json"
        formal_init = args.init_tir_checkpoint
        if args.resume and not latest.exists() and recovery is not None:
            recovery_path = ROOT / recovery["checkpoint"].replace("\\", "/")
            parent = recovery_path.parent.parent
            header = checkpoints.verify(parent,recovery_path)
            if sha256(parent / "run.json") != recovery["parent_run_sha256"] or header["files"] != recovery["files"]:
                raise ValueError("Recorded recovery source changed")
            validate_recovery(read_json(parent / "run.json"),identity)
        if args.resume and not latest.exists():
            log.event("RESUME", "No completed update checkpoint; restarting recorded initialization")
            if recovery is not None:
                pass  # Restore the complete optimizer/RNG state below, not the warm-up.
            elif initialization["kind"] == "RGB E-only conversion":
                rgb_init = ROOT / initialization["checkpoint"].replace("\\", "/")
                if (resolve_rgb_init(rgb_init) != rgb_init.resolve() or
                    sha256(rgb_init / "adapter.safetensors") != initialization["sha256"]):
                    raise ValueError("Recorded RGB initialization changed")
            elif initialization["kind"].startswith("formal D warm start"):
                formal_init = ROOT / initialization["checkpoint"].replace("\\", "/")
                original = checkpoints.verify(formal_init.parent.parent, formal_init)
                if original["files"] != initialization["files"]:
                    raise ValueError("Recorded formal initialization changed")
        log.event("LOAD", "Loading diffusion weights and frozen semantic models...")
        pipeline = load_pipeline(model_config(config), args.cache_dir, offline=True)
        generator = FormalGenerator(pipeline, config, args.prior, recipe,
                                    rgb_init=rgb_init if recovery is None and (not args.resume or not latest.exists()) else None).to("cuda:0")
        del pipeline
        reverse, critics = auxiliary_networks("cuda:0")
        if formal_init:
            from safetensors.torch import load_file
            generator.import_state(load_file(str(formal_init / "generator.safetensors")))
            reverse.load_state_dict(load_file(str(formal_init / "reverse.safetensors")), strict=True)
            critics.load_state_dict(load_file(str(formal_init / "critics.safetensors")), strict=True)
        builder = ConditionBuilder(args.semantic_checkpoint, rgb=False, flip=config["flip_reliability"]) \
                  if config["loss_weights"]["cycle_rgb"] else None
        teacher = RGBTeacher().to(config.get("semantic_loss_device", "cuda:0")) if config["loss_weights"]["semantic"] else None
        if teacher is not None and teacher.info != recipe["rgb_teacher"]:
            raise ValueError("RGB teacher differs from cache")
        engine = UnpairedEngine(generator, reverse, critics, builder, teacher, config)
        streams = dict(TIR=IndexStream(len(tir_data), "TIR"), RGB=IndexStream(len(rgb_data), "RGB"))
        if args.resume and latest.exists():
            location,_ = checkpoints.locate(folder)
            step = checkpoints.restore(folder,location,engine,streams)
        elif recovery_path is not None:
            step = checkpoints.restore(parent,recovery_path,engine,streams)
            checkpoints.save(folder,step,engine,streams,kind="recovery",metadata=dict(recovery=recovery))
            if val_data is not None:
                _copy_recovery_baseline(parent,folder,data["manifest_sha256"])
        else:
            step = 0
        trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        log.event("MODEL", f"{torch.cuda.get_device_name(0)} | trainable diffusion={trainable:,}"
                  f" | DDIM={int(config['sampling_steps']*config['strength'])} actual steps")
        log.event("DATA", f"TIR={len(tir_data)} RGB={len(rgb_data)} | start={step} target={args.steps} updates")
        log.event("METRICS", "G is weighted total; cycle/AUX are raw losses; replay=off means not scheduled")
        baseline = folder / "previews/step_000000"
        from .report import preview
        if val_data is not None and step == 0:
            preview(generator, val_data, baseline, resume=baseline.exists(),
                    metadata=dict(run_sha256=sha256(folder / "run.json"), step=0), log=log)
        run_updates(args,folder,engine,streams,tir_data,rgb_data,step,val_data,log=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="D_tir_unpaired_clarity_seed42")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--semantic-checkpoint", type=Path, default=default_semantic_checkpoint())
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-details", action="store_true", help="Print all numeric diagnostics as well; always recorded in loss.jsonl")
    parser.add_argument("--preview-every", type=int, default=250)
    parser.add_argument("--recovery-every", type=int, default=100, help="Keep the last two full recovery snapshots, in addition to archival epochs")
    parser.add_argument("--verify-updates", type=int, default=0, help="Stop after N committed updates without changing the original training horizon; --resume continues")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-rgb-run", type=Path)
    parser.add_argument("--init-tir-checkpoint", type=Path)
    parser.add_argument("--recover-from-checkpoint", type=Path, help="New run with an audited numerical fix; retain full optimizer/data/RNG state and original step")
    parser.add_argument("--check-only", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
