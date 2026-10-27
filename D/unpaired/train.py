"""Formal unpaired training: actual TIR, independent RGB, cycles and structure."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import time

import torch

from D.common import ROOT, DEFAULT_CACHE, EDGE_SOURCES, read_json, write_json, sha256
from D.models import load_pipeline
from .protocol import (DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_RUNS, DEFAULT_PRIOR, DEFAULT_MANIFEST,
                       config_from, model_config, seed_runtime, IndexStream)
from .semantics import ConditionBuilder, RGBTeacher, default_semantic_checkpoint, legacy_models
from .data import ConditionDataset, data_identity, on_device
from .model import FormalGenerator
from .engine import UnpairedEngine, auxiliary_networks
from . import checkpoint as checkpoints


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
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}


def execute(args):
    config = config_from(args.config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in (".", ".."):
        raise ValueError("Use one safe run name")
    if min(args.steps, args.save_every, args.log_every) < 1 or args.preview_every < 0:
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
    if args.resume and (args.init_rgb_run or args.init_tir_checkpoint):
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
    identity = dict(format="D_unpaired_run_v1", config=config, data=data,
                    prior_sha256=sha256(args.prior), initialization=initialization,
                    schedule=dict(steps=args.steps, save_every=args.save_every, log_every=args.log_every,
                                  preview_every=args.preview_every), code_sha256=source_hashes(),
                    versions={k:metadata.version(k) for k in ("torch", "torchvision", "diffusers", "transformers",
                              "numpy", "scipy", "scikit-image", "Pillow", "safetensors")},
                    limits="Unpaired research model; no guarantee of true RGB colors or exact geometry; "
                           "seed42; unsupported deterministic CUDA kernels warn; pretrained RGB data is an extra prior")
    if args.check_only:
        print(dict(TIR_train=len(tir_data), RGB_train=len(rgb_data), config=config,
                   initialization=initialization, input="TIR,E,0", data_valid=True))
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
    with FileLock(str(folder / "training.lock"), timeout=0):
        seed_runtime()
        latest = folder / "latest.json"
        formal_init = args.init_tir_checkpoint
        if args.resume and not latest.exists():
            print("No completed update checkpoint; restarting recorded initialization", flush=True)
            if initialization["kind"] == "RGB E-only conversion":
                rgb_init = ROOT / initialization["checkpoint"].replace("\\", "/")
                if (resolve_rgb_init(rgb_init) != rgb_init.resolve() or
                    sha256(rgb_init / "adapter.safetensors") != initialization["sha256"]):
                    raise ValueError("Recorded RGB initialization changed")
            elif initialization["kind"].startswith("formal D warm start"):
                formal_init = ROOT / initialization["checkpoint"].replace("\\", "/")
                original = checkpoints.verify(formal_init.parent.parent, formal_init)
                if original["files"] != initialization["files"]:
                    raise ValueError("Recorded formal initialization changed")
        pipeline = load_pipeline(model_config(config), args.cache_dir, offline=True)
        generator = FormalGenerator(pipeline, config, args.prior, recipe,
                                    rgb_init=rgb_init if not args.resume or not latest.exists() else None).to("cuda:0")
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
        step = checkpoints.restore(folder, folder / read_json(latest)["checkpoint"], engine, streams) \
               if args.resume and latest.exists() else 0
        trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(dict(TIR=len(tir_data), RGB=len(rgb_data), start_update=step, target_updates=args.steps,
                   trainable_diffusion=trainable, gpu=torch.cuda.get_device_name(0),
                   actual_denoise_steps=int(config["sampling_steps"]*config["strength"])), flush=True)
        baseline = folder / "previews/step_000000"
        from .report import preview
        if val_data is not None and step == 0:
            preview(generator, val_data, baseline, resume=baseline.exists(),
                    metadata=dict(run_sha256=sha256(folder / "run.json"), step=0))
        started = time.perf_counter()
        while step < args.steps:
            tir = on_device(tir_data[streams["TIR"].next()], "cuda:0")
            rgb = on_device(rgb_data[streams["RGB"].next()], "cuda:0")
            logging = (step+1) % args.log_every == 0 or step+1 == args.steps
            applied, values = engine.update(tir, rgb, step, args.steps, diagnostics=logging)
            if not applied:
                print(values, flush=True)
                continue
            step += 1
            write_json(folder / "status.json", dict(phase="training", step=step, total=args.steps))
            if logging:
                row = {key:float(value) if torch.is_tensor(value) else value for key,value in values.items()}
                row.update(step=step, epoch_equivalent=step/max(len(tir_data),len(rgb_data)),
                           elapsed=time.perf_counter()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                           lr_diffusion=engine.opt_g.param_groups[0]["lr"])
                with (folder / "loss.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(row)+"\n")
                print(row, flush=True)
            if step % args.save_every == 0 or step == args.steps:
                saved = checkpoints.save(folder, step, engine, streams)
                print(f"Saved {saved}",flush=True)
            if val_data is not None and (step % args.preview_every == 0 or step == args.steps):
                preview(generator, val_data, folder / "previews" / f"step_{step:06d}",
                        baseline=baseline if (baseline / "status.json").is_file() else None,
                        metadata=dict(run_sha256=sha256(folder / "run.json"), step=step))
        write_json(folder / "status.json", dict(phase="completed", step=step, total=args.steps))
        print(folder / "previews" / f"step_{step:06d}" / "index.html" if val_data is not None else folder / "latest.json", flush=True)


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
    parser.add_argument("--preview-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-rgb-run", type=Path)
    parser.add_argument("--init-tir-checkpoint", type=Path)
    parser.add_argument("--check-only", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
