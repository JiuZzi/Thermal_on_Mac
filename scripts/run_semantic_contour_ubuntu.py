#!/usr/bin/env python3
"""Preflight, train 40+40 epochs, and export the fixed 112 FLIR testB images."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "flir_v2_struct_sem512_protected_seed42_ubuntu_20261014_40_40"
EXPECTED_COUNTS = {"trainA": 6191, "trainB": 2110, "testA": 696, "testB": 112}


def preflight(root, data_root, checkpoint, device_name):
    train_root = root / "pytorch-CycleGAN-and-pix2pix"
    for path in (train_root / "train.py", train_root / "test.py",
                 root / "scripts/semantic_preview/checkpoint.py", checkpoint):
        if not path.is_file():
            raise RuntimeError(f"Missing required file: {path}")
    print(f"Python: {sys.executable}", flush=True)
    required = ("torch", "torchvision", "numpy", "PIL", "scipy", "skimage",
                "dominate", "wandb", "omegaconf")
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(
                f"Cannot import {name}: {error}\n"
                "Install scripts/requirements_semantic_train.txt in this interpreter; "
                "torch and torchvision must be a compatible CUDA pair."
            ) from error

    import numpy as np
    from PIL import Image
    import torch

    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError(
            f"PyTorch {torch.__version__} is older than the supported >=2.6 loading API. "
            "See docs/结构语义保护版Ubuntu训练.md for a separate CUDA environment."
        )
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}", flush=True)
    if device_name != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Run nvidia-smi and check the CUDA PyTorch installation.")
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}; total VRAM: {properties.total_memory / 2**30:.2f} GiB", flush=True)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        print("CPU implementation check only; this does not verify Ubuntu/CUDA training.", flush=True)

    sys.path.insert(0, str(train_root))
    sys.path.insert(0, str(root / "scripts"))
    from data.image_folder import make_dataset
    from models.semantic_contour import FrozenSemanticPrior
    from models.semantic_edges import make_structure_semantic_condition

    paths_by_split = {}
    for split, expected in EXPECTED_COUNTS.items():
        directory = data_root / split
        if not directory.is_dir():
            raise RuntimeError(f"Missing dataset split: {directory}")
        paths = [Path(path) for path in make_dataset(directory)]
        if len(paths) != expected:
            raise RuntimeError(f"{split}: {len(paths)} images; expected {expected}. Reuse the original processed FLIR split.")
        paths_by_split[split] = paths
        print(f"{split}: {len(paths)} images", flush=True)

    protocol = root / "FLIR_protocol_v2/manifest.csv"
    if protocol.is_file():
        with protocol.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for split, paths in paths_by_split.items():
            expected = {row["destination_name"] for row in rows if row["destination_split"] == split}
            actual = {path.name for path in paths}
            if actual != expected or len(actual) != len(paths):
                raise RuntimeError(f"{split}: image filenames differ from the packaged FLIR protocol. Check the dataset path.")
        print("FLIR split filenames match the packaged protocol.", flush=True)

    # Import both current CLI entrypoints without creating an experiment.
    for entrypoint in ("train.py", "test.py"):
        command = [sys.executable, entrypoint, "--model", "semantic_contour_cycle_gan", "--help"]
        result = subprocess.run(command, cwd=train_root, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"{entrypoint} import/option check failed:\n{result.stdout}\n{result.stderr}")

    print("Loading the complete frozen SegFormer weight and checking a real 256 crop...", flush=True)
    prior = FrozenSemanticPrior(checkpoint)
    manifest_path = root / "semantic_ubuntu_update_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["files"]["runs/pretrained/segformer_fmb_b3/best_model.pth"]["sha256"]
        if prior.info["sha256"] != expected_sha:
            raise RuntimeError("The frozen semantic weight differs from the update package.")
    print(f"Semantic checkpoint SHA256: {prior.info['sha256']}", flush=True)
    with Image.open(paths_by_split["trainB"][0]) as source:
        image = source.convert("L")
        if min(image.size) < 256:
            raise RuntimeError(f"Processed TIR is too small for the fixed 256 crop: {image.size}")
        left, top = (image.width - 256) // 2, (image.height - 256) // 2
        gray = np.asarray(image.crop((left, top, left + 256, top + 256)), dtype=np.float32) / 255.0
    probabilities = prior.predict(gray, torch.device(device_name))
    edge = make_structure_semantic_condition(gray, probabilities)
    if edge.shape != (256, 256) or not np.isfinite(edge).all() or np.any((edge < 0) | (edge > 1)):
        raise RuntimeError("Invalid real-image joint512 contour.")
    if device_name != "cpu":
        torch.cuda.synchronize()
    print(f"joint512: {edge.shape}, range [{edge.min():.6f}, {edge.max():.6f}]; preflight passed.", flush=True)


def run(command, directory):
    print("Running: " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal-root", type=Path, default=ROOT)
    parser.add_argument("--dataroot", type=Path, default=None, help="Default: <thermal-root>/FLIR_datasets")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help="0 = full 80 epochs; 5 = five complete epochs with the same 40+40 schedule")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true", help="Validate the environment/data/weight without training or writing checkpoints")
    group.add_argument("--test-only", action="store_true", help="Export an existing checkpoint without retraining")
    parser.add_argument("--epoch", type=int, default=80, help="Checkpoint epoch for --test-only")
    parser.add_argument("--check-device", choices=("cuda:0", "cpu"), default="cuda:0", help="cpu is only permitted with --check-only, for local implementation checks")
    args = parser.parse_args()
    if not 0 <= args.stop_after_epoch <= 80 or not 1 <= args.epoch <= 80 or args.num_threads < 0:
        parser.error("Use stop-after-epoch 0..80, epoch 1..80, and num-threads >=0.")
    if not args.name or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) is None:
        parser.error("Experiment names must be a single directory name using letters, digits, _, -, or .")
    if args.check_device == "cpu" and not args.check_only:
        parser.error("CPU is only supported for --check-only; training and test export require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("This launcher uses one CUDA device. Invoke python directly, without distributed torchrun.")

    root = args.thermal_root.expanduser().resolve()
    data_root = (args.dataroot.expanduser().resolve() if args.dataroot else root / "FLIR_datasets")
    train_root = root / "pytorch-CycleGAN-and-pix2pix"
    checkpoints = root / "runs/checkpoints"
    results = root / "runs/results"
    semantic = root / "runs/pretrained/segformer_fmb_b3/best_model.pth"
    epoch = args.epoch if args.test_only else (args.stop_after_epoch or 80)
    if not args.check_only:
        if args.test_only:
            for network in ("G_A", "G_B"):
                if not (checkpoints / args.name / f"{epoch}_net_{network}.pth").is_file():
                    parser.error(f"Missing epoch {epoch} {network} checkpoint for {args.name}")
        elif (checkpoints / args.name).exists() or (results / args.name).exists():
            parser.error("Experiment/result directory already exists. Use a new --name; this launcher starts from scratch.")
        if (results / args.name / f"test_{epoch}").exists():
            parser.error("Test output directory already exists; preserve it or choose a different experiment/epoch.")

    if args.check_only:
        preflight(root, data_root, semantic, args.check_device)
        return

    # A separate process releases the semantic preflight's CUDA context/cache
    # before the real training process claims VRAM on the 3080.
    run([sys.executable, "-u", str(Path(__file__).resolve()),
         "--thermal-root", str(root), "--dataroot", str(data_root), "--check-only"], root)

    common = [
        "--dataroot", str(data_root), "--name", args.name,
        "--checkpoints_dir", str(checkpoints),
        "--model", "semantic_contour_cycle_gan", "--semantic_checkpoint", str(semantic),
        "--condition_mode", "soft_saliency", "--fusion_mode", "direct",
        "--dataset_mode", "unaligned", "--direction", "BtoA", "--input_nc", "1", "--output_nc", "3",
        "--netG", "resnet_9blocks", "--netD", "basic", "--ngf", "64", "--ndf", "64",
        "--norm", "instance",
    ]
    if not args.test_only:
        train = [
            "--lambda_identity", "0", "--preprocess", "crop", "--crop_size", "256",
            "--batch_size", "1", "--num_threads", str(args.num_threads), "--seed", "42",
            "--n_epochs", "40", "--n_epochs_decay", "40", "--lr", "0.0002",
            "--print_freq", "100", "--display_freq", "400", "--update_html_freq", "400",
            "--save_latest_freq", "5000", "--save_epoch_freq", "5",
        ]
        if args.stop_after_epoch:
            train += ["--stop_after_epoch", str(args.stop_after_epoch)]
        run([sys.executable, "-u", "train.py", *common, *train], train_root)
    test = ["--results_dir", str(results), "--epoch", str(epoch), "--num_test", "112",
            "--preprocess", "none", "--no_flip", "--eval"]
    run([sys.executable, "-u", "test.py", *common, *test], train_root)
    print(f"Training preview: {checkpoints / args.name / 'web/index.html'}", flush=True)
    print(f"Fixed 112 testB results: {results / args.name / f'test_{epoch}/index.html'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
