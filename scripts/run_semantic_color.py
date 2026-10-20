#!/usr/bin/env python3
"""Explicit check/train/test launcher for the independent semantic/color stage."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run_semantic_contour_ubuntu import preflight

ROOT = Path(__file__).resolve().parents[1]


def run(command, directory):
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--train", action="store_true")
    action.add_argument("--test-only", action="store_true")
    parser.add_argument("--thermal-root", type=Path, default=ROOT)
    parser.add_argument("--name", default="flir_v2_semcolor_manual_v1_seed42")
    parser.add_argument("--color-condition", choices=("none", "semantic", "semantic_color"), default="semantic_color")
    parser.add_argument("--color-weight", type=float, default=None)
    parser.add_argument("--color-width", type=int, default=16)
    parser.add_argument("--check-device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--init-semantic-dir", type=Path, default=None)
    parser.add_argument("--init-semantic-epoch", default="80")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.0002)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This launcher supports a single CUDA training process")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in (".", ".."):
        parser.error("Use a single safe experiment directory name")
    if not 0 <= args.stop_after_epoch <= 80 or args.epoch < 1 or args.num_threads < 0 or args.color_width < 1 or not 0 < args.lr < float("inf"):
        parser.error("Invalid epoch, worker, width or learning rate")
    weight = args.color_weight if args.color_weight is not None else (1. if args.color_condition == "semantic_color" else 0.)
    if not 0 <= weight < float("inf") or (args.color_condition != "semantic_color" and weight):
        parser.error("Only semantic_color supports a positive finite color weight")
    if args.check_device == "cpu" and not args.check_only:
        parser.error("CPU is for --check-only; full training/export requires CUDA")
    root = args.thermal_root.expanduser().resolve()
    train_root, checkpoints, results = root / "pytorch-CycleGAN-and-pix2pix", root / "runs/checkpoints", root / "runs/results"
    semantic = root / "runs/pretrained/segformer_fmb_b3/best_model.pth"
    if args.check_only:
        # Use the same already-installed local metadata fallback as the weight loader.
        from semantic_preview.checkpoint import metadata_safe_globals
        metadata_safe_globals()
        preflight(root, root / "FLIR_datasets", semantic, args.check_device)
        # --help exits the upstream base parser before model-specific options exist.
        code = (
            "import sys; from options.train_options import TrainOptions; "
            "from options.test_options import TestOptions; "
            "args=['--dataroot',sys.argv[1],'--model','semantic_color_cycle_gan',"
            "'--color_condition','semantic_color','--color_width','16','--lambda_color_prior','1']; "
            "sys.argv=['check',*args]; a=TrainOptions().gather_options(); "
            "sys.argv=['check',*args]; b=TestOptions().gather_options(); "
            "assert a.color_width==b.color_width==16 and a.color_condition==b.color_condition=='semantic_color'; "
            "print('Semantic/color train and test options parsed successfully; no experiment created.')"
        )
        run([sys.executable, "-c", code, str(root / "FLIR_datasets")], train_root)
        return
    epoch = args.epoch if args.test_only else (args.stop_after_epoch or 80)
    if args.train:
        if (checkpoints / args.name).exists() or (results / args.name).exists():
            parser.error("Choose a new experiment name; existing experiments are preserved")
        if args.init_semantic_dir:
            for network in ("G_A", "G_B"):
                if not (args.init_semantic_dir / f"{args.init_semantic_epoch}_net_{network}.pth").is_file():
                    parser.error("Missing original semantic checkpoint for initialization")
    else:
        if args.init_semantic_dir:
            parser.error("Do not reinitialize an existing trained checkpoint in --test-only")
        for network in ("G_A", "G_B"):
            if not (checkpoints / args.name / f"{epoch}_net_{network}.pth").is_file():
                parser.error("Missing semantic/color checkpoint")
    if (results / args.name / f"test_{epoch}").exists():
        parser.error("Test output already exists; preserve it or choose another name/epoch")
    run([sys.executable, str(Path(__file__).resolve()), "--thermal-root", str(root), "--check-only"], root)
    common = ["--dataroot", str(root / "FLIR_datasets"), "--name", args.name,
              "--checkpoints_dir", str(checkpoints), "--model", "semantic_color_cycle_gan",
              "--semantic_checkpoint", str(semantic), "--color_condition", args.color_condition,
              "--color_width", str(args.color_width), "--lambda_color_prior", str(weight),
              "--condition_mode", "soft_saliency", "--fusion_mode", "direct",
              "--direction", "BtoA", "--input_nc", "1", "--output_nc", "3", "--dataset_mode", "unaligned",
              "--netG", "resnet_9blocks", "--netD", "basic", "--ngf", "64", "--ndf", "64",
              "--norm", "instance", "--no_dropout"]
    if args.train:
        train = ["--lambda_identity", "0", "--preprocess", "crop", "--crop_size", "256", "--batch_size", "1",
                 "--num_threads", str(args.num_threads), "--seed", "42", "--n_epochs", "40", "--n_epochs_decay", "40",
                 "--lr", str(args.lr), "--save_epoch_freq", "5", "--save_latest_freq", "5000"]
        if args.stop_after_epoch:
            train += ["--stop_after_epoch", str(args.stop_after_epoch)]
        if args.init_semantic_dir:
            train += ["--init_semantic_dir", str(args.init_semantic_dir.resolve()),
                      "--init_semantic_epoch", args.init_semantic_epoch]
        run([sys.executable, "-u", "train.py", *common, *train], train_root)
    run([sys.executable, "-u", "test.py", *common, "--results_dir", str(results), "--epoch", str(epoch),
         "--num_test", "112", "--preprocess", "none", "--no_flip", "--eval"], train_root)


if __name__ == "__main__":
    main()
