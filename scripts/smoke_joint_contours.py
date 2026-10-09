"""One unlabeled FLIR update per variant, using the real loader and model factory.

Copies one train RGB/TIR image into an isolated smoke dataset. test copies are
the SAME images, exclusively to test inference; never performance evidence.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from data import create_dataset
from data.image_folder import make_dataset
from models import create_model
from options.train_options import TrainOptions
from preview_joint_contours import parse_args as preview_args, run as preview


def run(args):
    if args.output_dir.exists():
        raise FileExistsError("Use a new smoke output directory")
    rgb_files = sorted(make_dataset(str(args.dataroot / "trainA"), 1))
    tir_files = sorted(make_dataset(str(args.dataroot / "trainB"), 1))
    if not rgb_files or not tir_files:
        raise ValueError("Source requires processed trainA/trainB")
    review = ROOT / "runs/edge_audit/important_contour_review_v1/review_manifest.json"
    if review.exists():
        filename = json.loads(review.read_text(encoding="utf-8"))["selection"]["samples"][0]["filename"]
        preferred = args.dataroot / "trainB" / filename
        if preferred.exists():
            tir_files = [str(preferred)]
    data_root = args.output_dir / "data"
    for split, source in (("trainA", rgb_files[0]), ("testA", rgb_files[0]),
                          ("trainB", tir_files[0]), ("testB", tir_files[0])):
        directory = data_root / split
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / Path(source).name)
    torch.set_num_threads(1)
    checkpoints, reports, initial_hashes = [], {}, []
    for variant in ("plain", "contrastive"):
        argv = ["smoke_joint", "--dataroot", str(data_root), "--name", variant,
                "--checkpoints_dir", str(args.output_dir / "checkpoints"),
                "--model", "joint_contour_cycle_gan", "--joint_variant", variant,
                "--direction", "BtoA", "--input_nc", "1", "--output_nc", "3",
                "--condition_mode", "soft_saliency", "--fusion_mode", "direct",
                "--lambda_identity", "0", "--n_epochs", "1", "--n_epochs_decay", "0",
                "--ngf", "8", "--ndf", "8", "--netG", "resnet_6blocks",
                "--contour_width", "8", "--contour_embedding_dim", "8",
                "--num_threads", "0", "--max_dataset_size", "1", "--preprocess", "none",
                "--no_flip", "--serial_batches", "--joint_smoke", "--seed", "2026", "--no_html"]
        original_argv = sys.argv
        try:
            sys.argv = argv
            opt = TrainOptions().parse()
        finally:
            sys.argv = original_argv
        opt.device = torch.device(args.device)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        dataset = create_dataset(opt)
        model = create_model(opt)
        model.setup(opt)
        initial_hashes.append(model.initial_parameters_sha256)
        for data in dataset:
            model.set_input(data)
            model.optimize_parameters()
        losses = model.get_current_losses()
        if not all(np.isfinite(value) for value in losses.values()):
            raise RuntimeError("Non-finite smoke losses")
        if variant == "contrastive" and losses["contour_active_anchors"] == 0:
            raise RuntimeError("This smoke image produced no NCE anchors; inspect heuristic coverage")
        model.update_learning_rate()
        model.save_networks("latest")
        model.save_networks(1)
        checkpoints.append(args.output_dir / "checkpoints" / variant / "latest_net_G_A.pth")
        reports[variant] = losses
        del model
    if len(set(initial_hashes)) != 1:
        raise RuntimeError("Initial parameters differed")
    preview(preview_args([
        "--plain-checkpoint", str(checkpoints[0]), "--contrastive-checkpoint", str(checkpoints[1]),
        "--image-dir", str(data_root / "testB"), "--output-dir", str(args.output_dir / "preview"),
        "--num-images", "1", "--device", args.device, "--probes"]))
    report = {"scope": "implementation_smoke", "passed": True, "initial_parameters_sha256": initial_hashes[0],
              "source_rgb": rgb_files[0], "source_tir": tir_files[0], "losses": reports,
              "limits": "One unlabeled update per variant. Same images reused for inference. No quality/generalization claims."}
    (args.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: unlabeled joint train/export smoke; {args.output_dir / 'smoke_report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=ROOT / "FLIR_datasets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    run(parser.parse_args())
