#!/usr/bin/env python3
"""Real-image forward/backward smoke; never a color-quality experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models import create_model
from options.train_options import TrainOptions


def image_tensor(path, size, channels):
    with Image.open(path) as source:
        image = source.convert("RGB" if channels == 3 else "L")
        if min(image.size) < size:
            raise ValueError("Image is smaller than requested crop")
        left, top = (image.width - size) // 2, (image.height - size) // 2
        values = np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float32) / 127.5 - 1
    if channels == 1:
        values = values[None]
    else:
        values = values.transpose(2, 0, 1)
    return torch.from_numpy(values.copy())[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/smoke/semantic_color_v1_20261020")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--init-semantic-dir", type=Path, default=None)
    parser.add_argument("--init-semantic-epoch", default="80")
    args = parser.parse_args()
    if args.output_dir.exists() or args.crop_size < 32 or args.crop_size % 4 or not 1 <= args.steps <= 2:
        parser.error("Use a new output directory, crop >=32 divisible by 4, and one or two smoke steps")
    output = args.output_dir.resolve()
    command = ["train.py", "--model", "semantic_color_cycle_gan", "--dataroot", str(ROOT / "FLIR_datasets"),
               "--direction", "BtoA", "--input_nc", "1", "--output_nc", "3", "--netG", "resnet_9blocks",
               "--ngf", "64", "--netD", "pixel", "--ndf", "8", "--no_dropout", "--norm", "instance",
               "--name", output.name, "--checkpoints_dir", str(output.parent), "--num_threads", "0", "--seed", "42"]
    if args.init_semantic_dir:
        command += ["--init_semantic_dir", str(args.init_semantic_dir.resolve()),
                    "--init_semantic_epoch", args.init_semantic_epoch]
    saved = sys.argv
    try:
        sys.argv = command
        opt = TrainOptions().gather_options()
    finally:
        sys.argv = saved
    opt.isTrain, opt.device = True, torch.device(args.device)
    torch.set_num_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    output.mkdir(parents=True)
    review = json.loads((ROOT / "scripts/semantic_preview/fixed_review_seed42.json").read_text())
    tir_path = ROOT / "FLIR_datasets/trainB" / review["items"][0]["filename"]
    rgb_path = next(path for path in sorted((ROOT / "FLIR_datasets/trainA").iterdir())
                    if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    tir, rgb = image_tensor(tir_path, args.crop_size, 1), image_tensor(rgb_path, args.crop_size, 3)
    data = {"B": tir, "A": rgb, "B_paths": [str(tir_path)], "A_paths": [str(rgb_path)]}
    started = time.perf_counter()
    model = create_model(opt); model.setup(opt); model.set_input(data)
    model.eval(); model.test()
    generator = model.netG_A
    with torch.no_grad():
        legacy_input = torch.cat((model.real_A, model.color_aux["edge"], torch.zeros_like(model.real_A)), 1)
        legacy_rgb = generator.backbone(legacy_input)
        max_difference = float((legacy_rgb - model.fake_B).abs().max())
        if max_difference != 0:
            raise RuntimeError("Zero-initialized branch changed the old generator output")
    losses = []
    for name in model.model_names:
        getattr(model, "net" + name).train()
    for _ in range(args.steps):
        model.set_input(data); model.optimize_parameters()
        loss = {name: float(getattr(model, "loss_" + name).detach())
                if torch.is_tensor(getattr(model, "loss_" + name)) else float(getattr(model, "loss_" + name))
                for name in model.loss_names}
        if not all(np.isfinite(value) for value in loss.values()):
            raise RuntimeError("Non-finite real-image losses")
        losses.append(loss)
    projection_gradient = sum(float(parameter.grad.abs().sum()) for parameter in generator.condition_projections.parameters()
                              if parameter.grad is not None)
    if projection_gradient <= 0:
        raise RuntimeError("New conditioning branch received no real-image gradient")
    if generator.semantic_prior.model.training or any(parameter.requires_grad or parameter.grad is not None
                                                    for parameter in generator.semantic_prior.model.parameters()):
        raise RuntimeError("Frozen semantic model entered training")
    report = dict(evidence="real-image implementation smoke only; no quality or generalization conclusion",
                  device=str(opt.device), torch=torch.__version__, crop_size=args.crop_size, steps=args.steps,
                  generator="full resnet_9blocks ngf64", discriminator="small pixel ndf8 (smoke only)",
                  tir=str(tir_path), tir_sha256=hashlib.sha256(tir_path.read_bytes()).hexdigest(),
                  unpaired_rgb=str(rgb_path), warmstart_zero_branch_max_abs_difference=max_difference,
                  projection_gradient_l1=projection_gradient, semantic_weights_frozen=True, losses=losses,
                  semantic_checkpoint_sha256=generator.semantic_metadata["checkpoint_sha256"],
                  elapsed_seconds=time.perf_counter() - started)
    (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
