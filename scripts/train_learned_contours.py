"""Two controlled contour learners: supervised vs supervised + local contrast.

Requires real boundary labels for real experiments. --synthetic-smoke only
checks implementation, and marks every checkpoint accordingly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from contour_learning.data import ContourDataset, read_annotations
from contour_learning.model import SoftContourNet, transform_view
from contour_learning.objectives import boundary_loss, decode_labels, local_contrastive_loss

ROOT = Path(__file__).resolve().parents[1]


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums, counts = [0.0, 0.0], [0, 0]
    predicted, truth, precision_matches, recall_matches = 0, 0, 0, 0
    for image, labels in loader:
        image, labels = image.to(device), labels.to(device)
        logits, _ = model(image)
        target, valid = decode_labels(labels)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        for value in (0, 1):
            mask = valid & (target == value)
            sums[value] += losses[mask].sum().item()
            counts[value] += mask.sum().item()
        pred = (logits.sigmoid() >= 0.5) & valid
        gt = (target == 1) & valid
        near_gt = F.max_pool2d(gt.float(), 3, 1, 1) > 0
        near_pred = F.max_pool2d(pred.float(), 3, 1, 1) > 0
        predicted += pred.sum().item()
        truth += gt.sum().item()
        precision_matches += (pred & near_gt).sum().item()
        recall_matches += (gt & near_pred).sum().item()
    precision = precision_matches / max(1, predicted)
    recall = recall_matches / max(1, truth)
    return {"balanced_bce": sum(s / n for s, n in zip(sums, counts)) / 2,
            "boundary_precision_tol1": precision, "boundary_recall_tol1": recall,
            "boundary_f1_tol1": 2 * precision * recall / max(1e-12, precision + recall),
            "edge_fraction_at_0.5": predicted / sum(counts),
            "annotated_pixels": sum(counts)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--variant", choices=("supervised", "contrastive"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--n-epochs-decay", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--max-negatives", type=int, default=256)
    parser.add_argument("--exclusion-radius", type=int, default=2)
    parser.add_argument("--edge-crop-probability", type=float, default=0.5)
    parser.add_argument("--intensity-jitter", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true",
                        help="Bypass FLIR membership only for synthetic implementation checks")
    return parser.parse_args(argv)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(args):
    if args.n_epochs < 1 or args.n_epochs_decay < 0 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("Invalid training budget/batch/save interval")
    if args.seed < 0 or args.lr <= 0 or args.contrastive_weight <= 0 or args.temperature <= 0:
        raise ValueError("Seed >= 0; lr, contrastive weight and temperature must be positive")
    if min(args.max_anchors, args.max_negatives) < 1 or args.exclusion_radius < 0 or args.num_workers < 0:
        raise ValueError("Invalid sampling/worker settings")
    if not 0 <= args.intensity_jitter <= 0.2:
        raise ValueError("intensity_jitter must be in [0, 0.2]")
    if args.synthetic_smoke and args.n_epochs + args.n_epochs_decay > 2:
        raise ValueError("Synthetic smoke runs are limited to two epochs; not a quality experiment")
    records, data_metadata = read_annotations(
        args.annotations, None if args.synthetic_smoke else args.protocol, args.processed_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.resolve().parent != args.output_dir.resolve():
        raise ValueError("Resume in the original output directory so best.pt/history remain available")
    if not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is nonempty; use a new experiment name or --resume")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = SoftContourNet(args.width, args.embedding_dim).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {key: getattr(args, key) for key in (
        "variant", "n_epochs", "n_epochs_decay", "batch_size", "crop_size", "width",
        "embedding_dim", "seed", "lr", "contrastive_weight", "temperature", "max_anchors",
        "max_negatives", "exclusion_radius", "edge_crop_probability", "intensity_jitter", "synthetic_smoke")}
    metadata = {**data_metadata, "initial_model_sha256": initial_hash,
                "purpose": "synthetic_implementation_check" if args.synthetic_smoke else "boundary_learning",
                "training_config": config, "torch_version": str(torch.__version__)}
    start, best, history = 1, float("inf"), []
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != "learned_contours_v1" or state["metadata"]["training_config"] != config:
            raise ValueError("Resume training configuration differs")
        if state["metadata"]["dataset_sha256"] != metadata["dataset_sha256"] or state["metadata"]["protocol_sha256"] != metadata["protocol_sha256"]:
            raise ValueError("Resume dataset/protocol differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["epoch"] + 1, state["best_val_bce"], state["history"]
        metadata = state["metadata"]
    train_set = ContourDataset(records, "train", args.crop_size, args.seed, args.edge_crop_probability)
    val_set = ContourDataset(records, "val", args.crop_size, args.seed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Purpose: {metadata['purpose']}; train={len(train_set)}, val={len(val_set)}, device={device}", flush=True)
    print(f"Initial weights SHA256: {initial_hash}", flush=True)
    for epoch in range(start, args.n_epochs + args.n_epochs_decay + 1):
        # Epoch-local RNGs permit exact epoch-boundary restarts without worker/RNG snapshots.
        order_rng = torch.Generator().manual_seed(args.seed + epoch * 3)
        view_rng = torch.Generator().manual_seed(args.seed + epoch * 3 + 1)
        contrast_rng = torch.Generator().manual_seed(args.seed + epoch * 3 + 2)
        train_set.epoch = epoch
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            generator=order_rng, num_workers=args.num_workers)
        # 40 constant + 40 positive decreasing learning rates; zero after completion.
        lr_factor = 1 if epoch <= args.n_epochs else (args.n_epochs + args.n_epochs_decay - epoch + 1) / (args.n_epochs_decay + 1)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_factor
        model.train()
        totals, active_anchors, active_batches = [0.0, 0.0, 0.0], 0, 0
        for image, labels in loader:
            image, labels = image.to(device), labels.to(device)
            views = []
            for _ in range(2):
                code = int(torch.randint(8, (), generator=view_rng))
                gain = 1 + (float(torch.rand((), generator=view_rng)) * 2 - 1) * args.intensity_jitter
                offset = (float(torch.rand((), generator=view_rng)) * 2 - 1) * args.intensity_jitter
                view = transform_view((image * gain + offset).clamp(-1, 1), code)
                logits, features = model(view)
                views.append((boundary_loss(logits, transform_view(labels, code)),
                              transform_view(features, code, inverse=True)))
            task = (views[0][0] + views[1][0]) / 2
            contrast, active = task * 0, 0
            if args.variant == "contrastive":
                contrast, active = local_contrastive_loss(
                    model.projection(views[0][1]), model.projection(views[1][1]), labels,
                    contrast_rng, args.max_anchors, args.max_negatives,
                    args.temperature, args.exclusion_radius)
            loss = task + args.contrastive_weight * contrast
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for i, value in enumerate((loss, task, contrast)):
                totals[i] += value.item()
            active_anchors += active
            active_batches += int(active > 0)
        if args.variant == "contrastive" and active_batches == 0:
            raise RuntimeError("No usable contrastive anchors this epoch; check labels/crops/negative radius")
        metrics = validate(model, val_loader, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "loss": totals[0] / len(loader), "boundary_loss": totals[1] / len(loader),
               "contrastive_loss": totals[2] / len(loader), "active_anchors": active_anchors,
               "contrastive_active_batch_fraction": active_batches / len(loader), "val": metrics}
        history.append(row)
        improved = metrics["balanced_bce"] < best
        best = min(best, metrics["balanced_bce"])
        state = {"format": "learned_contours_v1", "model_config": model.config,
                 "metadata": metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val_bce": best, "history": history}
        save_checkpoint(args.output_dir / "latest.pt", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", state)
        if epoch % args.save_every == 0:
            save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", state)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    return metadata


if __name__ == "__main__":
    run(parse_args())
