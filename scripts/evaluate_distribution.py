#!/usr/bin/env python3
"""Architecture-independent NTIR-to-RGB evaluation.

The FLIR test RGB and TIR domains are unpaired, so this deliberately does not
report paired-image measures such as PSNR, SSIM, or LPIPS.

It reports FID, KID, APCE-Py, and edge precision/recall/F1 in one command.
APCE-Py preserves PearlGAN's formula and threshold sweep but uses
scikit-image's Canny implementation. It is fair for comparisons within this
project, but is not PearlGAN's official MATLAB APCE.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    """Return a stable flat image list and fail early on invalid directories."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    return files


def generated_path(tir_path: Path, generated_dir: Path, suffix: str) -> Path:
    """Resolve a result file explicitly; never silently skip a missing image."""
    expected_name = tir_path.name if suffix == "" else f"{tir_path.stem}{suffix}"
    path = generated_dir / expected_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing generated image for {tir_path.name}. Expected: {path}\n"
            "Use --generated-suffix _fake_B.png for CycleGAN BtoA outputs, "
            "or --generated-suffix '' when results retain the source name."
        )
    return path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def tensor_batch(paths: list[Path], torch_module, device):
    arrays = [load_rgb(path) for path in paths]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(
            "All images in each evaluation set must have one common size. "
            f"Found shapes: {sorted(shapes)}"
        )
    # TorchMetrics with normalize=True expects RGB float tensors in [0, 1].
    data = np.stack(arrays).transpose(0, 3, 1, 2)
    return torch_module.from_numpy(data).float().div_(255.0).to(device)


def import_metrics():
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as error:
        raise SystemExit(
            "Missing metric dependencies. In the SAME Python environment used "
            "to run this script, execute:\n"
            "  python -m pip install -r scripts/requirements_metrics.txt\n"
            f"Original import error: {error}"
        ) from error
    return torch, FrechetInceptionDistance, KernelInceptionDistance


def import_canny():
    try:
        from skimage.feature import canny
    except ImportError as error:
        raise SystemExit(
            "Missing edge-metric dependency. In the SAME Python environment "
            "used to run this script, execute:\n"
            "  python -m pip install -r scripts/requirements_metrics.txt\n"
            f"Original import error: {error}"
        ) from error
    return canny


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def choose_device(torch_module, requested: str):
    if requested != "auto":
        return torch_module.device(requested)
    # Inception feature extraction is most reproducible on CUDA or CPU.
    # MPS is intentionally not selected automatically.
    return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")


def load_gray(path: Path, rgb_weights: bool) -> np.ndarray:
    """Load [0, 1] grayscale; generated RGB uses MATLAB rgb2gray weights."""
    with Image.open(path) as image:
        if rgb_weights:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def compute_edge_metrics(tir_files: list[Path], generated_files: list[Path], sigma: float):
    """Compute APCE-Py and P/R/F1 using PearlGAN's threshold/average protocol."""
    canny = import_canny()
    thresholds = np.arange(1, 100, dtype=np.float32) / 100.0
    image_pairs = []
    for tir_path, generated_path_ in zip(tir_files, generated_files):
        tir_gray = load_gray(tir_path, rgb_weights=False)
        generated_gray = load_gray(generated_path_, rgb_weights=True)
        if tir_gray.shape != generated_gray.shape:
            raise ValueError(
                f"Image size mismatch for {tir_path.name}: TIR={tir_gray.shape}, "
                f"generated={generated_gray.shape}"
            )
        image_pairs.append((tir_gray, generated_gray))

    rows = []
    for high_threshold in thresholds:
        low_threshold = float(high_threshold * 0.5)
        recall_sum = precision_sum = f1_sum = 0.0
        valid_images = 0
        for tir_gray, generated_gray in image_pairs:
            tir_edge = canny(tir_gray, sigma=sigma, low_threshold=low_threshold,
                             high_threshold=float(high_threshold))
            tir_count = int(tir_edge.sum())
            if tir_count == 0:
                # This is PearlGAN APCE's valid-image rule.
                continue
            generated_edge = canny(generated_gray, sigma=sigma, low_threshold=low_threshold,
                                   high_threshold=float(high_threshold))
            generated_count = int(generated_edge.sum())
            overlap = int(np.logical_and(tir_edge, generated_edge).sum())
            valid_images += 1
            recall_sum += overlap / tir_count
            if generated_count > 0:
                precision_sum += overlap / generated_count
            f1_sum += 2.0 * overlap / (tir_count + generated_count)

        if valid_images == 0:
            raise ValueError(f"No non-empty TIR edge maps at threshold {high_threshold:.2f}.")
        rows.append({
            "high_threshold": float(high_threshold),
            "low_threshold": low_threshold,
            "precision": precision_sum / valid_images,
            "recall_apce": recall_sum / valid_images,
            "f1": f1_sum / valid_images,
            "valid_images": valid_images,
        })

    return rows, {
        "apce_py": float(np.mean([row["recall_apce"] for row in rows])),
        "edge_precision_py": float(np.mean([row["precision"] for row in rows])),
        "edge_recall_py": float(np.mean([row["recall_apce"] for row in rows])),
        "edge_f1_py": float(np.mean([row["f1"] for row in rows])),
        "edge_canny_backend": "scikit-image.feature.canny",
        "edge_canny_sigma": sigma,
        "edge_high_thresholds": "0.01:0.01:0.99",
        "edge_low_threshold_rule": "low = 0.5 * high",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FID, KID, APCE-Py, and Edge-F1 for NTIR-to-RGB outputs."
    )
    parser.add_argument("--tir-dir", type=Path, required=True,
                        help="Fixed TIR test directory; defines exactly which outputs are scored.")
    parser.add_argument("--generated-dir", type=Path, required=True,
                        help="Flat directory containing generated RGB images.")
    parser.add_argument("--generated-suffix", default="_fake_B.png",
                        help="Suffix after each TIR stem; use '' for same-name outputs.")
    parser.add_argument("--reference-rgb-dir", type=Path, required=True,
                        help="Fixed daytime RGB reference domain, e.g. FLIR_datasets/testA.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kid-subset-size", type=int, default=50)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--edge-canny-sigma", type=float, default=1.0,
                        help="Fixed scikit-image Canny Gaussian sigma for every method.")
    parser.add_argument("--report-title", default=None,
                        help="Optional title for the automatically generated PNG/PDF report.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "cuda:0"),
                        help="Use CUDA when available; CPU otherwise. MPS is intentionally excluded.")
    args = parser.parse_args()

    if args.batch_size < 1 or args.kid_subset_size < 2 or args.kid_subsets < 1 or args.edge_canny_sigma <= 0:
        parser.error("batch >= 1, KID subset >= 2, KID subsets >= 1, and Canny sigma > 0 are required")

    tir_files = image_files(args.tir_dir)
    generated_files = [generated_path(path, args.generated_dir, args.generated_suffix)
                       for path in tir_files]
    reference_files = image_files(args.reference_rgb_dir)
    if min(len(generated_files), len(reference_files)) < args.kid_subset_size:
        parser.error(
            f"KID subset size ({args.kid_subset_size}) exceeds an image set: "
            f"generated={len(generated_files)}, reference={len(reference_files)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "evaluation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tir_filename", "generated_filename"))
        writer.writerows((tir.name, generated.name) for tir, generated in zip(tir_files, generated_files))

    torch, FrechetInceptionDistance, KernelInceptionDistance = import_metrics()
    device = choose_device(torch, args.device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(
        feature=2048,
        subset_size=args.kid_subset_size,
        subsets=args.kid_subsets,
        normalize=True,
    ).to(device)

    with torch.inference_mode():
        for paths in batches(reference_files, args.batch_size):
            values = tensor_batch(paths, torch, device)
            fid.update(values, real=True)
            kid.update(values, real=True)
        for paths in batches(generated_files, args.batch_size):
            values = tensor_batch(paths, torch, device)
            fid.update(values, real=False)
            kid.update(values, real=False)
        fid_value = float(fid.compute().detach().cpu())
        kid_mean, kid_std = kid.compute()
        kid_value = float(kid_mean.detach().cpu())
        kid_std_value = float(kid_std.detach().cpu())

    edge_rows, edge_summary = compute_edge_metrics(
        tir_files, generated_files, args.edge_canny_sigma
    )
    edge_output_path = args.output_dir / "edge_threshold_metrics.csv"
    with edge_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)

    result = {
        "metric_protocol": {
            "distribution": "TorchMetrics Inception-v3 features; normalize=True; unpaired RGB-domain distribution only",
            "structure": "PearlGAN APCE threshold/aggregation formula; scikit-image Canny backend, not official MATLAB",
        },
        "fid": fid_value,
        "kid_mean": kid_value,
        "kid_std": kid_std_value,
        "num_tir_inputs": len(tir_files),
        "num_generated": len(generated_files),
        "num_reference_rgb": len(reference_files),
        "generated_suffix": args.generated_suffix,
        "kid_subset_size": args.kid_subset_size,
        "kid_subsets": args.kid_subsets,
        "batch_size": args.batch_size,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torchmetrics": package_version("torchmetrics"),
        "torch-fidelity": package_version("torch-fidelity"),
    }
    result.update(edge_summary)
    result["scikit-image"] = package_version("scikit-image")
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep report generation in the same evaluation transaction: one command
    # always writes JSON, threshold CSV, PNG, and PDF from exactly the same data.
    from render_metrics_figure import render_metric_report
    report_title = args.report_title or f"{args.output_dir.name} evaluation"
    report_png, report_pdf = render_metric_report(
        result, edge_rows, args.output_dir / "metric_report", report_title
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_path}")
    print(f"Edges:    {edge_output_path}")
    print(f"Metrics:  {output_path}")
    print(f"Report:   {report_png}")
    print(f"Report:   {report_pdf}")


if __name__ == "__main__":
    main()
