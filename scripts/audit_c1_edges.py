#!/usr/bin/env python3
"""Audit C1 Canny edges before training; optional boundary-mask evaluation.

The default center crop previews a fixed 256x256 view of the processed TIR
images. C1 training uses random crops of the same images; Canny runs after the
loader's crop/flip in the generator. Use --geometry full to inspect test-time
360x288 inputs without using test images to choose thresholds.

With --gt-dir, provide a same-stem PNG containing manually traced, TIR-visible
boundaries. An optional --roi-dir PNG marks fully annotated evaluation regions
(white = evaluate, black = ignore). Without ROI masks, the entire image is
assumed to be exhaustively annotated. COCO detection boxes are not edge GT.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label
from skimage.feature import canny


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c1_trainB")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--low", type=float, default=0.08)
    parser.add_argument("--high", type=float, default=0.16)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="First N sorted files, for a quick pipeline check.")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Optional same-stem binary boundary PNGs.")
    parser.add_argument("--roi-dir", type=Path, default=None, help="Optional same-stem 255-valid, 0-ignore ROI PNGs.")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Matching distance in processed-image pixels.")
    args = parser.parse_args()
    if args.sigma <= 0 or not (0 <= args.low < args.high <= 1):
        parser.error("Require sigma > 0 and 0 <= low < high <= 1")
    if args.sample_count < 1 or (args.limit is not None and args.limit < 1):
        parser.error("sample-count and limit must be positive")
    if args.roi_dir is not None and args.gt_dir is None:
        parser.error("--roi-dir requires --gt-dir")
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    return args


def load_tir(path: Path, geometry: str) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("L")
        if geometry == "center_crop_256":
            if min(image.size) < 256:
                raise ValueError(f"Expected at least 256x256: {path}: {image.size}")
            left = (image.width - 256) // 2
            top = (image.height - 256) // 2
            image = image.crop((left, top, left + 256, top + 256))
        return image.copy()


def load_mask(path: Path, geometry: str, expected_size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = image.convert("L")
        if geometry == "center_crop_256" and image.size != expected_size:
            if min(image.size) < 256:
                raise ValueError(f"Mask too small: {path}: {image.size}")
            left = (image.width - 256) // 2
            top = (image.height - 256) // 2
            image = image.crop((left, top, left + 256, top + 256))
        if image.size != expected_size:
            raise ValueError(f"Mask size mismatch: {path}: {image.size}, expected {expected_size}")
        return np.asarray(image, dtype=np.uint8) > 127


def diagnostic_metrics(edge: np.ndarray, loose: np.ndarray, strict: np.ndarray) -> dict:
    labels, count = label(edge, structure=np.ones((3, 3), dtype=np.uint8))
    lengths = np.bincount(labels.ravel())[1:]
    edge_count = int(edge.sum())
    union = int(np.logical_or(loose, strict).sum())
    return {
        "edge_density": float(edge.mean()),
        "edge_pixels": edge_count,
        "component_count": int(count),
        "short_fragment_pixel_fraction": float(lengths[lengths < 8].sum() / edge_count) if edge_count else None,
        "threshold_jaccard_075x_125x": float(np.logical_and(loose, strict).sum() / union) if union else None,
    }


def boundary_metrics(edge: np.ndarray, gt: np.ndarray, roi: np.ndarray, tolerance: float) -> dict:
    predicted = edge & roi
    reference = gt & roi
    pred_count = int(predicted.sum())
    gt_count = int(reference.sum())
    if gt_count == 0:
        return {"gt_edge_pixels": 0, "pred_edge_pixels_in_roi": pred_count,
                "boundary_precision": None, "boundary_recall": None, "boundary_f1": None}
    near_gt = distance_transform_edt(~reference) <= tolerance
    near_pred = distance_transform_edt(~predicted) <= tolerance
    precision = float((predicted & near_gt).sum() / pred_count) if pred_count else 0.0
    recall = float((reference & near_pred).sum() / gt_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gt_edge_pixels": gt_count, "pred_edge_pixels_in_roi": pred_count,
            "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def overlay(image: Image.Image, edge: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    rgb[edge] = (255, 35, 35)
    return Image.fromarray(rgb)


def error_overlay(image: Image.Image, edge: np.ndarray, gt: np.ndarray,
                  roi: np.ndarray, tolerance: float) -> Image.Image:
    """Red = unsupported prediction; blue = missed annotation, within the ROI."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    predicted = edge & roi
    reference = gt & roi
    false_positive = predicted & (distance_transform_edt(~reference) > tolerance)
    false_negative = reference & (distance_transform_edt(~predicted) > tolerance)
    rgb[false_positive] = (255, 30, 30)
    rgb[false_negative] = (30, 90, 255)
    return Image.fromarray(rgb)


def select_samples(rows: list[dict], n: int) -> list[tuple[str, dict]]:
    ordered = sorted(rows, key=lambda row: row["edge_density"])
    groups = np.array_split(np.arange(len(ordered)), 3)
    labels = ("low_density", "middle_density", "high_density")
    selected = []
    counts = [n // 3 + (i < n % 3) for i in range(3)]
    for name, group, count in zip(labels, groups, counts):
        for index in np.linspace(0, len(group) - 1, min(count, len(group)), dtype=int):
            selected.append((name, ordered[int(group[index])]))
    return selected


def save_contact_sheets(selected: list[tuple[str, dict]], args: argparse.Namespace) -> None:
    for group in ("low_density", "middle_density", "high_density"):
        group_rows = [row for name, row in selected if name == group]
        if not group_rows:
            continue
        examples = []
        for row in group_rows:
            image = load_tir(args.input_dir / row["filename"], args.geometry)
            gray = np.asarray(image, dtype=np.float32) / 255.0
            edge = canny(gray, sigma=args.sigma, low_threshold=args.low, high_threshold=args.high)
            edge_image = Image.fromarray((edge * 255).astype(np.uint8)).convert("RGB")
            examples.append((row, image.convert("RGB"), edge_image, overlay(image, edge)))
            stem = Path(row["filename"]).stem
            sample_dir = args.output_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image.save(sample_dir / f"{stem}_tir.png")
            edge_image.save(sample_dir / f"{stem}_edge.png")
            examples[-1][3].save(sample_dir / f"{stem}_overlay.png")
            if args.gt_dir is not None and (args.gt_dir / f"{stem}.png").exists():
                gt = load_mask(args.gt_dir / f"{stem}.png", args.geometry, image.size)
                roi = (load_mask(args.roi_dir / f"{stem}.png", args.geometry, image.size)
                       if args.roi_dir is not None else np.ones_like(gt, dtype=bool))
                error_overlay(image, edge, gt, roi, args.tolerance).save(sample_dir / f"{stem}_errors.png")
        width, height = examples[0][1].size
        strip_height = height + 38
        sheet = Image.new("RGB", (width * 3, strip_height * len(examples)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (row, tir, edge_image, composite) in enumerate(examples):
            top = index * strip_height
            for column, panel in enumerate((tir, edge_image, composite)):
                sheet.paste(panel, (column * width, top))
            draw.text((3, top + height + 2), "TIR", fill="black")
            draw.text((width + 3, top + height + 2), f"Canny: {row['edge_density']:.1%} pixels", fill="black")
            draw.text((2 * width + 3, top + height + 2), "Red = detected edge", fill="black")
        sheet.save(args.output_dir / f"{group}.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No images in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gt_masks_found = 0
    for index, path in enumerate(files, start=1):
        image = load_tir(path, args.geometry)
        gray = np.asarray(image, dtype=np.float32) / 255.0
        edge = canny(gray, sigma=args.sigma, low_threshold=args.low, high_threshold=args.high)
        loose = canny(gray, sigma=args.sigma, low_threshold=args.high * 0.375,
                      high_threshold=args.high * 0.75)
        strict = canny(gray, sigma=args.sigma, low_threshold=args.high * 0.625,
                       high_threshold=min(args.high * 1.25, 1.0))
        row = {"filename": path.name, **diagnostic_metrics(edge, loose, strict)}
        if args.gt_dir is not None:
            gt_path = args.gt_dir / f"{path.stem}.png"
            if gt_path.exists():
                gt_masks_found += 1
                gt = load_mask(gt_path, args.geometry, image.size)
                roi = (load_mask(args.roi_dir / f"{path.stem}.png", args.geometry, image.size)
                       if args.roi_dir is not None else np.ones_like(gt, dtype=bool))
                row.update(boundary_metrics(edge, gt, roi, args.tolerance))
        rows.append(row)
        if index % 500 == 0:
            print(f"Audited {index}/{len(files)} images", flush=True)
    if args.gt_dir is not None and gt_masks_found == 0:
        raise ValueError(f"No same-stem PNG boundary masks found in {args.gt_dir}")
    write_csv(args.output_dir / "per_image.csv", rows)
    selected = select_samples(rows, args.sample_count)
    write_csv(args.output_dir / "selected_samples.csv", [{"group": name, **row} for name, row in selected])
    write_csv(
        args.output_dir / "human_review_template.csv",
        [{"group": name, "filename": row["filename"], "focus_object": "",
          "missing_edges_0_1_2": "", "irrelevant_edges_0_1_2": "", "notes": ""}
         for name, row in selected],
    )
    save_contact_sheets(selected, args)
    annotated = [row for row in rows if row.get("boundary_f1") is not None]
    def distribution(key: str) -> dict:
        values = np.array([row[key] for row in rows if row[key] is not None], dtype=float)
        return {"median": float(np.median(values)), "p10": float(np.quantile(values, 0.1)),
                "p90": float(np.quantile(values, 0.9))} if len(values) else {}
    summary = {
        "input_dir": str(args.input_dir.resolve()), "images": len(rows), "geometry": args.geometry,
        "canny": {"sigma": args.sigma, "low": args.low, "high": args.high},
        "edge_density": distribution("edge_density"),
        "short_fragment_pixel_fraction": distribution("short_fragment_pixel_fraction"),
        "threshold_jaccard_075x_125x": distribution("threshold_jaccard_075x_125x"),
        "empty_edge_maps": sum(row["edge_pixels"] == 0 for row in rows),
        "gt_masks_found": gt_masks_found,
        "annotated_images_scored": len(annotated),
        "boundary_metrics_mean": (
            {key: float(np.mean([row[key] for row in annotated]))
             for key in ("boundary_precision", "boundary_recall", "boundary_f1")}
            if annotated else None
        ),
        "interpretation": "Unlabeled diagnostics are not precision, recall, or hallucination rates. "
                          "Boundary scores require exhaustive visible-edge labels within each ROI.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote audit to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
