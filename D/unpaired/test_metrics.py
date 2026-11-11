"""Score completed FLIR D test results with an independent RGB detector."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from importlib import metadata
from io import StringIO
import math
from pathlib import Path

from filelock import FileLock
import numpy as np
from PIL import Image

from D.common import ROOT, read_json, sha256, write_json
from scripts.prepare_flir_v2 import PEARL_CROP_BOX, PEARL_RESIZE_SIZE
from .evaluate import checked_prediction, evaluate_predictions
from .progress import ItemProgress, event


DEFAULT_MANIFEST = ROOT / "runs/D/ubuntu_assets_20261030/cycle_manifest.json"
DEFAULT_COCO = ROOT / "FLIR_ADAS_v2/images_thermal_val/coco.json"
WEIGHTS_NAME = "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
WEIGHTS_URL = f"https://download.pytorch.org/models/{WEIGHTS_NAME}"
DEFAULT_WEIGHTS = ROOT / "runs/detectors/coco" / WEIGHTS_NAME
TARGETS = {1: "person", 3: "car"}
PROCESSED_SIZE = (PEARL_CROP_BOX[2] - PEARL_CROP_BOX[0], PEARL_CROP_BOX[3] - PEARL_CROP_BOX[1])


def processed_bbox(bbox, raw_size):
    """COCO xywh after the exact FLIR resize, center crop and clipping."""
    x, y, width, height = bbox
    if not all(math.isfinite(v) for v in bbox) or width <= 0 or height <= 0 or min(raw_size) <= 0:
        raise ValueError(f"Invalid raw COCO box/size: {bbox}, {raw_size}")
    left, top, right, bottom = PEARL_CROP_BOX
    sx, sy = (resized / raw for resized, raw in zip(PEARL_RESIZE_SIZE, raw_size))
    x1, y1 = max(x * sx, left), max(y * sy, top)
    x2, y2 = min((x + width) * sx, right), min((y + height) * sy, bottom)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - left, y1 - top, x2 - x1, y2 - y1]


def flir_ground_truth(raw_coco, sources):
    """Require exactly one raw image and preserve crowd flags and empty frames."""
    groups = {row["filename"]: row["group"] for row in sources}
    names = {name: category for category, name in TARGETS.items()}
    labels = {row["id"]: names[row["name"]] for row in raw_coco["categories"] if row["name"] in names}
    if sorted(labels.values()) != sorted(TARGETS):
        raise ValueError("FLIR annotation must define exactly one person and one car category")

    images, raw_sizes, selected_names = [], {}, set()
    for row in raw_coco["images"]:
        filename = Path(row["file_name"]).name
        if filename not in groups:
            continue
        if filename in selected_names or row["id"] in raw_sizes:
            raise ValueError(f"Ambiguous COCO image: {filename}")
        selected_names.add(filename)
        raw_sizes[row["id"]] = (row["width"], row["height"])
        images.append(dict(id=row["id"], file_name=filename, group=groups[filename],
                           width=PROCESSED_SIZE[0], height=PROCESSED_SIZE[1]))
    if missing := groups.keys() - selected_names:
        raise ValueError(f"Missing raw FLIR annotations for {len(missing)} frames, including {min(missing)}")

    annotations = []
    for row in raw_coco["annotations"]:
        image_id, raw_category = row["image_id"], row["category_id"]
        if image_id not in raw_sizes or raw_category not in labels:
            continue
        box = processed_bbox(row["bbox"], raw_sizes[image_id])
        if box is not None:
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id, category_id=labels[raw_category],
                                    bbox=box, area=box[2] * box[3], iscrowd=int(row["iscrowd"])))
    return dict(images=sorted(images, key=lambda row: row["file_name"]), annotations=annotations,
                categories=[dict(id=category, name=name) for category, name in TARGETS.items()])


def coco_scores(ground_truth, detections):
    """Use the official COCO bbox precision tensor for the full split and each video."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if detections:
            dt = gt.loadRes([row.copy() for row in detections])
        else:
            # COCO.loadRes indexes the first detection even when the list is empty.
            dt = COCO()
            dt.dataset = dict(ground_truth, annotations=[])
            dt.createIndex()

        def score(image_ids):
            scorer = COCOeval(gt, dt, "bbox")
            scorer.params.imgIds = sorted(image_ids)
            scorer.params.catIds = list(TARGETS)
            scorer.evaluate()
            scorer.accumulate()
            params = scorer.params
            precision = scorer.eval["precision"][:, :, :, params.areaRngLbl.index("all"), params.maxDets.index(100)]

            def average(values):
                values = values[values >= 0]
                return float(values.mean()) if values.size else None

            def ap(values):
                return dict(ap=average(values), ap50=average(values[np.isclose(params.iouThrs, .50)]),
                            ap75=average(values[np.isclose(params.iouThrs, .75)]))

            counts = Counter(row["category_id"] for row in ground_truth["annotations"]
                             if row["image_id"] in image_ids and not row["iscrowd"])
            return dict(frames=len(image_ids), **ap(precision), classes={
                TARGETS[category]: dict(boxes=counts[category], **ap(precision[:, :, index]))
                for index, category in enumerate(params.catIds)})

        images = ground_truth["images"]
        groups = sorted({row["group"] for row in images})
        return dict(overall=score({row["id"] for row in images}), by_video={
            group: score({row["id"] for row in images if row["group"] == group}) for group in groups})


def detect(prediction_dir, images, weights_path):
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    from scripts.train_flir_roi_detector import make_model

    torch.set_num_threads(4)
    model = make_model(weights_path).eval().requires_grad_(False)
    transform = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.transforms()
    progress = ItemProgress(len(images), "DETECT")
    detections = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            rows = images[start:start + 8]
            batch = []
            for row in rows:
                path = prediction_dir / "images" / f"{Path(row['file_name']).stem}_fake_B.png"
                with Image.open(path) as image:
                    if image.size != PROCESSED_SIZE:
                        raise ValueError(f"Generated image must use FLIR test coordinates {PROCESSED_SIZE}: {path}")
                    batch.append(transform(image.convert("RGB")))
            for row, result in zip(rows, model(batch)):
                for box, label, score in zip(*(result[key].tolist() for key in ("boxes", "labels", "scores"))):
                    if label in TARGETS:
                        x1, y1, x2, y2 = box
                        detections.append(dict(image_id=row["id"], category_id=label, score=score,
                                               bbox=[x1, y1, x2 - x1, y2 - y1]))
            progress.update(start + len(rows))
    return detections


def evaluate_test(test_dir, data_manifest, annotation_path, weights_path, output, boundary_dirs=None):
    prediction_dir, cache_dir, legacy_dir = (test_dir / name for name in ("adapted", "conditions", "evaluation"))
    prediction, cache, sources, _ = checked_prediction(prediction_dir, cache_dir, data_manifest, required_phase="test")
    if cache["identity"].get("dataset_id") != "flir_v2":
        raise ValueError("FLIR COCO boxes require the flir_v2 test geometry")
    for source in sources:
        with Image.open(source["path"]) as image:
            if image.size != PROCESSED_SIZE:
                raise ValueError(f"Input is outside the FLIR test geometry: {source['path']}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Place the official frozen detector at {weights_path}. Source: {WEIGHTS_URL}")
    weights_sha = sha256(weights_path)
    if not weights_sha.startswith("a79551df"):
        raise ValueError("Expected the original COCO RGB detector, with official SHA256 prefix a79551df")

    boundaries = None
    if boundary_dirs:
        from .boundary_metrics import boundary_files
        boundaries = boundary_files(sources, *boundary_dirs)

    legacy = evaluate_predictions(prediction_dir, cache_dir, data_manifest, legacy_dir, device="cpu", resume=True)
    identity = dict(
        prediction_sha256=sha256(prediction_dir / "prediction.json"),
        records_sha256=sha256(prediction_dir / "records.json"),
        data_manifest_sha256=sha256(data_manifest),
        original_metrics_sha256=sha256(legacy_dir / "metrics/metrics.json"),
        annotations_sha256=sha256(annotation_path),
        evaluator_sha256=sha256(Path(__file__)),
        detector=dict(name=WEIGHTS_NAME, sha256=weights_sha, device="cpu",
                      versions={key: metadata.version(key) for key in ("torch", "torchvision", "pycocotools")}),
        boundary_annotations={stem: [sha256(path) for path in paths] for stem, paths in boundaries.items()}
        if boundaries is not None else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(output.name + ".lock")), timeout=0):
        if output.exists():
            previous = read_json(output)
            if previous["provenance"] != identity:
                raise ValueError("Preserve existing test metrics; choose a new --output for changed inputs")
            return previous

        ground_truth = flir_ground_truth(read_json(annotation_path), sources)
        event("DETECT", f"{len(sources)} RGB predictions | frozen COCO SSDLite320 | person + car")
        scores = coco_scores(ground_truth, detect(prediction_dir, ground_truth["images"], weights_path))
        report = dict(
            provenance=identity,
            checkpoint_step=prediction["metadata"].get("step"),
            distribution={key: legacy[key] for key in ("fid", "kid_mean", "kid_std", "num_reference_rgb")},
            source_edges={key: legacy[key] for key in ("apce_py", "edge_precision_py", "edge_f1_py")},
            generated_rgb_detection=scores,
        )
        if boundaries is not None:
            from .boundary_metrics import evaluate_boundaries
            report["annotated_boundaries"] = evaluate_boundaries(prediction_dir, cache_dir, boundaries)
        write_json(output, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True, help="Local test containing adapted/, conditions/, evaluation/")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coco", type=Path, default=DEFAULT_COCO, help="Original FLIR thermal validation coco.json")
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--boundary-gt-dir", type=Path, help="Independent same-stem boundary PNGs at native test size")
    parser.add_argument("--boundary-roi-dir", type=Path, help="Same-stem PNGs marking exhaustively annotated regions")
    args = parser.parse_args()
    if (args.boundary_gt_dir is None) != (args.boundary_roi_dir is None):
        parser.error("Supply both boundary directories or omit both")
    boundary_dirs = (args.boundary_gt_dir, args.boundary_roi_dir) if args.boundary_gt_dir else None
    name = "extended_metrics_with_boundaries.json" if boundary_dirs else "extended_metrics.json"
    output = args.output or args.test_dir / name
    report = evaluate_test(args.test_dir, args.data_manifest, args.coco, args.detector_weights, output, boundary_dirs)
    overall = report["generated_rgb_detection"]["overall"]
    for name, row in [("person+car", overall), *overall["classes"].items()]:
        values = " | ".join(f"{key.upper()}={value:.4f}" if value is not None else f"{key.upper()}=NA"
                            for key in ("ap", "ap50", "ap75") for value in (row[key],))
        event("METRICS", f"{name} | {values}")
    print(f"Metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
