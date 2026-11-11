"""Score generated RGB against explicitly paired, geometrically aligned RGB."""

import argparse
from collections import defaultdict
from importlib import metadata
from pathlib import Path

from filelock import FileLock
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity
import torch

from D.common import read_json, sha256, write_json
from .progress import ItemProgress


FORMAT = "D_aligned_rgb_eval_v1"
REGION_EROSION_PX = 2


def aligned_pairs(manifest_path, generated_dir):
    manifest = read_json(manifest_path)
    if manifest.get("format") != FORMAT or not manifest.get("dataset") or not manifest.get("pairs"):
        raise ValueError(f"Provide a named, explicit registered test-pair manifest of format {FORMAT}")
    names = {int(label): name for label, name in manifest.get("label_names", {}).items()}
    if len(set(names.values())) != len(names):
        raise ValueError("Region label names must be unique")
    rows = {}
    for item in manifest["pairs"]:
        tir, reference = (manifest_path.parent / item[key] for key in ("tir", "rgb"))
        labels = manifest_path.parent / item["labels"] if "labels" in item else None
        stem = tir.stem
        if (reference.stem != stem or stem in rows or reference.resolve() == tir.resolve()
                or (labels is not None and labels.stem != stem) or bool(names) != (labels is not None)):
            raise ValueError(f"Expected a unique shared pair stem and a consistent label policy: {tir}")
        prediction = generated_dir / f"{stem}_fake_B.png"
        if prediction.resolve() == reference.resolve():
            raise ValueError(f"Use an independent RGB reference for {stem}")
        rows[stem] = (tir, reference, prediction, labels)
    return manifest["dataset"], names, rows


def lpips_backbone():
    from torchvision.models import SqueezeNet1_1_Weights

    url = SqueezeNet1_1_Weights.IMAGENET1K_V1.url
    path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if not path.is_file() or not sha256(path).startswith("b8a52dc0"):
        raise FileNotFoundError(f"Cache the original SqueezeNet weights at {path}. Source: {url}")
    return path


def rgb_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@torch.inference_mode()
def evaluate_pairs(rows, names):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    torch.set_num_threads(4)
    backbone = lpips_backbone()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).eval()
    progress = ItemProgress(len(rows), "PAIRED")
    per_frame, region_means = {}, defaultdict(list)
    for index, (stem, (tir_path, reference_path, prediction_path, label_path)) in enumerate(rows.items(), 1):
        with Image.open(tir_path) as tir:
            expected_shape = (tir.height, tir.width, 3)
        reference, prediction = map(rgb_array, (reference_path, prediction_path))
        if reference.shape != expected_shape or prediction.shape != expected_shape:
            raise ValueError(f"Paired evaluation requires identical native geometry for {stem}")

        tensors = [torch.from_numpy(rgb).permute(2, 0, 1)[None] for rgb in (prediction, reference)]
        lpips = float(metric(*tensors))
        metric.reset()
        delta = deltaE_ciede2000(rgb2lab(prediction), rgb2lab(reference))
        ssim = structural_similarity(prediction, reference, channel_axis=-1, data_range=1.0,
                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        values = dict(lpips_squeeze=lpips, ssim=float(ssim), delta_e00=float(delta.mean()))
        inputs = dict(tir=tir_path, reference=reference_path, generated=prediction_path)
        if label_path is not None:
            inputs["labels"] = label_path
            with Image.open(label_path) as label_image:
                labels = np.array(label_image)
            if labels.shape != delta.shape or labels.dtype.kind not in "ui":
                raise ValueError(f"Expected native-size integer class IDs in {label_path}")
            regions = {}
            for label, name in names.items():
                interior = binary_erosion(labels == label, iterations=REGION_EROSION_PX)
                if interior.any():
                    regions[name] = float(delta[interior].mean())
                    region_means[name].append(regions[name])
            values["region_delta_e00"] = regions
        values["input_sha256"] = {key: sha256(path) for key, path in inputs.items()}
        per_frame[stem] = values
        progress.update(index)

    regions = {}
    for name in names.values():
        values = region_means[name]
        regions[name] = dict(images=len(values), delta_e00=float(np.mean(values)) if values else None)
    return dict(
        frames=len(per_frame),
        overall={key: float(np.mean([row[key] for row in per_frame.values()]))
                 for key in ("lpips_squeeze", "ssim", "delta_e00")},
        regions=regions,
        lpips_backbone_sha256=sha256(backbone),
        region_erosion_px=REGION_EROSION_PX,
        versions={key: metadata.version(key) for key in ("torch", "torchvision", "torchmetrics", "scikit-image")},
        per_frame=per_frame,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True, help="RGB outputs named <pair-stem>_fake_B.png")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, names, rows = aligned_pairs(args.pair_manifest, args.generated_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output.with_name(args.output.name + ".lock")), timeout=0):
        if args.output.exists():
            parser.error("Preserve existing paired metrics; choose a new --output")
        report = evaluate_pairs(rows, names)
        write_json(args.output, dict(dataset=dataset, manifest_sha256=sha256(args.pair_manifest), **report))
    print(f"Metrics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
