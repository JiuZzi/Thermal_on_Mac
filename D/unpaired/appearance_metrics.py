"""Per-region Lab and color-transition diagnostics on verified held-out outputs."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from PIL import Image

from D.common import sha256, write_json
from .appearance import CLASSES, lab_cdf, region_samples
from .data import ConditionDataset, data_identity
from .evaluate import checked_prediction
from .quality import _lab, appearance_class_weights


MEASURES = ("lightness_mean", "chroma_mean", "lightness_below_50", "lightness_above_90", "interior_chroma_jump")


def interior_chroma_jump(rgb, sample, minimum):
    weights = appearance_class_weights(sample["p"], sample["q"], rgb.shape, minimum)
    ab = _lab((rgb + 1) / 2)[:, 1:]
    edge = sample["x"][:, 1:2]
    total = weights.new_zeros(weights.shape[:2])
    mass = torch.zeros_like(total)
    for axis in (-1, -2):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        support = (1 - torch.maximum(edge[left], edge[right])).square()
        pair_weights = torch.minimum(weights[left], weights[right]) * support
        jump = (ab[left] - ab[right]).norm(dim=1, keepdim=True)
        total += (pair_weights * jump).sum((-2, -1))
        mass += pair_weights.sum((-2, -1))
    return (total / mass.clamp_min(1e-8)).masked_fill(mass <= 0, float("nan"))


@torch.no_grad()
def summarize(dataset, minimum, *, prediction_dir=None):
    """Use the same reliable classes and give each source group equal weight."""
    summaries = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        frame = sample["frame"]
        rgb = sample["rgb"]
        if prediction_dir is not None:
            path = prediction_dir / "images" / f"{frame['stem']}_fake_B.png"
            with Image.open(path) as image:
                if image.size != tuple(frame["native_size"]):
                    raise ValueError(f"Prediction size differs from the cached source: {path}")
                height, width = rgb.shape[-2:]
                image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
                values = np.array(image)
            rgb = torch.from_numpy(values).permute(2, 0, 1)[None].float() / 127.5 - 1
        lab, active = region_samples(rgb, sample["p"], sample["q"], minimum)
        lightness = lab[:, 0]
        stats = torch.stack((lightness.mean(-1), lab[:, 1:].norm(dim=1).mean(-1),
                             (lightness < 50).float().mean(-1), (lightness > 90).float().mean(-1),
                             interior_chroma_jump(rgb, sample, minimum)), dim=-1)
        summaries[frame["group"]].append((active[0], stats[0], lab_cdf(lab)[0]))
    by_class, distributions = {}, {}
    for cls, name in enumerate(CLASSES):
        group_stats, group_cdfs, frame_count = [], [], 0
        for group in summaries.values():
            present = [(values[cls], cdf[cls]) for mask, values, cdf in group if mask[cls]]
            if present:
                values, cdfs = zip(*present)
                group_stats.append(torch.stack(values).nanmean(dim=0))
                group_cdfs.append(torch.stack(cdfs).mean(dim=0))
                frame_count += len(present)
        if group_stats:
            means = torch.stack(group_stats).nanmean(dim=0).tolist()
            means = [None if np.isnan(value) else value for value in means]
            by_class[name] = dict(zip(MEASURES, means), groups=len(group_stats), frames=frame_count)
            distributions[name] = torch.stack(group_cdfs).mean(dim=0)
    return by_class, distributions


def evaluate_appearance(prediction_dir, data_cache, data_manifest, minimum, *, rgb_cache=None):
    if not 0 < minimum <= 1:
        raise ValueError("Appearance metric reliability must be in (0,1]")
    prediction, cache, _, references = checked_prediction(prediction_dir, data_cache, data_manifest)
    phase = prediction["metadata"]["phase"]
    tir = ConditionDataset(data_cache, "TIR", phase)
    generated, cdfs = summarize(tir, minimum, prediction_dir=prediction_dir)
    result = dict(format="D_appearance_metrics_v1", phase=phase, frames=len(tir), min_quality=minimum,
                  prediction_sha256=sha256(prediction_dir / "prediction.json"),
                  condition_manifest_sha256=sha256(data_cache / "manifest.json"),
                  generated=generated,
                  note="Grouped semantic proxies. A smaller chroma jump alone can also reflect desaturation.")
    if rgb_cache is not None:
        identity = data_identity(rgb_cache)
        rgb = ConditionDataset(rgb_cache, "RGB", phase)
        recipe, expected = identity["source"]["recipe"], cache["identity"]["recipe"]
        if (identity["source"]["source_manifest_sha256"] != sha256(data_manifest) or
            any(recipe[key] != expected[key] for key in ("size", "rgb_teacher", "flip_reliability")) or
            {frame["stem"] for frame in rgb.frames} != {Path(row["image"]).stem for row in references}):
            raise ValueError("Use the complete held-out RGB split with matching teacher and geometry")
        result["rgb_reference"], rgb_cdfs = summarize(rgb, minimum)
        result["rgb_condition_manifest_sha256"] = identity["manifest_sha256"]
        result["cdf_distance_to_rgb"] = {}
        for name in sorted(cdfs.keys() & rgb_cdfs.keys()):
            distance = (cdfs[name] - rgb_cdfs[name]).abs().mean(dim=-1)
            result["cdf_distance_to_rgb"][name] = dict(lightness=float(distance[0]), chroma=float(distance[1:].mean()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, help="Optional same-phase held-out RGB condition cache")
    parser.add_argument("--min-quality", type=float, default=.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output) + ".lock", timeout=0):
        if args.output.exists():
            parser.error("Preserve existing appearance metrics; choose a new --output")
        result = evaluate_appearance(args.prediction_dir, args.data_cache, args.data_manifest,
                                     args.min_quality, rgb_cache=args.rgb_cache)
        write_json(args.output, result)
    print(f"[DONE] {args.output} | {result['phase']} | {result['frames']} images")


if __name__ == "__main__":
    main()
