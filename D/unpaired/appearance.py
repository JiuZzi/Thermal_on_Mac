"""Group-balanced Lab distributions for unpaired D-v3 appearance training."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from filelock import FileLock

from D.common import ROOT, sha256, write_json
from D.prepare_rgb import NAMES
from .data import ConditionDataset, data_identity
from .protocol import config_from
from .quality import APPEARANCE_REGION_MIN_MASS, _lab, appearance_class_weights


FORMAT = "D_region_lab_reference_v1"
CLASSES = NAMES[1:]
SLICES = ("lightness", "a", "b", "a_plus_b", "a_minus_b")
BINS = 65
REGION_SAMPLES = 128
MIN_TRAIN_GROUPS = 2
HISTORY_RATE = 1 / 16


def region_samples(rgb, probabilities, quality, minimum, *, offset=.5):
    """Systematic samples of reliable interior pixels, with gradients to RGB only."""
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not 0 <= offset < 1:
        raise ValueError("Expected RGB [N,3,H,W] and a sampling offset in [0,1)")
    with torch.no_grad():
        weights = appearance_class_weights(probabilities, quality, rgb.shape, minimum).flatten(2)
        cumulative = weights.cumsum(-1)
        mass = cumulative[..., -1:]
        positions = (torch.arange(REGION_SAMPLES, device=rgb.device) + offset) / REGION_SAMPLES
        indices = torch.searchsorted(cumulative, mass * positions, right=True).clamp_max(weights.shape[-1] - 1)
        active = mass.squeeze(-1) >= APPEARANCE_REGION_MIN_MASS
    pixels = rgb.float().flatten(2).transpose(1, 2)
    sampled = pixels[torch.arange(rgb.shape[0], device=rgb.device)[:, None, None], indices]
    return _lab((sampled.permute(0, 3, 1, 2) + 1) / 2), active


def lab_cdf(samples):
    """Soft CDFs of L* and four projections of (a*, b*) on a common scale."""
    lightness, a, b = samples.unbind(1)
    a, b = a / 128, b / 128
    values = torch.stack((lightness / 50 - 1, a, b, (a + b) / 2, (a - b) / 2), dim=-1)
    knots = torch.linspace(-1, 1, BINS, dtype=values.dtype, device=values.device)
    return ((knots - values[..., None]) * ((BINS - 1) / 2) + .5).clamp(0, 1).mean(dim=2)


@torch.no_grad()
def fit_reference(cache, minimum, *, frames_per_group=64):
    """Average images within each training group, then give groups equal weight."""
    if type(frames_per_group) is not int or frames_per_group < 1 or not 0 < minimum <= 1:
        raise ValueError("Use positive frame limits and a reliability threshold in (0,1]")
    identity = data_identity(cache)
    dataset = ConditionDataset(cache, "RGB", "train")
    groups = defaultdict(list)
    for index, frame in enumerate(dataset.frames):
        groups[frame["group"]].append(index)
    sums = torch.zeros(len(CLASSES), len(SLICES), BINS, dtype=torch.float64)
    group_counts = torch.zeros(len(CLASSES), dtype=torch.int64)
    frame_counts = torch.zeros_like(group_counts)
    sampled_frames = 0
    for _, indices in sorted(groups.items()):
        indices.sort(key=lambda index: dataset.frames[index]["id"])
        count = min(len(indices), frames_per_group)
        group_sum = torch.zeros_like(sums)
        group_frames = torch.zeros_like(group_counts)
        for slot in range(count):
            # Evenly cover each group. The cache order has no effect.
            sample = dataset[indices[(2 * slot + 1) * len(indices) // (2 * count)]]
            lab, active = region_samples(sample["rgb"], sample["p"], sample["q"], minimum)
            active = active[0]
            group_sum += lab_cdf(lab)[0] * active[:, None, None]
            group_frames += active
        present = group_frames > 0
        sums += group_sum / group_frames.clamp_min(1)[:, None, None]
        group_counts += present
        frame_counts += group_frames
        sampled_frames += count
    supported = group_counts >= MIN_TRAIN_GROUPS
    if not supported.any():
        raise ValueError("No reliable semantic class appears in multiple RGB training groups")
    cdf = (sums / group_counts.clamp_min(1)[:, None, None]) * supported[:, None, None]
    if sha256(Path(cache) / "manifest.json") != identity["manifest_sha256"]:
        raise ValueError("Condition manifest changed during appearance fitting")
    return dict(format=FORMAT, domain="RGB", role="train",
                condition_manifest_sha256=identity["manifest_sha256"], min_quality=minimum,
                classes=list(CLASSES), slices=list(SLICES), frames_per_group=frames_per_group,
                sampled_frames=sampled_frames, groups=len(groups), group_counts=group_counts.tolist(),
                frame_counts=frame_counts.tolist(), cdf=cdf.tolist())


def _valid_cdf(cdf):
    return (isinstance(cdf, torch.Tensor) and cdf.is_floating_point() and
            cdf.shape == (len(CLASSES), len(SLICES), BINS) and
            bool(torch.isfinite(cdf).all() and (cdf >= 0).all() and (cdf <= 1).all()
                 and (cdf.diff(dim=-1) >= -1e-6).all()))


def load_reference(path, manifest_sha256, minimum):
    raw = Path(path).read_bytes()
    reference = json.loads(raw)
    if (reference.get("format") != FORMAT or reference.get("domain") != "RGB" or
        reference.get("role") != "train" or reference.get("condition_manifest_sha256") != manifest_sha256 or
        reference.get("min_quality") != minimum or reference.get("classes") != list(CLASSES) or
        reference.get("slices") != list(SLICES)):
        raise ValueError("Fit the appearance reference on the exact RGB training cache and reliability threshold")
    counts = reference.get("group_counts", [])
    groups = reference.get("groups")
    if (type(groups) is not int or groups < MIN_TRAIN_GROUPS or len(counts) != len(CLASSES) or
        any(type(count) is not int or not 0 <= count <= groups for count in counts)):
        raise ValueError("Invalid appearance reference group support")
    supported = torch.tensor(counts) >= MIN_TRAIN_GROUPS
    cdf = torch.tensor(reference["cdf"], dtype=torch.float32)
    if not _valid_cdf(cdf) or not supported.any() or cdf[~supported].count_nonzero():
        raise ValueError("Invalid appearance reference distributions")
    return cdf, supported, hashlib.sha256(raw).hexdigest()


class RegionAppearanceDistribution:
    def __init__(self, reference, minimum, device):
        cdf, supported, _ = reference
        self.target = cdf.to(device)
        self.supported = supported.to(device)
        self.minimum = minimum
        self.history = self.target.clone()

    def loss(self, generated, probabilities, quality, *, step):
        # Shift the systematic sample positions by a deterministic sequence.
        lab, eligible = region_samples(generated, probabilities, quality, self.minimum,
                                       offset=(step * .61803398875 + .5) % 1)
        cdf = lab_cdf(lab)
        counts = eligible.sum(dim=0)
        current = (cdf * eligible[..., None, None]).sum(dim=0) / counts.clamp_min(1)[:, None, None]
        active = (counts > 0) & self.supported
        next_history = torch.where(active[:, None, None],
                                   torch.lerp(self.history, current.detach(), HISTORY_RATE), self.history)
        # Evaluate the population discrepancy on the running estimate, with
        # the current batch providing the gradient of that estimate. This lets
        # different images supply different color modes, including at batch 1.
        residual = next_history - self.target + (current - current.detach())
        error = residual.square().mean(dim=-1)
        by_slice = (error * active[:, None]).sum(dim=0) / active.sum().clamp_min(1)
        lightness, chroma = by_slice[0], by_slice[1:].mean()
        return (lightness + chroma) / 2, next_history, {
            "appearance_dist_lightness": lightness.detach(), "appearance_dist_chroma": chroma.detach(),
            "appearance_dist_classes": active.sum(),
        }

    def validate_history(self, history):
        if not _valid_cdf(history) or history[~self.supported.to(history.device)].count_nonzero():
            raise ValueError("Invalid checkpoint appearance distribution history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "D/unpaired/config_quality_v3_appearance.json")
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-group", type=int, default=64)
    args = parser.parse_args()
    config = config_from(args.config)
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output) + ".lock", timeout=0):
        if args.output.exists():
            parser.error("Preserve existing appearance reference; choose a new --output")
        print("Fitting group-balanced distributions from cached RGB training frames...", flush=True)
        result = fit_reference(args.data_cache, config["color_min_quality"], frames_per_group=args.frames_per_group)
        write_json(args.output, result)
    active = [name for name, count in zip(CLASSES, result["group_counts"]) if count >= MIN_TRAIN_GROUPS]
    print(f"[READY] {args.output} | {result['sampled_frames']} train RGB frames | classes={', '.join(active)}")


if __name__ == "__main__":
    main()
