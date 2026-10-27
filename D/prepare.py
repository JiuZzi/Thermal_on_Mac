"""Cache unchanged clarity_v1 E before loading any diffusion model."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from .common import (ROOT, EDGE_SOURCES, DEFAULT_CONDITIONS, GEOMETRY,
                     select_sources, sha256, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "runs/learned_color/data_seed42_v1/cycle_manifest.json")
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("val", "test", "train"), default="val")
    parser.add_argument("--limit", type=int, default=12, help="0 exports the complete selected split")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--semantic-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    rows, counts = select_sources(args.data_manifest, args.phase, args.limit, seed=42)
    if args.output_dir.exists():
        parser.error("Preserve existing condition cache; choose a new output-dir")
    if not args.semantic_checkpoint.is_file():
        parser.error("Missing complete FMB best_model.pth")
    import torch
    torch.set_num_threads(args.cpu_threads)
    sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
    from models.semantic_contour import FrozenSemanticPrior
    from models.clarity_semantic_edges import clarity_components, CLARITY_SETTINGS
    prior = FrozenSemanticPrior(args.semantic_checkpoint)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "maps").mkdir()
    (output / "images").mkdir()
    identity = dict(format="thermal_D_conditions_v1", phase=args.phase, seed=42,
                    selection="all" if not args.limit else "seed42 balanced video groups",
                    geometry=GEOMETRY, contour_variant="clarity_v1", settings=CLARITY_SETTINGS,
                    semantic_sha256=prior.info["sha256"], semantic_short_side=512, semantic_flip_tta=False,
                    manifest_sha256=sha256(args.data_manifest), counts=counts,
                    code_sha256={name: sha256(ROOT / name) for name in EDGE_SOURCES}, frames=[])
    write_json(output / "status.json", dict(phase="preparing", completed=0, total=len(rows)))
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        with Image.open(row["path"]) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8).copy()
        # Match the existing [-1,1] TIR -> [0,1] condition path, including float rounding.
        tir = (gray.astype(np.float32) / 255 - .5) / .5
        image = np.clip((tir + 1) / 2, 0, 1)
        probabilities = prior.predict(image, args.semantic_device)
        maps, diagnostics = clarity_components(image, probabilities, include_probes=False)
        relative = f"maps/{row['stem']}.npz"
        np.savez_compressed(output / relative, TIR_uint8=gray, E=maps["clarity_v1"],
                            joint_importance=maps["joint_importance"], importance_gain=maps["importance_gain"])
        Image.fromarray(gray).save(output / "images" / f"{row['stem']}_TIR.png")
        Image.fromarray(np.rint(maps["clarity_v1"] * 255).astype(np.uint8)).save(
            output / "images" / f"{row['stem']}_E.png")
        identity["frames"].append(dict(filename=row["filename"], stem=row["stem"], group=row["group"],
             source=row["source"], source_sha256=sha256(row["path"]), native_size=[gray.shape[1], gray.shape[0]],
             condition=relative, condition_sha256=sha256(output / relative), **diagnostics))
        write_json(output / "manifest.json", identity)
        write_json(output / "status.json", dict(phase="preparing", completed=index, total=len(rows)))
        print(f"E [{index}/{len(rows)}] {row['filename']}", flush=True)
    write_json(output / "status.json", dict(phase="completed", completed=len(rows), total=len(rows),
                                           seconds=time.perf_counter() - started))
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
