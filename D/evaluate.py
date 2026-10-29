"""Reuse the existing FID/KID/APCE/edge evaluator on a complete split only."""

import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image

from .common import ROOT, read_json, select_sources, sha256
from .metric_inputs import rgb_reference_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "runs/learned_color/data_seed42_v1/cycle_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run = read_json(args.preview_dir / "run.json")
    status = read_json(args.preview_dir / "status.json")
    source, phase = run["source"], run["source"]["phase"]
    if phase not in ("val", "test") or status.get("phase") != "completed":
        parser.error("Evaluate only a completed validation/test preview")
    if sha256(args.data_manifest) != source["manifest_sha256"]:
        parser.error("Data manifest differs from condition preparation")
    rows, _ = select_sources(args.data_manifest, phase=phase, limit=0)
    if {r["filename"] for r in rows} != {r["filename"] for r in source["frames"]}:
        parser.error("Distribution evaluation requires the entire split; a 12-image pilot is for visual review only")
    saved_sources = {r["filename"]: r["source_sha256"] for r in source["frames"]}
    if any(sha256(row["path"]) != saved_sources[row["filename"]] for row in rows):
        parser.error("A source TIR image changed after condition preparation")
    if args.output_dir.exists():
        parser.error("Choose a new output-dir to preserve previous metrics")
    records = read_json(args.preview_dir / "records.json")
    for frame in source["frames"]:
        for variant in run["variants"]:
            record = records[f"{variant}/{frame['stem']}"]
            if record["safety_flagged"] is not False:
                parser.error("Inspect missing/flagged safety-checker results before distribution evaluation")
            generated = args.preview_dir / variant / "images" / f"{frame['stem']}_fake_B.png"
            if sha256(generated) != record["sha256"]:
                parser.error("A generated image changed after inference")
    # Materialize this exact split using copies. No move/delete/link to the dataset.
    source_data = read_json(args.data_manifest)
    references = rgb_reference_rows(source_data, phase)
    args.output_dir.mkdir(parents=True)
    tir_dir, rgb_dir = args.output_dir / "inputs_TIR", args.output_dir / "reference_RGB"
    tir_dir.mkdir(); rgb_dir.mkdir()
    for row in rows:
        with Image.open(row["path"]) as im:
            im.convert("L").save(tir_dir / f"{row['stem']}.png")
    for row in references:
        with Image.open(args.data_manifest.parent / row["image"]) as im:
            im.convert("RGB").save(rgb_dir / f"{Path(row['image']).stem}.png")
    code = "import runpy,sys,torch; script=sys.argv.pop(1); torch.manual_seed(42); runpy.run_path(script,run_name='__main__')"
    for variant in run["variants"]:
        command = [sys.executable, "-u", "-c", code, str(ROOT / "scripts/evaluate_distribution.py"),
                   "--tir-dir", str(tir_dir), "--generated-dir", str(args.preview_dir / variant / "images"),
                   "--generated-suffix", "_fake_B.png", "--reference-rgb-dir", str(rgb_dir),
                   "--output-dir", str(args.output_dir / variant), "--device", args.device,
                   "--batch-size", "16", "--kid-subset-size", "50", "--kid-subsets", "100", "--edge-canny-sigma", "1.0"]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
