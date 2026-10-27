"""Full-split FID/KID/APCE/edge evaluation using the unchanged project evaluator."""

import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image

from D.common import ROOT, read_json, sha256, select_sources
from .protocol import DEFAULT_MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True, help="The adapted/ directory from formal infer")
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    prediction = read_json(args.prediction_dir / "prediction.json")
    cache = read_json(args.data_cache / "manifest.json")
    phase = prediction["metadata"].get("phase")
    if phase not in ("val", "test") or "generator_sha256" not in prediction["metadata"]:
        parser.error("Evaluate an adapted validation/test inference output, not initial or short training preview")
    if (read_json(args.prediction_dir / "status.json").get("phase") != "completed" or
        prediction["condition_manifest_sha256"] != sha256(args.data_cache / "manifest.json") or
        cache["identity"]["source_manifest_sha256"] != sha256(args.data_manifest)):
        parser.error("Incomplete predictions or data manifest mismatch")
    sources, _ = select_sources(args.data_manifest, phase=phase, limit=0)
    frames = [r for r in cache["frames"] if r["domain"] == "TIR" and r["role"] == phase]
    if {r["filename"] for r in frames} != {r["filename"] for r in sources} or set(prediction["frames"]) != {r["id"] for r in frames}:
        parser.error("Use the complete split; fixed 12-image previews are not distribution evidence")
    records = read_json(args.prediction_dir / "records.json")
    for source in sources:
        record = records[source["stem"]]
        if (sha256(source["path"]) != record["source_sha256"] or
            sha256(args.prediction_dir / "images" / f"{source['stem']}_fake_B.png") != record["sha256"]):
            parser.error("Source or generated image changed")
    if args.output_dir.exists():
        parser.error("Preserve existing metrics; choose a new output-dir")
    args.output_dir.mkdir(parents=True)
    tir_dir, rgb_dir = args.output_dir / "inputs_TIR", args.output_dir / "reference_RGB"
    tir_dir.mkdir(); rgb_dir.mkdir()
    for row in sources:
        with Image.open(row["path"]) as image:
            image.convert("L").save(tir_dir / f"{row['stem']}.png")
    for row in read_json(args.data_manifest)["domains"]["A"]:
        if row["split"] == phase:
            with Image.open(args.data_manifest.parent / row["image"]) as image:
                image.convert("RGB").save(rgb_dir / f"{Path(row['image']).stem}.png")
    code = "import runpy,sys,torch; script=sys.argv.pop(1); torch.manual_seed(42); runpy.run_path(script,run_name='__main__')"
    subprocess.run([sys.executable,"-u","-c",code,str(ROOT / "scripts/evaluate_distribution.py"),
        "--tir-dir",str(tir_dir),"--generated-dir",str(args.prediction_dir / "images"),"--generated-suffix","_fake_B.png",
        "--reference-rgb-dir",str(rgb_dir),"--output-dir",str(args.output_dir / "metrics"),
        "--device",args.device,"--batch-size","16","--kid-subset-size","50","--kid-subsets","100",
        "--edge-canny-sigma","1.0"],check=True)


if __name__ == "__main__":
    main()
