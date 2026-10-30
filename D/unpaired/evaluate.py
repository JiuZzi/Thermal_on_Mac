"""Full-split FID/KID/APCE/edge evaluation using the unchanged project evaluator."""

import argparse
from importlib import metadata
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image

from D.common import ROOT, read_json, sha256, select_sources
from D.metric_inputs import rgb_reference_rows
from .protocol import DEFAULT_MANIFEST

REPORT_FILES = ("metrics.json", "evaluation_manifest.csv", "edge_threshold_metrics.csv", "metric_report.png", "metric_report.pdf")


def checked_prediction(prediction_dir, data_cache, data_manifest, *, required_phase=None):
    """Verify the full split and image hashes before scoring or registering best."""
    prediction_dir, data_cache, data_manifest = map(Path, (prediction_dir, data_cache, data_manifest))
    prediction = read_json(prediction_dir / "prediction.json")
    cache = read_json(data_cache / "manifest.json")
    phase = prediction["metadata"].get("phase")
    if (prediction.get("format") != "D_unpaired_prediction_v1" or phase not in ("val", "test") or
        "generator_sha256" not in prediction["metadata"] or
        (required_phase is not None and phase != required_phase)):
        raise ValueError("Use adapted full-split predictions; best selection requires validation, never test")
    status = read_json(prediction_dir / "status.json")
    cache_status = read_json(data_cache / "status.json")
    if (status.get("phase") != "completed" or cache_status.get("phase") != "completed" or
        prediction["condition_manifest_sha256"] != sha256(data_cache / "manifest.json") or
        cache["identity"]["source_manifest_sha256"] != sha256(data_manifest)):
        raise ValueError("Incomplete predictions/conditions or data manifest mismatch")
    if any(cache_status.get(key, len(cache["frames"])) != len(cache["frames"]) for key in ("completed", "total")):
        raise ValueError("Condition status/frame count mismatch")
    sources, _ = select_sources(data_manifest, phase=phase, limit=0)
    frames = [r for r in cache["frames"] if r["domain"] == "TIR" and r["role"] == phase]
    if (len(frames) != len(sources) or len({r["stem"] for r in frames}) != len(frames) or
        len(prediction["frames"]) != len(frames) or status.get("count") != len(frames) or
        {r["filename"] for r in frames} != {r["filename"] for r in sources} or
        set(prediction["frames"]) != {r["id"] for r in frames}):
        raise ValueError("Use the complete split; fixed 12-image previews cannot select best")
    records = read_json(prediction_dir / "records.json")
    if set(records) != {r["stem"] for r in sources}:
        raise ValueError("Prediction records differ from the complete split")
    for source in sources:
        record = records[source["stem"]]
        if (record.get("raw_model_output") is not True or
            sha256(source["path"]) != record["source_sha256"] or
            sha256(prediction_dir / "images" / f"{source['stem']}_fake_B.png") != record["sha256"]):
            raise ValueError("Source or raw generated image changed")
    references = rgb_reference_rows(read_json(data_manifest), phase)
    return prediction, cache, sources, references


def checked_metrics(values, count, references):
    for key in ("fid", "kid_mean", "kid_std", "apce_py", "edge_precision_py", "edge_recall_py", "edge_f1_py"):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite validation metric: {key}")
        if key != "kid_mean" and value < 0:
            raise ValueError(f"Invalid negative validation metric: {key}")
        if key in ("apce_py", "edge_precision_py", "edge_recall_py", "edge_f1_py") and value > 1:
            raise ValueError(f"Invalid edge validation metric: {key}")
    expected = dict(num_tir_inputs=count, num_generated=count, num_reference_rgb=references,
                    generated_suffix="_fake_B.png", kid_subset_size=50, kid_subsets=100,
                    edge_canny_sigma=1.0)
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("Metric counts/protocol differ from the complete validation/test split")
    return values


def metric_environment(device):
    return dict(seed=42, device=device, batch_size=16, kid_subset_size=50, kid_subsets=100,
                edge_canny_sigma=1.0, versions={key:metadata.version(key) for key in
                ("torch", "torchvision", "torchmetrics", "torch-fidelity", "numpy", "scikit-image", "Pillow")})


def evaluate_predictions(prediction_dir, data_cache, data_manifest, output_dir, *, device="cuda", resume=False):
    """Run the original evaluator; commit provenance only after every report succeeds."""
    from filelock import FileLock
    from D.common import write_json
    prediction_dir, data_cache, data_manifest, output_dir = map(Path,
        (prediction_dir, data_cache, data_manifest, output_dir))
    prediction, cache, sources, references = checked_prediction(prediction_dir, data_cache, data_manifest)
    identity = dict(format="D_unpaired_evaluation_v1", phase=prediction["metadata"]["phase"],
                    prediction_sha256=sha256(prediction_dir / "prediction.json"),
                    records_sha256=sha256(prediction_dir / "records.json"),
                    condition_manifest_sha256=sha256(data_cache / "manifest.json"),
                    data_manifest_sha256=sha256(data_manifest), environment=metric_environment(device),
                    evaluator_sha256=sha256(ROOT / "scripts/evaluate_distribution.py"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir.with_name(output_dir.name + ".lock")), timeout=0):
        if output_dir.exists():
            if not resume or not (output_dir / "evaluation.json").is_file() or read_json(output_dir / "evaluation.json") != identity:
                raise ValueError("Preserve existing metrics; use matching --resume or a new output-dir")
            status_path = output_dir / "evaluation_status.json"
            if status_path.is_file() and read_json(status_path).get("phase") == "completed":
                saved_files = read_json(status_path)["files"]
                if set(saved_files) != set(REPORT_FILES):
                    raise ValueError("Incomplete evaluation report manifest")
                for name, digest in saved_files.items():
                    if sha256(output_dir / "metrics" / name) != digest:
                        raise ValueError("Completed evaluation report changed")
                return checked_metrics(read_json(output_dir / "metrics/metrics.json"), len(sources), len(references))
        else:
            output_dir.mkdir()
            write_json(output_dir / "evaluation.json", identity)
        write_json(output_dir / "evaluation_status.json", dict(phase="evaluating"))
        try:
            tir_dir, rgb_dir = output_dir / "inputs_TIR", output_dir / "reference_RGB"
            tir_dir.mkdir(exist_ok=True); rgb_dir.mkdir(exist_ok=True)
            for row in sources:
                with Image.open(row["path"]) as image:
                    image.convert("L").save(tir_dir / f"{row['stem']}.png")
            for row in references:
                with Image.open(data_manifest.parent / row["image"]) as image:
                    image.convert("RGB").save(rgb_dir / f"{Path(row['image']).stem}.png")
            code = ("import runpy,sys,torch; script=sys.argv.pop(1); torch.manual_seed(42); "
                    "torch.set_num_threads(4); runpy.run_path(script,run_name='__main__')")
            subprocess.run([sys.executable,"-u","-c",code,str(ROOT / "scripts/evaluate_distribution.py"),
                "--tir-dir",str(tir_dir),"--generated-dir",str(prediction_dir / "images"),"--generated-suffix","_fake_B.png",
                "--reference-rgb-dir",str(rgb_dir),"--output-dir",str(output_dir / "metrics"),
                "--device",device,"--batch-size","16","--kid-subset-size","50","--kid-subsets","100",
                "--edge-canny-sigma","1.0"],check=True)
            result = checked_metrics(read_json(output_dir / "metrics/metrics.json"), len(sources), len(references))
            files = {name:sha256(output_dir / "metrics" / name) for name in REPORT_FILES}
            write_json(output_dir / "evaluation_status.json", dict(phase="completed", files=files))
            return result
        except BaseException as error:
            try:
                write_json(output_dir / "evaluation_status.json", dict(phase="interrupted" if isinstance(error, KeyboardInterrupt)
                           else "failed", error=f"{type(error).__name__}: {error}"))
            except OSError:
                pass
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True, help="The adapted/ directory from formal infer")
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume only an evaluation with identical provenance")
    args = parser.parse_args()
    evaluate_predictions(args.prediction_dir, args.data_cache, args.data_manifest,
                         args.output_dir, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
