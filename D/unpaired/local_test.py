"""Run a portable best-generator export on the local CPU with the original test split."""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
from safetensors.torch import load_file

from D.common import ROOT, DEFAULT_CACHE, read_json, sha256, write_json
from D.models import load_pipeline
from .data import ConditionDataset
from .evaluate import evaluate_predictions
from .model import FormalGenerator
from .protocol import DEFAULT_CONFIG, model_config, seed_runtime
from .report import preview
from .validation import POLICIES, winners


def read_bundle(folder, data_manifest):
    run_path = folder / "run.json"
    run = read_json(run_path)
    best = read_json(folder / "best.json")
    header = read_json(folder / "checkpoint.json")
    run_hash = sha256(run_path)
    if (run.get("format") != "D_unpaired_run_v1" or
        best.get("format") != "D_best_selection_v1" or best.get("policy") not in POLICIES or
        best["best"] != winners(best["records"], best["policy"])):
        raise ValueError("Expected an original formal D run and its validation best selection")
    record = next(item for item in best["records"] if item["id"] == best["best"]["primary"])
    generator_hash = header["files"]["generator.safetensors"]
    if (record["phase"] != "val" or record["run_sha256"] != run_hash or
        header.get("format") != "D_unpaired_checkpoint_v1" or header["run_sha256"] != run_hash or
        header["step"] != record["step"] or generator_hash != record["generator_sha256"]):
        raise ValueError("The exported generator does not belong to the primary validation best")
    source = run["data"]["source"]
    for path, digest in (
        (folder / "generator.safetensors", generator_hash),
        (folder / "best_model.pth", source["recipe"]["tir_semantic_sha256"]),
        (folder / "rgb_prior_seed42_loss_v1.json", run["prior_sha256"]),
        (data_manifest, source["source_manifest_sha256"]),
    ):
        if sha256(path) != digest:
            raise ValueError(f"File differs from the original run: {path}")
    return run, record


def check_local_conditions(original, local):
    """Allow CPU recomputation and library versions, retaining recipe and asset identity."""
    def comparable(identity):
        recipe = dict(identity["recipe"])
        recipe.pop("semantic_backend", None)
        if "condition_pipeline" in recipe:
            pipeline = dict(recipe["condition_pipeline"])
            pipeline.pop("versions", None)
            recipe["condition_pipeline"] = pipeline
        return dict(identity, recipe=recipe)

    if local["recipe"]["semantic_backend"] != "cpu" or comparable(original) != comparable(local):
        raise ValueError("Local test conditions must preserve the split, teachers, geometry and condition code")


def prepare_conditions(args, run, output):
    config = run["config"]
    if "condition_pipeline" not in run["data"]["source"]["recipe"]:
        # Some V3 runs reuse caches made under the original identity policy.
        config = {**read_json(DEFAULT_CONFIG),
                  **{key: config[key] for key in ("width", "height", "flip_reliability")}}
    config_file = output / "condition_config.json"
    if config_file.exists():
        if read_json(config_file) != config:
            raise ValueError("This local output directory uses another condition configuration")
    else:
        write_json(config_file, config)
    cache = output / "conditions"
    command = [
        sys.executable, "-u", "-m", "D.unpaired.prepare",
        "--config", str(config_file), "--data-manifest", str(args.data_manifest),
        "--semantic-checkpoint", str(args.bundle_dir / "best_model.pth"),
        "--roles", "test", "--domains", "TIR", "--test-limit", "0",
        "--device", "cpu", "--output-dir", str(cache),
    ]
    if cache.exists():
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)
    dataset = ConditionDataset(cache, "TIR", "test")
    check_local_conditions(run["data"]["source"], dataset.manifest["identity"])
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path,
                        default=ROOT / "runs/D/ubuntu_assets_20261030/cycle_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.bundle_dir = args.bundle_dir.resolve()
    args.data_manifest = args.data_manifest.resolve()
    output = args.output_dir.resolve()
    run, record = read_bundle(args.bundle_dir, args.data_manifest)
    config = run["config"]
    print(f"BEST primary: update {record['step']} | CPU float32", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(output.name + ".lock")), timeout=0):
        if output.exists() and not args.resume:
            parser.error("Use --resume for the existing local test or choose a new output-dir")
        output.mkdir(exist_ok=True)
        dataset = prepare_conditions(args, run, output)
        seed_runtime()
        pipeline = load_pipeline(model_config(config), DEFAULT_CACHE)
        prior = args.bundle_dir / "rgb_prior_seed42_loss_v1.json"
        generator = FormalGenerator(pipeline, config, prior, run["data"]["source"]["recipe"])
        del pipeline
        generator.float()
        generator.import_state(load_file(str(args.bundle_dir / "generator.safetensors"), device="cpu"))
        prediction_dir = output / "adapted"
        report = preview(generator, dataset, prediction_dir, resume=args.resume, metadata=dict(
            run_sha256=record["run_sha256"], phase="test", step=record["step"],
            generator_sha256=record["generator_sha256"], inference_device="cpu", inference_dtype="float32",
            inference_versions={key: metadata.version(key) for key in ("torch", "diffusers")},
        ))
        del generator
        print(f"Images: {report}", flush=True)
        evaluation = output / "evaluation"
        values = evaluate_predictions(prediction_dir, dataset.folder, args.data_manifest,
                                      evaluation, device="cpu", resume=args.resume)
        print(f"FID={values['fid']:.4f} KID={values['kid_mean']:.6f} "
              f"Edge-F1={values['edge_f1_py']:.6f} APCE={values['apce_py']:.6f}", flush=True)
        print(f"Metrics: {evaluation / 'metrics/metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
