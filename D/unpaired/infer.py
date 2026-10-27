"""Load a formal-D checkpoint and generate validation/test RGB at source coordinates."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path

from safetensors.torch import load_file

from D.common import ROOT, DEFAULT_CACHE, read_json, sha256
from D.models import load_pipeline
from .protocol import DEFAULT_DATA, DEFAULT_PRIOR, model_config, seed_runtime
from .data import ConditionDataset
from .model import FormalGenerator
from .checkpoint import verify
from .report import preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--phase", choices=("val", "test"), default="val")
    parser.add_argument("--step", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--compare-initial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_runtime()
    run = read_json(args.run_dir / "run.json")
    if run.get("format") != "D_unpaired_run_v1" or sha256(args.prior) != run["prior_sha256"]:
        parser.error("Use a formal D run and its original color-prior JSON")
    step = args.step if args.step is not None else read_json(args.run_dir / "latest.json")["step"]
    checkpoint = args.run_dir / "checkpoints" / f"step_{step:06d}"
    header = verify(args.run_dir, checkpoint)
    dataset = ConditionDataset(args.data_cache, "TIR", args.phase)
    if dataset.manifest["identity"] != run["data"]["source"]:
        parser.error("Test/validation conditions must use the same source manifest, semantic weights and geometry recipe")
    output = args.output_dir or args.run_dir / "predictions" / f"{args.phase}_step_{step:06d}"
    initialization = run["initialization"]
    recorded_checkpoint = initialization.get("checkpoint", "").replace("\\", "/")
    rgb_init = ROOT / recorded_checkpoint if args.compare_initial and initialization["kind"] == "RGB E-only conversion" else None
    if rgb_init is not None and sha256(rgb_init / "adapter.safetensors") != initialization["sha256"]:
        parser.error("The recorded RGB initialization changed")
    pipeline = load_pipeline(model_config(run["config"]), args.cache_dir)
    model = FormalGenerator(pipeline, run["config"], args.prior, run["data"]["source"]["recipe"], rgb_init=rgb_init).to("cuda:0")
    del pipeline
    baseline = output / "initial" if args.compare_initial else None
    if args.compare_initial:
        if initialization["kind"].startswith("formal D warm start"):
            original = ROOT / recorded_checkpoint / "generator.safetensors"
            if sha256(original) != initialization["files"]["generator.safetensors"]:
                parser.error("The recorded formal initialization changed")
            model.import_state(load_file(str(original)))
        preview(model, dataset, baseline, resume=args.resume,
                metadata=dict(run_sha256=sha256(args.run_dir / "run.json"), phase=args.phase, initialization=initialization))
    model.import_state(load_file(str(checkpoint / "generator.safetensors")))
    print(preview(model, dataset, output / "adapted", baseline=baseline, resume=args.resume,
                  metadata=dict(run_sha256=sha256(args.run_dir / "run.json"), phase=args.phase, step=step,
                                generator_sha256=header["files"]["generator.safetensors"])), flush=True)


if __name__ == "__main__":
    main()
