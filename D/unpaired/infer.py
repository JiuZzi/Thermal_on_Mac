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
from .checkpoint import verify, locate
from .report import preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--phase", choices=("val", "test"), default="val")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--step", type=int)
    choice.add_argument("--best", action="store_true", help="Use best.json from --run-dir")
    choice.add_argument("--best-file", type=Path, help="Use a best.json selection, including imported legacy results")
    parser.add_argument("--best-kind", choices=("primary", "structure", "fid", "kid"), default="primary")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--compare-initial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Verify checkpoint and condition identities without loading a GPU model")
    args = parser.parse_args()
    seed_runtime()
    best_checkpoint = None
    if args.best or args.best_file is not None:
        from .validation import selected_record
        if args.best and args.run_dir is None:
            parser.error("--best requires --run-dir; alternatively supply --best-file")
        best_file = args.best_file or args.run_dir / "best.json"
        selected_run, best_checkpoint, record = selected_record(best_file, args.best_kind)
        if args.run_dir is not None and args.run_dir.resolve() != selected_run:
            parser.error("The chosen best belongs to another run; use its run-dir or omit run-dir with best-file")
        args.run_dir = selected_run
        args.step = record["step"]
        print(f"BEST {args.best_kind}: update {args.step} | {best_checkpoint}", flush=True)
    elif args.run_dir is None or args.best_kind != "primary":
        parser.error("Supply --run-dir, or --best-file to select an automatically ranked checkpoint")
    run = read_json(args.run_dir / "run.json")
    if run.get("format") != "D_unpaired_run_v1" or sha256(args.prior) != run["prior_sha256"]:
        parser.error("Use a formal D run and its original color-prior JSON")
    checkpoint, step = locate(args.run_dir, args.step)
    if best_checkpoint is not None and checkpoint.resolve() != best_checkpoint.resolve():
        parser.error("Best checkpoint location changed")
    header = verify(args.run_dir, checkpoint)
    dataset = ConditionDataset(args.data_cache, "TIR", args.phase)
    if dataset.manifest["identity"] != run["data"]["source"]:
        parser.error("Test/validation conditions must use the same source manifest, semantic weights and geometry recipe")
    if args.check_only:
        print(dict(run_dir=str(args.run_dir), checkpoint=str(checkpoint), step=step,
                   phase=args.phase, images=len(dataset), seed=42), flush=True)
        return
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
