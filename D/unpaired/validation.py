"""Full-validation checkpoint selection; no optimizer, model or LR changes."""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from D.common import ROOT, read_json, write_json, sha256, select_sources
from D.metric_inputs import rgb_reference_rows
from . import checkpoint as checkpoints
from .data import ConditionDataset
from .evaluate import REPORT_FILES, checked_prediction, checked_metrics, evaluate_predictions, metric_environment


POLICIES = ("structure", "fid", "kid")
POLICY_DESCRIPTION = {
    "structure": "highest Edge-F1; ties: lowest FID, lowest KID, earliest step",
    "fid": "lowest FID; ties: highest Edge-F1, lowest KID, earliest step",
    "kid": "lowest KID; ties: lowest FID, highest Edge-F1, earliest step",
}
METRIC_KEYS = ("fid", "kid_mean", "kid_std", "apce_py", "edge_precision_py", "edge_recall_py", "edge_f1_py")


def stored_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def restore_path(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def full_validation(data_cache, data_manifest, *, source_identity=None):
    data_cache, data_manifest = Path(data_cache), Path(data_manifest)
    dataset = ConditionDataset(data_cache, "TIR", "val")
    sources, _ = select_sources(data_manifest, phase="val", limit=0)
    if (len(dataset) != len(sources) or
        {r["filename"] for r in dataset.frames} != {r["filename"] for r in sources}):
        raise ValueError("Best selection requires the complete validation split, not the 12-image preview")
    identity = dataset.manifest["identity"]
    if identity["source_manifest_sha256"] != sha256(data_manifest):
        raise ValueError("Validation source manifest changed")
    if source_identity is not None and identity != source_identity:
        raise ValueError("Validation conditions must use the training recipe, geometry and teachers")
    references = rgb_reference_rows(read_json(data_manifest), "val")
    return dataset, sources, references


def rank(record, policy):
    m = record["metrics"]
    tail = (record["step"], record["run_sha256"])
    if policy == "structure":
        return (-m["edge_f1_py"], m["fid"], m["kid_mean"], *tail)
    if policy == "fid":
        return (m["fid"], -m["edge_f1_py"], m["kid_mean"], *tail)
    if policy == "kid":
        return (m["kid_mean"], m["fid"], -m["edge_f1_py"], *tail)
    raise ValueError(f"Unknown best policy: {policy}")


def winners(records, policy):
    if not records:
        raise ValueError("Cannot select best without validation results")
    selected = {kind:min(records, key=lambda r:rank(r, kind))["id"] for kind in POLICIES}
    return dict(primary=selected[policy], **selected)


def _check_exported_inputs(evaluation_dir, sources, references, data_manifest):
    """Legacy metric files lacked a receipt: check their actual exported inputs."""
    for name, rows, mode in (("inputs_TIR", sources, "L"), ("reference_RGB", references, "RGB")):
        directory = evaluation_dir / name
        expected = {f"{r['stem']}.png" if name == "inputs_TIR" else f"{Path(r['image']).stem}.png" for r in rows}
        if {p.name for p in directory.iterdir() if p.is_file()} != expected:
            raise ValueError("Evaluation exports do not contain the complete validation split")
        for row in rows:
            source = row["path"] if name == "inputs_TIR" else data_manifest.parent / row["image"]
            stem = row["stem"] if name == "inputs_TIR" else Path(row["image"]).stem
            with Image.open(source) as a, Image.open(directory / f"{stem}.png") as b:
                if not np.array_equal(np.asarray(a.convert(mode)), np.asarray(b.convert(mode))):
                    raise ValueError("Exported validation metric input changed")
    with (evaluation_dir / "metrics/evaluation_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(f"{r['stem']}.png", f"{r['stem']}_fake_B.png") for r in sources}
    if len(rows) != len(expected) or {(r["tir_filename"], r["generated_filename"]) for r in rows} != expected:
        raise ValueError("Metric evaluation manifest differs from validation predictions")


def candidate_record(run_dir, step, prediction_dir, metrics_file, data_cache, data_manifest, *, import_existing=False):
    """Bind a score to the original run, immutable checkpoint and full val images."""
    run_dir, prediction_dir, metrics_file, data_cache, data_manifest = map(Path,
        (run_dir, prediction_dir, metrics_file, data_cache, data_manifest))
    if metrics_file.name != "metrics.json" or metrics_file.parent.name != "metrics":
        raise ValueError("Use the original evaluation/metrics/metrics.json")
    run = read_json(run_dir / "run.json")
    if run.get("format") != "D_unpaired_run_v1":
        raise ValueError("Best selection needs a formal D run")
    full_validation(data_cache, data_manifest, source_identity=run["data"]["source"])
    checkpoint, _ = checkpoints.locate(run_dir, step)
    # Only archival snapshots survive recovery pruning. Never point best at a rolling save.
    if checkpoint.resolve().parent != (run_dir / "checkpoints").resolve():
        raise ValueError("Best must reference an immutable archival checkpoint")
    header = checkpoints.verify(run_dir, checkpoint)
    prediction, cache, sources, references = checked_prediction(prediction_dir, data_cache, data_manifest, required_phase="val")
    metadata = prediction["metadata"]
    if (metadata.get("step") != step or metadata.get("run_sha256") != sha256(run_dir / "run.json") or
        metadata["generator_sha256"] != header["files"]["generator.safetensors"]):
        raise ValueError("Validation predictions do not belong to this checkpoint")
    values = checked_metrics(read_json(metrics_file), len(sources), len(references))
    evaluation_dir = metrics_file.parent.parent
    receipt_path = evaluation_dir / "evaluation.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        status = read_json(evaluation_dir / "evaluation_status.json")
        if (receipt.get("phase") != "val" or status.get("phase") != "completed" or
            receipt["prediction_sha256"] != sha256(prediction_dir / "prediction.json") or
            receipt["records_sha256"] != sha256(prediction_dir / "records.json") or
            receipt["condition_manifest_sha256"] != sha256(data_cache / "manifest.json") or
            receipt["data_manifest_sha256"] != sha256(data_manifest)):
            raise ValueError("Validation evaluation is incomplete or its provenance changed")
        if set(status["files"]) != set(REPORT_FILES):
            raise ValueError("Incomplete validation report manifest")
        for name, digest in status["files"].items():
            if sha256(metrics_file.parent / name) != digest:
                raise ValueError("Validation metric report changed")
        evidence = "completed evaluation with hashed provenance"
    elif import_existing:
        _check_exported_inputs(evaluation_dir, sources, references, data_manifest)
        # A legacy JSON could be written before a plotting failure. Require the complete report.
        for name in ("edge_threshold_metrics.csv", "metric_report.png", "metric_report.pdf"):
            if not (metrics_file.parent / name).is_file():
                raise ValueError("Legacy evaluation has incomplete reports")
        evidence = "imported existing scores; full validation sources/exports/predictions verified; scores not recomputed"
    else:
        raise ValueError("Missing completed evaluation provenance")
    cohort = dict(config=run["config"], prior_sha256=run["prior_sha256"],
                  source=run["data"]["source"], validation_manifest_sha256=sha256(data_cache / "manifest.json"),
                  metric_environment={k:values.get(k) for k in
                    ("metric_protocol", "device", "torch", "torchmetrics", "torch-fidelity", "scikit-image",
                     "batch_size", "kid_subset_size", "kid_subsets", "edge_canny_sigma", "edge_high_thresholds", "edge_low_threshold_rule")})
    run_sha = sha256(run_dir / "run.json")
    return dict(id=f"{run_sha}:{step}", run_dir=stored_path(run_dir), run_sha256=run_sha,
                checkpoint=checkpoint.relative_to(run_dir).as_posix(), step=step, phase="val",
                generator_sha256=header["files"]["generator.safetensors"],
                metrics={key:values[key] for key in METRIC_KEYS}, num_tir=len(sources), num_rgb=len(references),
                prediction_dir=stored_path(prediction_dir), prediction_sha256=sha256(prediction_dir / "prediction.json"),
                metrics_file=stored_path(metrics_file), metrics_sha256=sha256(metrics_file), evidence=evidence,
                cohort=cohort)


def register_best(folder, record, *, policy="structure"):
    """A single atomic best.json holds all winners and the complete selection history."""
    from filelock import FileLock
    if policy not in POLICIES or record.get("phase") != "val":
        raise ValueError("Choose a supported validation-only best policy")
    if any(isinstance(record["metrics"].get(key), bool) or
           not isinstance(record["metrics"].get(key), (int, float)) or
           not math.isfinite(record["metrics"][key]) for key in METRIC_KEYS):
        raise ValueError("Refuse nonfinite best metrics")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "best.lock"), timeout=0):
        path = folder / "best.json"
        if path.is_file():
            state = read_json(path)
            if state.get("format") != "D_best_selection_v1" or state["policy"] != policy or state["cohort"] != record["cohort"]:
                raise ValueError("Best selection must preserve its policy, data, config and metric environment")
        else:
            state = dict(format="D_best_selection_v1", policy=policy, rule=POLICY_DESCRIPTION[policy],
                         cohort=record["cohort"], records=[], best={})
        existing = next((r for r in state["records"] if r["id"] == record["id"]), None)
        if existing is not None and existing != record:
            raise ValueError("Previously registered checkpoint/validation evidence changed")
        if existing is None:
            state["records"].append(record)
        state["records"].sort(key=lambda r:(r["step"], r["run_sha256"]))
        state["best"] = winners(state["records"], policy)
        write_json(path, state)
        # A human-readable derivative; best.json is the authoritative transaction.
        temporary = folder / "validation_history.csv.tmp"
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("step", *METRIC_KEYS, "primary_best", "structure_best", "fid_best", "kid_best", "preview"))
            writer.writeheader()
            for item in state["records"]:
                writer.writerow(dict(step=item["step"], **item["metrics"],
                    **{f"{kind}_best":state["best"][kind] == item["id"] for kind in ("primary", *POLICIES)},
                    preview=str(restore_path(item["prediction_dir"]) / "index.html")))
        temporary.replace(folder / "validation_history.csv")
        return state


def selected_record(best_file, kind="primary"):
    state = read_json(best_file)
    if (state.get("format") != "D_best_selection_v1" or state.get("policy") not in POLICIES or
        kind not in ("primary", *POLICIES) or state["best"] != winners(state["records"], state["policy"])):
        raise ValueError("Best selection is invalid or its ranking changed")
    record = next(r for r in state["records"] if r["id"] == state["best"][kind])
    run_dir = restore_path(record["run_dir"])
    checkpoint = run_dir / record["checkpoint"]
    if checkpoint.resolve().parent != (run_dir / "checkpoints").resolve():
        raise ValueError("Best checkpoint must remain inside its original archival directory")
    if record.get("phase") != "val" or sha256(run_dir / "run.json") != record["run_sha256"]:
        raise ValueError("Best run identity changed")
    header = checkpoints.verify(run_dir, checkpoint)
    if header["step"] != record["step"] or header["files"]["generator.safetensors"] != record["generator_sha256"]:
        raise ValueError("Best model weights changed")
    return run_dir, checkpoint, record


def validation_identity(args, source_identity):
    cache = getattr(args, "validation_cache", None)
    if cache is None:
        if getattr(args, "validation_every", 0):
            raise ValueError("--validation-every requires a complete --validation-cache")
        return None
    dataset, _, references = full_validation(cache, args.data_manifest, source_identity=source_identity)
    if min(len(dataset), len(references)) < 50:
        raise ValueError("Full validation needs at least 50 images per domain for the fixed KID protocol")
    every = getattr(args, "validation_every", 0) or args.save_every
    if every < 1:
        raise ValueError("Validation interval must be positive")
    paths = ("D/unpaired/validation.py", "D/unpaired/evaluate.py", "D/unpaired/report.py",
             "D/unpaired/select_best.py", "D/metric_inputs.py", "scripts/evaluate_distribution.py", "scripts/render_metrics_figure.py")
    return dict(format="D_training_validation_v1", cache_manifest_sha256=sha256(Path(cache) / "manifest.json"),
                every=every, policy=getattr(args, "best_policy", "structure"), num_tir=len(dataset), num_rgb=len(references),
                metric_environment=metric_environment(getattr(args, "validation_device", "cpu")),
                code_sha256={p:hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths})


class TrainingValidation:
    def __init__(self, folder, args, identity):
        self.folder, self.args, self.identity = Path(folder), args, identity
        self.dataset, _, _ = full_validation(args.validation_cache, args.data_manifest)

    def due(self, step, total):
        return step > 0 and (step % self.identity["every"] == 0 or step == total)

    def run(self, engine, step, log):
        from .report import preview
        checkpoint = self.folder / "checkpoints" / f"step_{step:06d}"
        header = checkpoints.verify(self.folder, checkpoint)
        run_sha = sha256(self.folder / "run.json")
        identifier = f"{run_sha}:{step}"
        best_file = self.folder / "best.json"
        if best_file.is_file():
            state = read_json(best_file)
            if (state.get("format") != "D_best_selection_v1" or state["policy"] != self.identity["policy"] or
                state["cohort"]["validation_manifest_sha256"] != self.identity["cache_manifest_sha256"] or
                state["best"] != winners(state["records"], state["policy"])):
                raise ValueError("Saved best selection differs from this training validation policy/data")
            if any(r["id"] == identifier for r in state["records"]):
                return
        # Validation must not change training RNG, replay pools, modes or diagnostics.
        rng, retry = checkpoints.capture_rng(), engine.retry_state()
        flags = [(module, module.training) for module in engine.generator.modules()]
        diagnostics = [(obj, key, getattr(obj, key)) for obj, keys in
                       ((engine.generator, ("output_diagnostics",)),
                        (getattr(engine.generator, "fusion", None), ("diagnostics", "collect_diagnostics")))
                       if obj is not None for key in keys if hasattr(obj, key)]
        try:
            log.event("VALIDATE", f"update {step} | full val TIR={self.identity['num_tir']} RGB={self.identity['num_rgb']}"
                      f" | metric device={self.identity['metric_environment']['device']}")
            output = self.folder / "validation" / f"step_{step:06d}"
            prediction = output / "predictions/adapted"
            preview(engine.generator, self.dataset, prediction, resume=prediction.exists(), log=log,
                    metadata=dict(run_sha256=run_sha, phase="val", step=step,
                                  generator_sha256=header["files"]["generator.safetensors"]))
            evaluate_predictions(prediction, self.args.validation_cache, self.args.data_manifest, output / "evaluation",
                                 device=self.identity["metric_environment"]["device"], resume=True)
            record = candidate_record(self.folder, step, prediction, output / "evaluation/metrics/metrics.json",
                                      self.args.validation_cache, self.args.data_manifest)
            state = register_best(self.folder, record, policy=self.identity["policy"])
            selected = next(r for r in state["records"] if r["id"] == state["best"]["primary"])
            log.event("BEST", f"policy={state['policy']} | update {selected['step']}"
                      f" | Edge-F1={selected['metrics']['edge_f1_py']:.6f} FID={selected['metrics']['fid']:.4f}"
                      f" KID={selected['metrics']['kid_mean']:.6f} | {best_file}")
        finally:
            for module, flag in flags:
                module.training = flag
            for obj, key, value in diagnostics:
                setattr(obj, key, value)
            engine.restore_retry(retry)
            checkpoints.restore_rng(rng)
