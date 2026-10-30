"""Register already completed full-validation results without retraining or rescoring."""

import argparse
import csv
from pathlib import Path

from D.common import read_json
from .protocol import DEFAULT_MANIFEST
from .validation import POLICIES, candidate_record, register_best, selected_record


def import_comparison(comparison_csv, data_cache, output_dir, *, data_manifest=DEFAULT_MANIFEST, policy="structure"):
    with Path(comparison_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty validation comparison table")
    # Verify every candidate before publishing the first best pointer.
    records = []
    for row in rows:
        prediction = Path(row["Preview"]).parent
        metrics = Path(row["MetricsJSON"]) if row.get("MetricsJSON") else prediction.parent.parent / "evaluation/metrics/metrics.json"
        records.append(candidate_record(Path(row["RunDir"]), int(row["Step"]), prediction, metrics,
                                        Path(data_cache), Path(data_manifest), import_existing=True))
    cohorts = [r["cohort"] for r in records]
    if any(value != cohorts[0] for value in cohorts[1:]):
        raise ValueError("Compare only the same model recipe, validation split and metric environment")
    state = None
    for record in records:
        state = register_best(output_dir, record, policy=policy)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, default="structure")
    args = parser.parse_args()
    import_comparison(args.comparison_csv, args.data_cache, args.output_dir,
                      data_manifest=args.data_manifest, policy=args.policy)
    best_file = args.output_dir / "best.json"
    state = read_json(best_file)
    for kind in ("primary", *POLICIES):
        run, checkpoint, record = selected_record(best_file, kind)
        print(f"{kind}: step={record['step']} Edge-F1={record['metrics']['edge_f1_py']:.6f} "
              f"FID={record['metrics']['fid']:.4f} KID={record['metrics']['kid_mean']:.6f} | {checkpoint}", flush=True)
    print(f"Rule: {state['rule']}\nBest: {best_file}", flush=True)


if __name__ == "__main__":
    main()
