#!/usr/bin/env python3
"""Render a reusable paper-style summary figure from unified metric outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_edge_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def render_metric_report(metrics: dict, rows: list[dict[str, float]], output_prefix: Path, title: str) -> tuple[Path, Path]:
    """Write the PNG/PDF report; reused automatically by evaluate_distribution.py."""
    if not rows:
        raise ValueError("No edge-threshold rows supplied for report rendering.")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=(0.75, 1.75))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.93,
        f"Unpaired FLIR v2 | {metrics['num_generated']} generated TIR-to-RGB images | "
        f"{metrics['num_reference_rgb']} daytime RGB references",
        ha="center", va="center", fontsize=10, color="#555555",
    )

    cards = [
        ("KID ↓", metrics["kid_mean"], "lower is better", "{:.4f}", "#3B6FB6"),
        ("FID ↓", metrics["fid"], "lower is better", "{:.2f}", "#3B6FB6"),
        ("APCE-Py ↑", metrics["apce_py"], "higher is better", "{:.4f}", "#C77700"),
        ("Edge-F1-Py ↑", metrics["edge_f1_py"], "higher is better", "{:.4f}", "#23854B"),
    ]
    for index, (label, value, direction, pattern, color) in enumerate(cards):
        axis = fig.add_subplot(grid[0, index])
        axis.set_axis_off()
        axis.add_patch(Rectangle((0.04, 0.08), 0.92, 0.78, fill=False, linewidth=1.3, edgecolor=color))
        axis.text(0.5, 0.66, label, ha="center", va="center", fontsize=12, fontweight="bold")
        axis.text(0.5, 0.41, pattern.format(value), ha="center", va="center", fontsize=20, fontweight="bold", color=color)
        axis.text(0.5, 0.20, direction, ha="center", va="center", fontsize=9, color="#555555")

    axis = fig.add_subplot(grid[1, :])
    threshold = [row["high_threshold"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall_apce"] for row in rows]
    f1 = [row["f1"] for row in rows]
    axis.plot(threshold, precision, label="Edge precision", linewidth=2.2, color="#3B6FB6")
    axis.plot(threshold, recall, label="Edge recall (APCE)", linewidth=2.2, color="#C77700")
    axis.plot(threshold, f1, label="Edge F1", linewidth=2.2, color="#23854B")
    axis.set_title("Edge consistency across 99 Canny threshold pairs", pad=10)
    axis.set_xlabel("Canny high threshold (low threshold = 0.5 × high threshold)")
    axis.set_ylabel("Mean score over valid TIR images")
    axis.set_xlim(0.01, 0.99)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FID/KID/APCE/Edge-F1 metric report PNG and PDF.")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--edge-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="Output path without extension, e.g. runs/evaluations/name/metric_report")
    parser.add_argument("--title", default="NTIR-to-RGB evaluation report")
    args = parser.parse_args()
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    rows = load_edge_rows(args.edge_csv)
    png_path, pdf_path = render_metric_report(metrics, rows, args.output_prefix, args.title)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
