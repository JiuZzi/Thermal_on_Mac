#!/usr/bin/env python3
"""Build a ZIP update with code, semantic dependencies, and the complete weight."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".venv",
                 "datasets", "checkpoints", "results", "imgs", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".ipynb"}
EXTRAS = (
    "scripts/run_semantic_contour_ubuntu.py",
    "scripts/package_semantic_contour_ubuntu.py",
    "scripts/requirements_semantic_train.txt",
    "scripts/requirements_semantic_preview.txt",
    "scripts/evaluate_distribution.py",
    "scripts/render_metrics_figure.py",
    "scripts/requirements_metrics.txt",
    "scripts/评估指标使用说明.md",
    "docs/结构语义保护版Ubuntu训练.md",
    "FLIR_protocol_v2/manifest.csv",
    "FLIR_protocol_v2/protocol.json",
    "runs/pretrained/segformer_fmb_b3/best_model.pth",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/exports/semantic_contour_ubuntu_20261014.zip")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        parser.error("Output/checksum already exists. Select a new --output filename.")

    paths = set()
    for directory in (ROOT / "pytorch-CycleGAN-and-pix2pix", ROOT / "scripts/semantic_preview"):
        for path in directory.rglob("*"):
            relative = path.relative_to(ROOT)
            if (path.is_file() and not EXCLUDED_DIRS.intersection(relative.parts)
                    and path.suffix not in EXCLUDED_SUFFIXES and path.name != ".DS_Store"):
                if path.is_symlink():
                    raise RuntimeError(f"Do not package a source symlink: {relative}")
                paths.add(path)
    for relative in EXTRAS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing required payload: {relative}")
        paths.add(path)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "extract_into": "Thermal_on_Mac project root, the parent of pytorch-CycleGAN-and-pix2pix",
        "model": "semantic_contour_cycle_gan", "epochs": "40+40", "seed": 42,
        "dataset_included": False,
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packaging {len(paths)} files, including the complete frozen semantic weight...", flush=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, relative)
            manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        archive.writestr("semantic_ubuntu_update_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(f"ZIP: {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    print(f"SHA256: {checksum}")
    print("Reuses server FLIR_datasets; excludes old checkpoints, outputs, caches, and raw datasets.")


if __name__ == "__main__":
    main()
