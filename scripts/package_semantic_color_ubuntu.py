#!/usr/bin/env python3
"""Build or verify a small tar.gz code update for the two semantic/color runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "semantic_color_ubuntu_update_manifest.json"
LEGACY = "runs/checkpoints/flir_v2_struct_sem512_protected_seed42_ubuntu_aligned_40_40"
EXCLUDED_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".venv",
                 "datasets", "checkpoints", "results", "imgs", ".ipynb_checkpoints"}
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".json", ".toml", ".cfg"}
EXTRAS = (
    "scripts/run_semantic_contour_ubuntu.py",
    "scripts/run_semantic_color.py",
    "scripts/run_semantic_color_pair_ubuntu.sh",
    "scripts/package_semantic_color_ubuntu.py",
    "scripts/check_semantic_color.py",
    "scripts/requirements_semantic_train.txt",
    "scripts/requirements_semantic_preview.txt",
    "docs/语义与颜色先验第一版.md",
    "docs/语义颜色两组Ubuntu训练.md",
    "docs/项目进度.md",
    "docs/结构语义保护版Ubuntu训练.md",
    "docs/结构语义保护版Windows训练.md",
    "README.md",
    "我的方法的详细描述.md",
)
REUSED = (
    "runs/pretrained/segformer_fmb_b3/best_model.pth",
    f"{LEGACY}/80_net_G_A.pth",
    f"{LEGACY}/80_net_G_B.pth",
    "FLIR_protocol_v2/manifest.csv",
    "FLIR_protocol_v2/protocol.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    return ROOT.joinpath(*path.parts)


def verify_installed():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    if manifest["format"] != "semantic_color_code_update_v1":
        raise ValueError("Unexpected update manifest")
    for category in ("files", "required_existing"):
        for relative, expected in manifest[category].items():
            path = checked_path(relative)
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Missing or different {category}: {relative}")
        print(f"Verified {len(manifest[category])} {category} files.", flush=True)
    print("Code, original semantic weight, legacy G_A/G_B and split protocol match. CUDA/data checks are separate.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/exports/semantic_color_ubuntu_20261020.tar.gz")
    parser.add_argument("--verify-installed", action="store_true")
    args = parser.parse_args()
    if args.verify_installed:
        verify_installed()
        return
    output = args.output.expanduser().resolve()
    checksum = output.with_name(output.name + ".sha256")
    if not output.name.endswith(".tar.gz") or output.exists() or checksum.exists():
        parser.error("Choose a new .tar.gz output; existing archives/checksums are preserved")
    paths = set()
    for directory in (ROOT / "pytorch-CycleGAN-and-pix2pix", ROOT / "scripts/semantic_preview"):
        for path in directory.rglob("*"):
            relative = path.relative_to(ROOT)
            if (path.is_file() and not EXCLUDED_DIRS.intersection(relative.parts)
                    and (path.suffix in SOURCE_SUFFIXES or path.name in ("LICENSE", "NOTICE"))):
                if path.is_symlink():
                    raise RuntimeError(f"Source symlinks are not packaged: {relative}")
                paths.add(path)
    paths.update(ROOT / relative for relative in EXTRAS)
    for path in paths | {ROOT / relative for relative in REUSED}:
        if not path.is_file():
            raise RuntimeError(f"Missing required source or reusable prerequisite: {path}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    manifest = dict(format="semantic_color_code_update_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_git_commit=git.stdout.strip() if git.returncode == 0 else None,
                    extract_into="existing Thermal_on_Mac project root", model="semantic_color_cycle_gan",
                    modes=["semantic", "semantic_color"], seed=42, new_epochs=5, learning_rate=.00002,
                    initialization="old epoch-80 G_A/G_B; fresh discriminators and optimizer",
                    includes="source, launchers, docs, licenses; no images, weights, results or credentials",
                    files={}, required_existing={})
    for relative in REUSED:
        path = ROOT / relative
        manifest["required_existing"][relative] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as archive:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size, info.mtime, info.mode = len(data), int(path.stat().st_mtime), 0o644
            archive.addfile(info, io.BytesIO(data))
            manifest["files"][relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo(MANIFEST)
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(f"TAR: {output}\nSize: {output.stat().st_size / 2**20:.2f} MiB; {len(paths)} source files\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
