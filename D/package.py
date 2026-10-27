"""Source-only Ubuntu upload, including unchanged contour/metric dependencies."""

import argparse
import hashlib
import io
from pathlib import Path
import tarfile

from .common import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checksum = args.output.with_name(args.output.name + ".sha256")
    if not args.output.name.endswith(".tar.gz") or args.output.exists() or checksum.exists():
        parser.error("Choose a new .tar.gz and preserve existing packages")
    files = set()
    for folder in ("D", "pytorch-CycleGAN-and-pix2pix/models", "pytorch-CycleGAN-and-pix2pix/util", "pytorch-CycleGAN-and-pix2pix/data",
                   "scripts/semantic_preview"):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts and path.name != "source_manifest.json"
                and (path.suffix in (".py", ".json", ".md", ".txt") or path.name.startswith("LICENSE") or path.name == "NOTICE")):
                files.add(path)
    for name in ("scripts/evaluate_distribution.py", "scripts/render_metrics_figure.py",
                 "pytorch-CycleGAN-and-pix2pix/LICENSE"):
        files.add(ROOT / name)
    manifest = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as archive:
        for path in sorted(files):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing/unsafe source file: {path}")
            payload = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            info = tarfile.TarInfo(name); info.size = len(payload); info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            manifest[name] = hashlib.sha256(payload).hexdigest()
        import json
        payload = (json.dumps(dict(format="D_source_upload_v1", files=manifest), indent=2) + "\n").encode()
        info = tarfile.TarInfo("D/source_manifest.json"); info.size = len(payload); info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    checksum.write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8", newline="\n")
    print(f"{args.output}: {len(files)} source files, {args.output.stat().st_size} bytes; {checksum}")


if __name__ == "__main__":
    main()
