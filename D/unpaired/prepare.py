"""Offline native-coordinate P/E/Q for independent TIR and RGB video splits."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from D.common import ROOT, EDGE_SOURCES, read_json, write_json, sha256, select_sources
from D.prepare_rgb import balanced, PROBABILITY_POLICY
from .protocol import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_MANIFEST, config_from, validate_arrays
from .semantics import ConditionBuilder, default_semantic_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--semantic-checkpoint", type=Path, default=default_semantic_checkpoint())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--roles", choices=("train", "val", "test"), nargs="+", default=["train", "val"])
    parser.add_argument("--domains", choices=("TIR", "RGB"), nargs="+", default=["TIR", "RGB"])
    parser.add_argument("--tir-limit", type=int, default=0, help="Training images; 0=all")
    parser.add_argument("--rgb-limit", type=int, default=0, help="Training images; 0=all")
    parser.add_argument("--val-limit", type=int, default=12)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu", help="Teacher inference only; clarity extraction stays on CPU")
    args = parser.parse_args()
    config = config_from(args.config)
    if min(args.tir_limit, args.rgb_limit, args.val_limit, args.test_limit) < 0:
        parser.error("Limits must be nonnegative; 0 means all")
    if args.output_dir.exists() != args.resume:
        parser.error("Use a new output-dir, or --resume for an existing cache")
    if not args.semantic_checkpoint.is_file():
        parser.error("Missing complete FMB best_model.pth")
    select_sources(args.data_manifest, limit=1)  # All paths and cross-modal video roles.
    data = read_json(args.data_manifest)
    selected = []
    for domain in sorted(set(args.domains)):
        for role in ("train", "val", "test"):
            if role not in args.roles:
                continue
            rows = [r for r in data["domains"]["B" if domain == "TIR" else "A"] if r["split"] == role]
            limit = (args.tir_limit if domain == "TIR" else args.rgb_limit) if role == "train" else \
                    (args.val_limit if role == "val" else args.test_limit)
            if limit:
                rows = balanced(rows, limit)
            for row in sorted(rows, key=lambda r:r["image"]):
                selected.append(dict(row, domain=domain))
    if not selected:
        parser.error("No selected images")
    torch.set_num_threads(4)
    builder = ConditionBuilder(args.semantic_checkpoint, rgb=True, allow_download=args.allow_download,
                               flip=config["flip_reliability"], device=args.device)
    from models.clarity_semantic_edges import CLARITY_SETTINGS
    recipe = dict(input="TIR01,clarity_E01,reserved_zero", size=[config["width"],config["height"]],
                  processing="bicubic resize only if needed, before P/E/Q; no random crop, flip or per-image stretching",
                  tir_semantic_sha256=builder.tir.info["sha256"], rgb_teacher=builder.rgb.info,
                  flip_reliability=config["flip_reliability"], semantic_backend=args.device,
                  contour_variant="clarity_v1", settings=CLARITY_SETTINGS,
                  rgb_probability_policy=PROBABILITY_POLICY,
                  code_sha256={p:sha256(ROOT / p) for p in EDGE_SOURCES})
    identity = json.loads(json.dumps(dict(dataset_id=data.get("dataset_id"), seed=42,
                                          source_manifest_sha256=sha256(args.data_manifest), recipe=recipe)))
    plan = [dict(source=r["image"], domain=r["domain"], role=r["split"], group=str(r["group"])) for r in selected]
    output = args.output_dir; seen = {}
    if args.resume:
        result = read_json(output / "manifest.json")
        status = read_json(output / "status.json")
        if result["identity"] != identity or result["selection"] != plan or status["total"] != len(selected):
            raise ValueError("Resume requires identical selection, recipe, semantic weights and source manifest")
        for index, frame in enumerate(result["frames"]):
            row = selected[index]
            source = (args.data_manifest.parent / row["image"]).resolve()
            if (frame["source"] != row["image"] or frame["domain"] != row["domain"] or
                frame["role"] != row["split"] or sha256(source) != frame["source_sha256"] or
                sha256(output / frame["condition"]) != frame["condition_sha256"]):
                raise ValueError("Previously prepared source/cache changed")
            with np.load(output / frame["condition"], allow_pickle=False) as cached:
                validate_arrays(cached["gray"], cached["E"], cached["P"], cached["Q"])
                content = hashlib.sha256(cached["RGB"].tobytes()).hexdigest()
            if content in seen and seen[content] != row["split"]:
                raise ValueError("Decoded duplicate leaks across roles")
            seen[content] = row["split"]
        print(f"Verified {len(result['frames'])}/{len(selected)} cached samples", flush=True)
    else:
        output.mkdir(parents=True); (output / "maps").mkdir()
        result = dict(format="D_unpaired_conditions_v1", identity=identity, selection=plan, frames=[])
        write_json(output / "manifest.json", result)
    start = len(result["frames"])
    for index, row in enumerate(selected[start:], start + 1):
        source = (args.data_manifest.parent / row["image"]).resolve()
        with Image.open(source) as im:
            image = im.convert("RGB"); native = list(image.size)
            if image.size != (config["width"], config["height"]):
                if image.width * config["height"] != image.height * config["width"]:
                    raise ValueError("Config would stretch image geometry; use the native aspect ratio")
                image = image.resize((config["width"],config["height"]), Image.Resampling.BICUBIC)
            rgb = np.asarray(image).copy()
        content = hashlib.sha256(rgb.tobytes()).hexdigest()
        if content in seen and seen[content] != row["split"]:
            raise ValueError("Decoded duplicate leaks across roles")
        seen[content] = row["split"]
        gray, edge, p, q = builder.compute(rgb, row["domain"])
        validate_arrays(gray, edge, p, q)
        stem = source.stem
        name = f"{row['domain']}_{row['split']}_{stem}"
        relative = f"maps/{name}.npz"
        with tempfile.NamedTemporaryFile(dir=output / "maps", suffix=".pending", delete=False) as temporary:
            np.savez_compressed(temporary, RGB=rgb, gray=gray, E=edge, P=p, Q=q)
            pending = Path(temporary.name)
        pending.replace(output / relative)
        result["frames"].append(dict(id=name, filename=source.name, stem=stem, domain=row["domain"], role=row["split"],
            group=str(row["group"]), source=row["image"], source_sha256=sha256(source), native_size=native,
            condition=relative, condition_sha256=sha256(output / relative)))
        write_json(output / "manifest.json", result)
        write_json(output / "status.json", dict(phase="preparing", completed=index, total=len(selected)))
        print(f"P/E/Q [{index}/{len(selected)}] {row['domain']} {row['split']} {source.name}", flush=True)
    write_json(output / "status.json", dict(phase="completed", completed=len(selected), total=len(selected)))
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
