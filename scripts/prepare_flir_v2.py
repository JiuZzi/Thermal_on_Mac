#!/usr/bin/env python3
"""Build a reproducible FLIR ADAS v2 split for unpaired TIR-to-RGB training.

The output layout is directly readable by both the official PyTorch CycleGAN
repository and PearlGAN:

    FLIR_datasets/
      trainA/  # daytime RGB
      trainB/  # nighttime 8-bit thermal images
      testA/   # held-out daytime RGB reference domain
      testB/   # held-out nighttime thermal inputs

This is an FLIR ADAS v2 adapted protocol, not the original PearlGAN FLIR
file-list split: the original FLIR_* names are absent from FLIR ADAS v2.
Every selected source image is recorded in CSV manifests for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_THERMAL_ROOT = Path("/Users/tanwenjie/代码/thermal")
DEFAULT_RAW_ROOT = DEFAULT_THERMAL_ROOT / "FLIR_ADAS_v2"
DEFAULT_OUTPUT_ROOT = DEFAULT_THERMAL_ROOT / "FLIR_datasets"
DEFAULT_PROTOCOL_ROOT = DEFAULT_THERMAL_ROOT / "FLIR_protocol_v2"

# A = daytime visible RGB; B = nighttime 8-bit thermal images.
# We use validation images only for testA/testB so no source frame crosses
# train/test boundaries.  Full held-out domains are retained for future FID/KID
# and bidirectional tests; they are not paired image-to-image ground truth.
SPLITS = (
    ("images_rgb_train", "day", "trainA", "rgb"),
    ("images_thermal_train", "night", "trainB", "thermal"),
    ("images_rgb_val", "day", "testA", "rgb"),
    ("images_thermal_val", "night", "testB", "thermal"),
)


@dataclass(frozen=True)
class Record:
    destination_split: str
    domain: str
    source_split: str
    source: Path
    destination_name: str
    hours: str
    scene: str
    video_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print fixed split counts without writing images or manifests.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional deterministic first-N limit for a smoke-test dataset only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed files. Existing files are otherwise left intact.",
    )
    return parser.parse_args()


def load_records(raw_root: Path, limit: int | None) -> list[Record]:
    records: list[Record] = []
    for source_split, required_hours, destination_split, domain in SPLITS:
        annotation_path = raw_root / source_split / "coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO metadata: {annotation_path}")
        images = json.loads(annotation_path.read_text(encoding="utf-8"))["images"]
        selected = [
            image
            for image in images
            if image.get("extra_info", {}).get("hours") == required_hours
        ]
        selected.sort(key=lambda item: item["file_name"])
        if limit is not None:
            selected = selected[:limit]
        for image in selected:
            extra = image.get("extra_info", {})
            source = raw_root / source_split / image["file_name"]
            records.append(
                Record(
                    destination_split=destination_split,
                    domain=domain,
                    source_split=source_split,
                    source=source,
                    destination_name=Path(image["file_name"]).name,
                    hours=extra.get("hours", ""),
                    scene=extra.get("scene", ""),
                    video_id=extra.get("video_id", ""),
                )
            )
    return records


def validate_records(records: Iterable[Record]) -> None:
    missing = [record.source for record in records if not record.source.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} selected images are missing. Examples:\n{preview}")
    duplicated = [
        (split, name)
        for (split, name), count in Counter(
            (record.destination_split, record.destination_name) for record in records
        ).items()
        if count > 1
    ]
    if duplicated:
        raise ValueError(f"Duplicate output names found: {duplicated[:10]}")


def pearl_geometry(image: Image.Image, domain: str) -> Image.Image:
    """Apply PearlGAN's documented offline geometry: 500x400 then 360x288 crop."""
    mode = "RGB" if domain == "rgb" else "L"
    image = image.convert(mode)
    image = image.resize((500, 400), Image.Resampling.BICUBIC)
    left = (500 - 360) // 2
    top = (400 - 288) // 2
    return image.crop((left, top, left + 360, top + 288))


def write_images(records: Iterable[Record], output_root: Path, overwrite: bool) -> tuple[int, int]:
    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        destination = output_root / record.destination_split / record.destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        with Image.open(record.source) as source_image:
            output = pearl_geometry(source_image, record.domain)
            output.save(destination, quality=95, subsampling=0)
        written += 1
        if index % 500 == 0:
            print(f"Processed {index} images...", flush=True)
    return written, skipped


def write_protocol(records: Iterable[Record], protocol_root: Path, limit: int | None) -> None:
    protocol_root.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    manifest_path = protocol_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "destination_split",
                "domain",
                "source_split",
                "source_path",
                "destination_name",
                "hours",
                "scene",
                "video_id",
            ),
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "destination_split": record.destination_split,
                    "domain": record.domain,
                    "source_split": record.source_split,
                    "source_path": str(record.source),
                    "destination_name": record.destination_name,
                    "hours": record.hours,
                    "scene": record.scene,
                    "video_id": record.video_id,
                }
            )
    counts = Counter(record.destination_split for record in rows)
    protocol = {
        "name": "FLIR ADAS v2 adapted unpaired TIR-to-RGB protocol",
        "source": "FLIR_ADAS_v2",
        "domains": {"A": "daytime RGB", "B": "nighttime 8-bit thermal infrared"},
        "offline_geometry": {
            "resize": [500, 400],
            "center_crop": [360, 288],
            "resampling": "bicubic",
        },
        "online_training_geometry": {"random_crop": [256, 256], "horizontal_flip": True},
        "counts": dict(sorted(counts.items())),
        "limit_per_split": limit,
        "notes": [
            "Unpaired domains are selected from official train and validation partitions.",
            "This is not the original PearlGAN file-list split.",
            "Thermal edge maps for strict PearlGAN reproduction require MCI and are not generated here.",
            "FLIR v2 detection boxes are not full-image semantic masks or calibrated boundary labels.",
        ],
    }
    (protocol_root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    records = load_records(args.raw_root, args.limit_per_split)
    validate_records(records)
    counts = Counter(record.destination_split for record in records)
    print("Fixed FLIR ADAS v2 selection:")
    for split in ("trainA", "trainB", "testA", "testB"):
        print(f"  {split}: {counts[split]}")
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return
    write_protocol(records, args.protocol_root, args.limit_per_split)
    written, skipped = write_images(records, args.output_root, args.overwrite)
    print(f"Done. Wrote {written} images; skipped {skipped} existing images.")
    print(f"Dataset root: {args.output_root}")
    print(f"Protocol root: {args.protocol_root}")


if __name__ == "__main__":
    main()
