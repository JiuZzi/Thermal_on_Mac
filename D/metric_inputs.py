"""Guard flat metric reference exports against silent filename collisions."""

from pathlib import Path


def rgb_reference_rows(manifest, phase):
    rows = [row for row in manifest["domains"]["A"] if row["split"] == phase]
    names = [Path(row["image"]).stem for row in rows]
    if not rows:
        raise ValueError(f"No RGB references for {phase}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate RGB reference stems would silently overwrite metric inputs")
    return rows
