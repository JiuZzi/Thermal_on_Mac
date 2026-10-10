"""Run JR0/JR1 through five complete epochs on MPS, then export fixed reviews.

Keep the original full-data 40+40 schedule and model architecture. This is an
early training preview, not a reduced-data smoke experiment or a final result.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "pytorch-CycleGAN-and-pix2pix"


def run(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently train on CPU")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = {
        "plain": "flir_v2_jr0_c2c_joint_plain_mps_20261010_40_40",
        "contrastive": "flir_v2_jr1_c2c_joint_contrastive_mps_20261010_40_40",
    }
    checkpoints = ROOT / "runs/checkpoints"
    for name in names.values():
        if (checkpoints / name).exists():
            raise FileExistsError(f"Existing experiment: {name}")
    status = {
        "scope": "early_training_preview_quality_unverified", "device": "mps",
        "runner_pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_epochs_per_variant": 5, "schedule": "40+40", "names": names,
        "data": str(ROOT / "FLIR_datasets"), "completed": [],
    }
    commands = {}
    for variant, name in names.items():
        commands[variant] = [
            sys.executable, "-u", "train.py",
            "--dataroot", str(ROOT / "FLIR_datasets"),
            "--name", name, "--checkpoints_dir", str(checkpoints),
            "--model", "joint_contour_cycle_gan", "--joint_variant", variant,
            "--condition_mode", "soft_saliency", "--fusion_mode", "direct",
            "--dataset_mode", "unaligned", "--direction", "BtoA",
            "--input_nc", "1", "--output_nc", "3", "--lambda_identity", "0",
            "--lambda_contour_anchor", "1.0", "--lambda_contour_nce", "0.1",
            "--preprocess", "crop", "--crop_size", "256", "--batch_size", "1",
            "--num_threads", "4", "--seed", "2026",
            "--n_epochs", "40", "--n_epochs_decay", "40", "--stop_after_epoch", "5",
            "--print_freq", "100", "--display_freq", "400",
            "--update_html_freq", "1000", "--save_latest_freq", "5000",
            "--save_epoch_freq", "5",
        ]
    review = ROOT / "runs/edge_audit/important_contour_review_v1/review_manifest.json"
    commands["preview"] = [
        sys.executable, "-u", str(ROOT / "scripts/preview_joint_contours.py"),
        "--plain-checkpoint", str(checkpoints / names["plain"] / "5_net_G_A.pth"),
        "--contrastive-checkpoint", str(checkpoints / names["contrastive"] / "5_net_G_A.pth"),
        "--image-dir", str(ROOT / "FLIR_datasets/trainB"),
        "--review-manifest", str(review), "--num-images", "40", "--device", "mps",
        "--output-dir", str(output / "preview"),
    ]
    if not review.is_file():
        raise FileNotFoundError(review)
    (output / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")

    def record(phase, **changes):
        status.update(phase=phase, updated_at_utc=datetime.now(timezone.utc).isoformat(), **changes)
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")
        print(json.dumps(status, ensure_ascii=False), flush=True)

    # Prevent idle sleep while this explicitly requested local training is active.
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    child = None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        for phase, command in commands.items():
            with (output / f"{phase}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=TRAIN_ROOT if phase != "preview" else ROOT,
                                         stdout=log, stderr=subprocess.STDOUT, env=env)
                record(phase, child_pid=child.pid, log=str(output / f"{phase}.log"))
                code = child.wait()
            if code:
                record("failed", failed_phase=phase, returncode=code, child_pid=None)
                return code
            status["completed"].append(phase)
            record(f"{phase}_complete", child_pid=None)
        record("complete", preview_dir=str(output / "preview"))
        return 0
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        record("failed", error=repr(error), child_pid=None)
        raise
    finally:
        if awake.poll() is None:
            awake.terminate()
            awake.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs/training/joint_mps_preview_20261010")
    raise SystemExit(run(parser.parse_args()))
