#!/usr/bin/env bash
# Run the region-balance comparison against the completed D-v3 luma control.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

mode="${1:---check-only}"
if [[ "$#" -gt 1 || ( "$mode" != --check-only && "$mode" != --start && "$mode" != --resume ) ]]; then
    printf 'Usage: bash %s [--check-only | --start | --resume]\n' "$0" >&2
    exit 2
fi

run_name=D_v3_region_balance_u1000_seed42
run_dir="runs/D/unpaired/training/$run_name"
parent_dir=runs/D/unpaired/training/D_quality_v3_ubuntu_seed42
checkpoint="$parent_dir/checkpoints/step_011062"
run_record="$parent_dir/run.json"
if [[ "$mode" == --start && -e "$run_dir" ]]; then
    printf '已有实验 %s。需要继续时使用 --resume。\n' "$run_dir" >&2
    exit 1
fi
if [[ "$mode" == --resume ]]; then
    run_record="$run_dir/run.json"
fi

# The D venv inherits Torch from c1_sem512 and holds its own diffusion dependencies.
d_python="${D_PYTHON:-$PWD/runs/D/train_env_20261031/bin/python}"
if [[ -z "${D_PYTHON:-}" && ! -x "$d_python" ]]; then
    d_python="$PWD/../conda_envs/c1_sem512/bin/python"
fi
if [[ ! -x "$d_python" ]]; then
    printf '找不到 D 训练环境的 Python。请用 D_PYTHON=/完整路径/bin/python 指定原环境。\n' >&2
    exit 1
fi

mkdir -p runs/tmp runs/D/mpl_cache runs/training
export TMPDIR="$PWD/runs/tmp"
export TORCH_HOME="$PWD/runs/D/torch_cache"
export MPLCONFIGDIR="$PWD/runs/D/mpl_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

paths=$("$d_python" -u - "$run_record" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

import torch

from D.common import read_json, sha256

record = Path(sys.argv[1])
if not record.is_file():
    raise SystemExit(f"找不到实验记录 {record}")
run = read_json(record)
expected_config = dict(run["config"])
expected_config.pop("rgb_appearance_region_balance", None)
if read_json(Path("D/unpaired/config_quality_v3.json")) != expected_config:
    raise SystemExit("config_quality_v3.json 与实验记录不一致。请使用亮度对照组的原配置。")
versions = {**run["versions"], **run["validation"]["metric_environment"]["versions"]}
for name, expected in versions.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"原 D-v3 环境需要 {name}=={expected}，当前解释器 {sys.executable}") from None
    if actual != expected:
        raise SystemExit(f"{name} 版本为 {actual}，原 D-v3 使用 {expected}。请用 D_PYTHON 指定原解释器。")
if not torch.cuda.is_available():
    raise SystemExit(f"当前解释器无法使用 CUDA。{sys.executable}")
print(f"[GPU] {torch.cuda.get_device_name(0)} | {sys.executable}", file=sys.stderr)

train_sha = run["data"]["manifest_sha256"]
val_sha = run["validation"]["cache_manifest_sha256"]
caches = {}
for manifest in sorted(Path("runs/D/unpaired/conditions").rglob("manifest.json")):
    digest = sha256(manifest)
    status = manifest.with_name("status.json")
    if digest in (train_sha, val_sha) and status.is_file() and read_json(status).get("phase") == "completed":
        caches.setdefault(digest, manifest.parent)
for role, digest in (("训练", train_sha), ("完整验证", val_sha)):
    if digest not in caches:
        raise SystemExit(f"找不到原 D-v3 的{role}缓存。清单 SHA256 为 {digest}")

expected_fmb = run["data"]["source"]["recipe"]["tir_semantic_sha256"]
fmb = next((path for path in (
    Path("runs/pretrained/segformer_fmb_b3/best_model.pth"),
    Path("runs/checkpoints/semantic_epoch80_for_mac/best_model.pth"),
) if path.is_file() and sha256(path) == expected_fmb), None)
if fmb is None:
    raise SystemExit("找不到与原 D-v3 缓存对应的 FMB 权重。")

inception = Path(torch.hub.get_dir()) / "checkpoints/weights-inception-2015-12-05-6726825d.pth"
inception_sha = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
if not inception.is_file() or sha256(inception) != inception_sha:
    raise SystemExit(f"完整验证所需的 FID 权重缺失或损坏。{inception}")

print(caches[train_sha], caches[val_sha], fmb, sep="\t")
PY
)
IFS=$'\t' read -r train_cache val_cache fmb <<< "$paths"

command=("$d_python" -u -m D.unpaired.train
    --name "$run_name" --config D/unpaired/config_quality_v3.json
    --data-cache "$train_cache" --validation-cache "$val_cache"
    --data-manifest runs/D/ubuntu_assets_20261030/cycle_manifest.json
    --prior runs/D/ubuntu_assets_20261030/rgb_prior_seed42_loss_v1.json
    --semantic-checkpoint "$fmb" --rgb-appearance-region-balance
    --steps 1000 --save-every 1000 --log-every 20 --preview-every 0 --recovery-every 100
    --validation-every 1000 --validation-device cpu --best-policy structure)

case "$mode" in
    --check-only)
        exec "${command[@]}" --init-tir-checkpoint "$checkpoint" --check-only
        ;;
    --start)
        "${command[@]}" --init-tir-checkpoint "$checkpoint" --check-only
        "${command[@]}" --init-tir-checkpoint "$checkpoint" --verify-updates 20
        exec "${command[@]}" --resume
        ;;
    --resume)
        "${command[@]}" --resume --check-only
        exec "${command[@]}" --resume
        ;;
esac
