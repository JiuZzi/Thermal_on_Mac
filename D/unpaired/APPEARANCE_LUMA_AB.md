# D-v3 RGB 自配对亮度扰动对照

这组实验检验 RGB 自配对能否减少真实 TIR 的大范围热明暗在输出中的残留。开启 `--rgb-appearance-luma-augment` 时，仅在 `rgb_appearance` 分支扰动输入的灰度通道。对应的 RGB 目标与用于该分支的 E、P、Q 均来自同一张图。两组使用同一生成器、逆向网络、判别器检查点和四步采样，沿用原损失权重及 RGB 自配对频率。

每张图用随机 3×3 网格插值得到平滑的偏移量 `δ`，范围为 `[-0.15, 0.15]`。灰度按 `g' = g + δ·4g(1-g)` 变化，在每个位置保持对灰度的单调性，黑白端点固定。扰动由实验种子和更新步数确定。该策略写入运行配置，扩散噪声和数据顺序使用各自原有的随机流。训练日志中的 `luma_aug` 是本步灰度改变量的平均绝对值。

## Ubuntu 预检

在已有 D-v3 工程根目录解压源码更新包。两组从原实验第 11,062 步的**完整**检查点做权重初始化，优化器和计数从新实验开始。Mac 下载包适用于推理，这里使用 Ubuntu 原目录中的完整检查点。下面按原实验的清单哈希找训练及完整验证缓存。

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac

D_PYTHON="$PWD/runs/D/train_env_20261031/bin/python"
if [ ! -x "$D_PYTHON" ]; then
  D_PYTHON=/home/ubuntu/work/JiuZzi/conda_envs/c1_sem512/bin/python
fi
mkdir -p runs/tmp runs/D/mpl_cache runs/training
export TMPDIR="$PWD/runs/tmp"
export TORCH_HOME="$PWD/runs/D/torch_cache"
export MPLCONFIGDIR="$PWD/runs/D/mpl_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

D_PARENT=runs/D/unpaired/training/D_quality_v3_ubuntu_seed42
D_CKPT="$D_PARENT/checkpoints/step_011062"
D_PATHS=$("$D_PYTHON" - "$D_PARENT/run.json" <<'PY'
import sys
from pathlib import Path

from D.common import read_json, sha256

run = read_json(Path(sys.argv[1]))
cache_paths = {sha256(path): path.parent for path in
               sorted(Path("runs/D/unpaired/conditions").rglob("manifest.json"))}
try:
    train = cache_paths[run["data"]["manifest_sha256"]]
    val = cache_paths[run["validation"]["cache_manifest_sha256"]]
except KeyError as error:
    raise SystemExit(f"找不到原 D-v3 的缓存清单 SHA256 {error.args[0]}")

expected = run["data"]["source"]["recipe"]["tir_semantic_sha256"]
fmb = next((path for path in (
    Path("runs/pretrained/segformer_fmb_b3/best_model.pth"),
    Path("runs/checkpoints/semantic_epoch80_for_mac/best_model.pth"),
) if path.is_file() and sha256(path) == expected), None)
if fmb is None:
    raise SystemExit("找不到与原 D-v3 一致的 FMB 权重")
print(train, val, fmb, sep="\t")
PY
) &&
IFS=$'\t' read -r D_TRAIN_CACHE D_VAL_CACHE D_FMB <<< "$D_PATHS" &&
"$D_PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

看到 GPU 型号后，在同一个终端执行以下预检。预算先设为每组 1,000 次更新，在末尾使用完整验证集。前 20 次更新只用于检查数值和显存。

```bash
D_AB_COMMON=(
  --config D/unpaired/config_quality_v3.json
  --data-cache "$D_TRAIN_CACHE"
  --data-manifest runs/D/ubuntu_assets_20261030/cycle_manifest.json
  --prior runs/D/ubuntu_assets_20261030/rgb_prior_seed42_loss_v1.json
  --semantic-checkpoint "$D_FMB"
  --steps 1000 --save-every 1000 --log-every 20 --preview-every 0
  --recovery-every 100 --validation-cache "$D_VAL_CACHE"
  --validation-every 1000 --validation-device cpu --best-policy structure
)
D_AB_CONTROL=(--name D_v3_luma_control_u1000_seed42)
D_AB_AUG=(--name D_v3_luma_aug_u1000_seed42 --rgb-appearance-luma-augment)

"$D_PYTHON" -u -m D.unpaired.train "${D_AB_COMMON[@]}" "${D_AB_CONTROL[@]}" \
  --init-tir-checkpoint "$D_CKPT" --check-only &&
"$D_PYTHON" -u -m D.unpaired.train "${D_AB_COMMON[@]}" "${D_AB_AUG[@]}" \
  --init-tir-checkpoint "$D_CKPT" --check-only
```

两组都应出现 `[CHECK] passed` 和 `[VALIDATE] full val`。`[APPEAR]` 分别显示 `off` 和 `smooth_field_v1`。把预检命令末尾的 `--check-only` 换成 `--verify-updates 20` 即可按顺序做 GPU 短验证，训练预算仍为 1,000。两组均出现 `[VERIFIED]` 后，保留各自配置，用以下命令依次续训。

```bash
nohup bash -c '
  d_python=$1
  shift
  "$d_python" -u -m D.unpaired.train "$@" --name D_v3_luma_control_u1000_seed42 --resume &&
  "$d_python" -u -m D.unpaired.train "$@" --name D_v3_luma_aug_u1000_seed42 --rgb-appearance-luma-augment --resume
' bash "$D_PYTHON" "${D_AB_COMMON[@]}" > runs/training/D_v3_luma_pair_u1000_seed42.log 2>&1 &

tail -f runs/training/D_v3_luma_pair_u1000_seed42.log
```

两组都用第 1,000 步的完整验证结果比较。观察相同场景中的暗部和高亮，以及人车轮廓和颜色串扰，再决定后续是否单独调整色度监督。已完成的旧实验可用于权重初始化和推理。需要继续旧实验的优化器状态时，使用创建该实验时的原源码和配置。
