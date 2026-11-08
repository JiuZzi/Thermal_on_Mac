# Ubuntu模型下载完成后：新开完整D训练

日期：2026-10-31。用户已完成模型下载。本轮继续当前V1、clarity_v1、语义与学得颜色先验，RGB预热1000步后正式训练27655步，验证集选best。5个等效epoch是既定完整数据预算，尚不是已验证的收敛轮数。补边和规则G仍为独立诊断。

下面在同一个Ubuntu Bash终端按顺序执行，每个阶段完成后再进行下一阶段。新目录使用20261031，保留旧实验。所有Python命令使用完整解释器路径，避免同名Conda环境混淆。

## 1. 公共设置与轻量D环境

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac

mkdir -p runs/tmp runs/training runs/D/torch_cache runs/D/pip_cache runs/D/mpl_cache runs/D/hf_home

export TMPDIR="$PWD/runs/tmp"
export TORCH_HOME="$PWD/runs/D/torch_cache"
export PIP_CACHE_DIR="$PWD/runs/D/pip_cache"
export MPLCONFIGDIR="$PWD/runs/D/mpl_cache"
export HF_HOME="$PWD/runs/D/hf_home"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/home/ubuntu/work/JiuZzi/conda_envs/c1_sem512/bin/python -m venv --system-site-packages runs/D/train_env_20261031

D_PYTHON="$PWD/runs/D/train_env_20261031/bin/python"

"$D_PYTHON" -m pip install -r D/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

"$D_PYTHON" -c "import sys,torch,torchvision,diffusers,transformers; print(sys.executable); print('Torch:',torch.__version__,'Diffusers:',diffusers.__version__,'Transformers:',transformers.__version__); print('CUDA:',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'"
```

这个轻量环境通过`--system-site-packages`复用原环境的Torch，在独立目录安装D依赖，不修改原c1_sem512。模型缓存与Python环境独立，继续使用项目下已下载的文件。新终端需重新执行公共设置并设置`D_PYTHON`；环境已建好后不必再创建或安装。

## 2. 确定原划分和颜色先验，检查数据

优先使用更新包副本；原文件作为回退。这里检查真实数据、先验和FMB，不依赖旧离线大包的asset_manifest。

```bash
DATA_MANIFEST=runs/D/ubuntu_assets_20261030/cycle_manifest.json
COLOR_PRIOR=runs/D/ubuntu_assets_20261030/rgb_prior_seed42_loss_v1.json

if [ ! -f "$DATA_MANIFEST" ]; then
  DATA_MANIFEST=runs/learned_color/data_seed42_v1/cycle_manifest.json
fi
if [ ! -f "$COLOR_PRIOR" ]; then
  COLOR_PRIOR=runs/learned_color/rgb_prior_seed42_loss_v1.json
fi

"$D_PYTHON" - "$DATA_MANIFEST" "$COLOR_PRIOR" <<'PY'
import sys
from pathlib import Path
from collections import Counter
from D.common import read_json, select_sources
from D.unpaired.semantics import default_semantic_checkpoint, legacy_models

manifest, prior = map(Path, sys.argv[1:])
assert manifest.is_file(), f"缺少数据清单：{manifest}"
assert prior.is_file(), f"缺少颜色先验：{prior}"
checkpoint = default_semantic_checkpoint()
assert checkpoint.is_file(), f"缺少完整FMB权重：{checkpoint}"
select_sources(manifest, phase="train", limit=1)
data = read_json(manifest)
counts = {k: dict(Counter(r["split"] for r in rows)) for k, rows in data["domains"].items()}
assert counts["A"]["train"] == 5531 and counts["B"]["train"] == 1420, "训练划分与当前5轮预算不同"
legacy_models()
from data.grouped_unaligned_dataset import validate_prior_partitions
from models.learned_color_prior import LearnedChromaPrior
from models.semantic_edges import FMB_CLASS_NAMES
validate_prior_partitions(data, manifest.parent, prior)
LearnedChromaPrior(prior, FMB_CLASS_NAMES, require_loss_support=True)
print("数据清单：", manifest)
print("颜色先验：", prior)
print("FMB：", checkpoint)
print("划分：", counts)
print("前置检查通过")
PY
```

出现“前置检查通过”后继续。若报错，处理该缺项再运行后续阶段。

## 3. 准备RGB预热缓存

```bash
"$D_PYTHON" -u -m D.prepare_rgb \
  --data-manifest "$DATA_MANIFEST" \
  --train-limit 256 \
  --val-limit 12 \
  --device cuda:0 \
  --output-dir runs/D/rgb_selfpairs/train256_val12_ubuntu_20261031
```

## 4. RGB预热1000步

```bash
nohup "$D_PYTHON" -u -m D.train \
  --name D_rgb_softE_pilot_seed42_ubuntu_20261031 \
  --rgb-cache runs/D/rgb_selfpairs/train256_val12_ubuntu_20261031 \
  --steps 1000 \
  --save-every 250 \
  > runs/training/D_rgb_warmup_ubuntu_20261031.log 2>&1 &

tail -f runs/training/D_rgb_warmup_ubuntu_20261031.log
```

日志出现`Saved ...step_001000`和`Next: python -m D.review`，并且该run的`status.json`为`completed`后，再进行下一步。`Ctrl+C`退出tail仅停止查看，不停止后台预热。

## 5. 正式训练与完整验证P/E/Q缓存

训练缓存使用全部5531张RGB和1420张TIR，并保留两域各12张预览；完整验证单独缓存690张TIR，660张RGB参考由评估器读取原清单。以下两条顺序执行：

```bash
"$D_PYTHON" -u -m D.unpaired.prepare \
  --config D/unpaired/config.json \
  --data-manifest "$DATA_MANIFEST" \
  --tir-limit 0 \
  --rgb-limit 0 \
  --val-limit 12 \
  --device cuda:0 \
  --output-dir runs/D/unpaired/conditions/full_seed42_ubuntu_20261031
```

```bash
"$D_PYTHON" -u -m D.unpaired.prepare \
  --config D/unpaired/config.json \
  --data-manifest "$DATA_MANIFEST" \
  --roles val \
  --domains TIR \
  --val-limit 0 \
  --device cuda:0 \
  --output-dir runs/D/unpaired/conditions/val_full_seed42_ubuntu_20261031
```

缓存可能占数十GB，用`df -h /home/ubuntu/work`检查数据盘。缓存中断后在原命令末尾加`--resume`，已完成的匹配缓存也通过原命令加`--resume`验证复用，不删除重建。预热、缓存、正式训练依次运行，避免同GPU并发。

## 6. 正式训练预检

```bash
"$D_PYTHON" -u -m D.unpaired.train \
  --name D_tir_unpaired_full_e5_valbest_ubuntu_20261031 \
  --config D/unpaired/config.json \
  --data-cache runs/D/unpaired/conditions/full_seed42_ubuntu_20261031 \
  --data-manifest "$DATA_MANIFEST" \
  --prior "$COLOR_PRIOR" \
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42_ubuntu_20261031 \
  --steps 27655 \
  --save-every 5531 \
  --log-every 10 \
  --preview-every 5531 \
  --recovery-every 100 \
  --validation-cache runs/D/unpaired/conditions/val_full_seed42_ubuntu_20261031 \
  --validation-every 5531 \
  --validation-device cpu \
  --best-policy structure \
  --check-only
```

须看到`CHECK ... passed`及`VALIDATE ... full val`。预检不创建正式run、不加载扩散模型；会核对缓存、FMB身份、数据清单、原先验、RGB初始化及完整验证集。

## 7. 正式后台训练

```bash
nohup "$D_PYTHON" -u -m D.unpaired.train \
  --name D_tir_unpaired_full_e5_valbest_ubuntu_20261031 \
  --config D/unpaired/config.json \
  --data-cache runs/D/unpaired/conditions/full_seed42_ubuntu_20261031 \
  --data-manifest "$DATA_MANIFEST" \
  --prior "$COLOR_PRIOR" \
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42_ubuntu_20261031 \
  --steps 27655 \
  --save-every 5531 \
  --log-every 10 \
  --preview-every 5531 \
  --recovery-every 100 \
  --validation-cache runs/D/unpaired/conditions/val_full_seed42_ubuntu_20261031 \
  --validation-every 5531 \
  --validation-device cpu \
  --best-policy structure \
  > runs/training/D_tir_full_e5_valbest_ubuntu_20261031.log 2>&1 &

tail -f runs/training/D_tir_full_e5_valbest_ubuntu_20261031.log
```

每5531次成功更新为当前数据的一轮；完整验证每轮一次。终端有TRAIN/LOSS/AUX、ETA、显存、SAVE/VALIDATE/BEST。第一次完整验证成功后写best.json和validation_history.csv，主规则为Edge-F1最高，同分比较更低FID、KID和更早step。best是已验证检查点中的指标最优，不是对颜色或人车结构正确性的保证。

## 8. 查看状态及中断恢复

```bash
cat runs/D/unpaired/training/D_tir_unpaired_full_e5_valbest_ubuntu_20261031/status.json
cat runs/D/unpaired/training/D_tir_unpaired_full_e5_valbest_ubuntu_20261031/best.json
```

正式训练中断后，恢复相同环境和变量，复用第7节命令，删去`--init-rgb-run`一行，加入`--resume`，其余参数完全一致，日志重定向改为`>>`追加。不要改run.json绕过身份校验。
