# 微量学得颜色损失：实现与Ubuntu试跑

日期：2026-10-21。模型入口仍为`learned_color_cycle_gan`；旧人工颜色版与轮廓版不修改。默认新增损失权重0.01，训练效果尚未验证。

用户随后决定**从零训练**，当前应使用[包含RGB语义缓存的80轮完整流程](从零训练学得颜色先验Ubuntu完整流程.md)。本页下方的五轮命令保留为旧权重开发试跑记录。

36项实现／回归检查与真实旧权重的CPU两步更新通过；真实检查原始颜色损失对输出RGB梯度非零、合格像素比例0.56396、FMB冻结。检查使用128裁剪、小型判别器和开发先验，仅证明实现通路。[检查报告](../runs/smoke/learned_color_micro_loss_20261021/real_weight_two_steps/smoke_report.json)

## 损失定义

RGB教师只生成离线伪标签，原图Lab的a、b用来拟合冻结GMM。选定GMM后，仅对**按组平衡的训练色样**计算负对数密度NLL，取99%分位`max_nll`写入每类`loss_support`。val用于选K，test不参与拟合；旧JSON没有这一字段时，启用损失会明确要求重新fit，不用人工阈值冒充学得先验。

生成RGB经可微Lab转换得到ab。对TIR语义最高概率类别c计算：

`excess = relu(-log p_GMM(ab | c) - max_nll[c])`

`penalty = excess / (1 + excess)`

支持范围内不惩罚，不持续吸向中心；范围外惩罚有界，原始损失在[0,1]。仅支持的最高概率类别、Q>=0.5且Q未被类别边界屏蔽的像素生效，再乘冻结Q与该类语义概率。先平均每个类别内的合格像素，再平均每图活跃类别，最后平均batch；不按Q加权和作分母，以免可靠度被归一化抵消。大面积天空／道路不会单凭像素数量主导。

`L_G = L_GAN + L_cycle + effective_weight * L_color_raw`

有效权重在前1000次生成器更新从0逐步增至0.01，此后保持0.01；最大加权颜色损失不超过0.01。这是损失数值上限，不代表梯度占比或颜色变化百分比。计数器随G_A检查点保存；损失配置进入检查点签名，训练和导出须使用相同配置。`--color-loss-weight 0`关闭损失并保持原条件输入版的签名和加载方式。

损失不约束亮度L，不给网络饱和度越大越好的目标；不增加新教师，不更新FMB或GMM，也不更改joint512／第三零通道。高置信错误语义仍可能被强化；有界梯度与支持范围不能保证颜色多样性、自然度或跨域泛化。

## 1. 手工上传与解压

将tar.gz和同名.sha256上传到项目根目录：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512
sha256sum -c learned_color_micro_loss_ubuntu_20261021.tar.gz.sha256 &&
tar -xzf learned_color_micro_loss_ubuntu_20261021.tar.gz -C .
python scripts/package_learned_color_ubuntu.py --verify-installed
```

更新包只有源码、配置和文档，不包含RGB教师、先验、数据或GAN权重。尚未制作RGB缓存时，先执行[分组清单与RGB教师缓存步骤1–2](RGB学得颜色先验实现与训练.md)，检查按组预览，再执行下方fit。不要用既有67块开发种子替代全量缓存审查。

## 2. 拟合包含训练支持阈值的先验

已完成RGB缓存可以复用；重新fit不会重新跑教师。使用新输出名保留旧产物。

```bash
python -u scripts/learn_color_prior.py fit \
  --manifest runs/learned_color/rgb_teacher_seed42_v1/manifest.json \
  --config configs/learned_color/regularized_v1.json \
  --output runs/learned_color/rgb_prior_seed42_loss_v1.json
```

审查每类`loss_support`、独立组数量、train/val覆盖率和验证NLL；阈值仅定义色样支持，不是分割准确率或着色质量证明。若已存在同名先验并由新版fit生成，可直接复用，勿覆盖。

## 3. 预检与五轮试跑

两种颜色机制都使用同一份学得先验；主实验继续从已有**仅语义第5轮**G_A/G_B开始，判别器／优化器重新初始化。不要接着人工颜色版的权重启动这个对照。

```bash
python -u scripts/run_learned_color.py --check-only \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_micro_loss001_seed42_v1 --mode learned --gate-semantic \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000
```

预检通过后：

```bash
mkdir -p runs/training
nohup /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512/bin/python -u scripts/run_learned_color.py --train \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --init-condition-dir runs/checkpoints/flir_v2_semonly_from80_seed42_20261020_preview5 \
  --name rgb_prior_micro_loss001_seed42_v1 --mode learned --gate-semantic --epochs 5 \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000 \
  > runs/training/rgb_prior_micro_loss001_seed42_v1.log 2>&1 &
```

```bash
tail -f runs/training/rgb_prior_micro_loss001_seed42_v1.log
```

日志中`color_prior_raw`为未加权损失，`color_prior`为实际加入总损失的值，`color_coverage`为合格像素占全图的比例，`color_weight`为当前渐增权重。控制台只显示三位小数，前期微量损失显示0.000可能是舍入；epoch HTML或损失下降也不构成颜色改善证据。若coverage持续为0，先检查语义Q／支持类别／缓存，而非直接加大损失权重。

同预算对照使用同一命令，只改新name为`rgb_prior_learned_noloss_seed42_v2`、日志名和`--color-loss-weight 0`。同一GPU依次运行。保留原`zero_tokens`组可以检查条件使用，但该组不应启用颜色损失冒充无颜色信息基线。

## 4. 导出验证集

```bash
python -u scripts/run_learned_color.py --test-only --phase val --epoch 5 \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_micro_loss001_seed42_v1 --mode learned --gate-semantic \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000
```

输出到`runs/results/rgb_prior_micro_loss001_seed42_v1/val_5/`。无损失对照导出时用其name及`--color-loss-weight 0`。先比较颜色合理性、边界串色、结构、伪纹理和条件使用；验证后固定配置，再将phase换成test导出。旧起点权重可能已见过新val组，因此严格泛化仍需完全未见视频或外部数据，不为单张棕榈树调色。
