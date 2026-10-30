# D：完整验证集选模与 best 检查点

主规则经用户确认：**Edge-F1 最高；完全同分时依次比较更低的 FID、更低的 KID、更早的步数**。代码名为 `structure`。不将不同量纲的指标随意加权，不按训练 loss 选择。`best.json` 同时保存 `structure`、`fid`、`kid` 三个候选及完整历史，`primary` 由 `--best-policy` 指定，默认是 `structure`。

这里的 best 是指定验证指标的最优检查点。Edge-F1 衡量整图边缘，不专门衡量行人/车辆正确性，也不能自动判断颜色是否自然；对比网页继续保留，供人工审查。

## 数据与训练行为

- 当前完整验证集为 TIR 690 / RGB 660。只生成 TIR 的 P/E/Q，RGB 参考图由原评估器读取。
- 验证缓存必须使用同一数据清单、语义教师、几何和清晰轮廓配方；12张预览或部分验证缓存会被拒绝。
- best 只接受 `phase=val` 的完整预测；112张 test 不参与选择。
- seed42、FID/KID、APCE-Py 与边缘阈值协议保持原样；KID 使用 subset50 × 100，Canny sigma1.0。
- 网络、输入 `[TIR,E,0]`、训练数据、九项损失、学习率和训练总步数均未改变；没有加入早停。
- 每次先保存完整归档检查点，再验证。即使验证频率与保存频率不同，该次检查点也会永久归档，不会被滚动恢复清理掉。
- 验证前后恢复 Python / NumPy / Torch / CUDA RNG、训练噪声与采样 RNG、历史图池、各子模块模式和诊断状态。独立指标子进程不参与训练反传。
- 默认在 CPU 计算指标，避免与训练中的 SD 模型争用 8GB 显存。完整验证会增加耗时，计入后续训练 ETA。
- 验证或画报告失败时，不更新 best；保留已完成的训练检查点。匹配参数 `--resume` 后，会先续完该步验证，再读取下一训练批次。

## 后续训练启用

先准备一次完整缓存。当前本地已经有 `val_full_seed42` 时跳过本段；准备中断后可在原命令末尾加 `--resume`。

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location "$thermalRoot"

& $dPython -m D.unpaired.prepare `
  --config D/unpaired/config.json `
  --roles val `
  --domains TIR `
  --val-limit 0 `
  --device cpu `
  --output-dir runs/D/unpaired/conditions/val_full_seed42
```

下面仍是原 V1、5轮27655步、相同RGB预热起点；使用新 run 名。

```powershell
& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_full_e5_valbest_seed42 `
  --config D/unpaired/config.json `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42 `
  --steps 27655 `
  --save-every 5531 `
  --log-every 10 `
  --preview-every 5531 `
  --recovery-every 100 `
  --validation-cache runs/D/unpaired/conditions/val_full_seed42 `
  --validation-every 5531 `
  --validation-device cpu `
  --best-policy structure
```

`--validation-every 0` 默认沿用 `--save-every`。省略 `--validation-cache` 会保持原来的训练与预览行为，不启动完整指标评估。训练开启后，验证缓存、频率、设备、库版本、指标实现和选择策略都写入 `run.json`；续训必须保持一致。旧代码/旧 run 的严格恢复检查继续生效，不应编辑旧 `run.json` 绕过。

产物：

```text
runs/D/unpaired/training/<run>/
  best.json                          # 原子提交的完整选择历史与所有候选
  validation_history.csv             # 便于查看的指标表，包含各轮对比网页路径
  checkpoints/step_XXXXXX/            # 真正的完整权重、优化器和RNG快照
  validation/step_XXXXXX/
    predictions/adapted/index.html
    evaluation/metrics/metrics.json
    evaluation/evaluation.json        # 数据、预测和指标环境溯源
    evaluation/evaluation_status.json # 全部报告成功后才标记 completed
```

`best.json` 引用真正的完整归档权重，不重复复制几百MB，也不改变 `latest.json`。导出时需要同时保留所选 run 的 `run.json` 和被引用的 `checkpoints/step_XXXXXX/` 完整目录。

## 已跑完的五轮直接登记

这一步校验已有690张预测、源数据、导出的RGB参考、检查点SHA和报告完整性，导入现有分数，不重新生成图片或计算指标。前三轮来自旧 run，后两轮来自稳定性恢复 run；同配方的来源都保留在记录中，旧实验文件不改动。

```powershell
& $dPython -m D.unpaired.select_best `
  --comparison-csv runs/D/unpaired/evaluations/full_e5_val_selection_seed42/comparison.csv `
  --data-cache runs/D/unpaired/conditions/val_full_seed42 `
  --output-dir runs/D/unpaired/evaluations/full_e5_val_selection_seed42/best_selection `
  --policy structure
```

重复导入相同证据是幂等的；已登记分数、来源、图像或检查点变化会报错。旧分数没有新的评估收据，因此会核验实际导出的输入像素及所有完整报告，并在记录里明确标记为历史分数导入，不冒充新计算结果。

## 使用选出的 best 测试

完成112张 test 缓存准备后，当前五轮选择结果可直接这样使用：

```powershell
& $dPython -m D.unpaired.infer `
  --best-file runs/D/unpaired/evaluations/full_e5_val_selection_seed42/best_selection/best.json `
  --data-cache runs/D/unpaired/conditions/test_seed42 `
  --phase test
```

对后续开启自动验证的新 run：

```powershell
& $dPython -m D.unpaired.infer `
  --run-dir runs/D/unpaired/training/D_tir_unpaired_full_e5_valbest_seed42 `
  --best `
  --data-cache runs/D/unpaired/conditions/test_seed42 `
  --phase test
```

默认读取 `primary`，可加 `--best-kind structure`、`--best-kind fid` 或 `--best-kind kid` 查看对应候选。`--best` / `--best-file` 与 `--step` 互斥。加 `--check-only` 只核对权重、身份与条件数量，不加载GPU模型。生成后仍按实际所选 run/step 的输出目录调用 `D.unpaired.evaluate`；测试结果不会回写为 best 的选模依据。

## 验证边界

CPU回归覆盖完整验证集限制、测试集隔离、权重/预测/指标来源绑定、原子best更新、不同策略与并列规则、评估中断恢复和随机状态保护。额外用实际Adam优化、Dropout及两域历史图池核对：中途插入验证后，下一次更新的三个网络权重与不中断更新逐值一致。这不代表跨GPU、操作系统或不同模型版本的逐位一致保证。
