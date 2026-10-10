# Mac MPS：JR0/JR1 前五轮轮廓预览

启动日期：2026-10-10。

## 停止记录

用户于 2026-10-10 决定改用 Windows，Mac 后台任务已停止，并确认训练、数据加载器和防休眠进程全部退出。最后日志为第 1 epoch、第 200 次更新；没有保存网络权重，JR1 尚未启动。本次没有五轮预览结果。Windows 两版应从头训练，不使用这次未保存的中途状态续训。

## 本次运行

- 解释器：`/Users/tanwenjie/miniforge3/bin/python`；MPS 在沙箱外可用，已通过两版单步训练与导出检查。
- 两版顺序运行：JR0（plain）五个完整 epoch → JR1（contrastive）五个完整 epoch → 固定目标预览。
- 使用全量 FLIR trainA=6191、trainB=2110；非配对加载器每轮 6191 次迭代，五轮每版 30955 次更新。
- 保留原模型规模、C2c 初始候选、256 裁剪、batch=1、4 workers、seed=2026 和 40+40 学习率计划。
- 新增 `--stop_after_epoch 5`，整轮结束并保存权重、训练状态后退出。没有改为 `n_epochs=5`，没有缩小数据或模型。
- 启动器强制检查 MPS，禁用算子回退，不会静默改为 CPU 训练。固定轮廓预处理本身仍有 CPU 运算。
- 运行中启用 `caffeinate -i` 防止空闲休眠，顺序任务退出后释放。
- 五轮结果是早期诊断，不是 80 epoch 正式效果；MPS 与 CUDA 的逐位一致性未验证。

## 位置与进度

实验名：

```text
flir_v2_jr0_c2c_joint_plain_mps_20261010_40_40
flir_v2_jr1_c2c_joint_contrastive_mps_20261010_40_40
```

相对项目根目录的位置：

```text
runs/training/joint_mps_preview_20261010/status.json
runs/training/joint_mps_preview_20261010/commands.json
runs/training/joint_mps_preview_20261010/plain.log
runs/training/joint_mps_preview_20261010/contrastive.log
runs/training/joint_mps_preview_20261010/preview/
runs/checkpoints/<实验名>/web/index.html
runs/checkpoints/<实验名>/5_net_G_A.pth
runs/checkpoints/<实验名>/5_joint_metadata.json
runs/checkpoints/<实验名>/5_joint_training.pt
```

后续阶段的日志、权重和预览只有执行到该阶段后才会出现。HTML 在首次显示保存时生成，展示当前训练裁剪；不是固定目标追踪。状态文件的 `phase=complete` 才表示两版与预览全部完成，`phase=failed` 表示对应阶段失败。

在 Mac 终端查看 JR0 实时日志：

```zsh
cd /Users/tanwenjie/代码/thermal
tail -f runs/training/joint_mps_preview_20261010/plain.log
```

固定审查清单沿用 `runs/edge_audit/important_contour_review_v1/review_manifest.json`。完成后预览包含初始图、JR0/JR1 及 RGB，并对重点目标做相同裁剪和最近邻四倍放大。训练样本审查用于诊断，不能作为泛化证据。

本次启动脚本：[run_joint_mps_preview.py](../scripts/run_joint_mps_preview.py)。具体参数以本次 `commands.json` 为准。不要重复启动同一实验；续训需要原参数和完整编号检查点，后续如继续训练，从 epoch 6 开始并调整或移除停止参数，保持 40+40 预算。
