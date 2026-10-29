# 2026-10-28：非有限梯度修复与完整状态续训

2026-10-29更新：本文保留Windows首轮修复记录。合并后日志与V2调用遗漏已经重新整合，并补充恢复边界检查；采用最新源码时请按 [HARDENING_20261029.md](HARDENING_20261029.md) 迁移，原来的stable_v1命令对应当时源码。

原实验 `D_tir_unpaired_full_e5_seed42` 在20236次成功更新后异常停止，目标27655；最后完整存档16593。原实验的权重、日志、run.json和数据缓存保持不变。

## 已复现的原因

轮廓损失原先计算 `sqrt(clamp(local_variance, min=0))`。当生成图局部平坦，方差为0时，sqrt的导数无穷大；后续乘0也会产生NaN。这在FP32下同样发生，损失的前向数值仍可能正常。

已在原始第3轮权重和真实训练图上复现：前向图像有限、轮廓损失约0.596，但393个图像梯度非有限。autograd定位到 `SqrtBackward0`。诊断记录在 `runs/D/analysis/stability_20261028/real_checkpoint_original_probe.json`，原始代码已归档至该目录下的 `original_source/`。

AMP缩放也曾被错误地限制为不小于1，导致无法继续降低。PyTorch的[AMP说明](https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md)明确不保证缩放值总大于1。

## 修复内容

1. 方差在sqrt之前设置最小值1e-8，消除零点无限导数。正常方差区域公式不变；极平坦区域仅有数值下限差异。
2. AMP允许下降到1以下。异常时重试同一对图像、同一组噪声/局部位置，不通过换图绕过异常。
3. 分别检查TIR分支、RGB循环、RGB回放的损失和梯度；VAE原始输出检查发生在clamp之前，避免Inf被截断掩盖。
4. G、D两侧梯度全部检查通过后，才提交两个优化器。拒绝更新时回退噪声、位置采样和历史图池。
5. 默认每100次成功更新生成完整恢复快照，仅保留最近两个；原每轮长期检查点仍保留。
6. 重复数值异常或在可安全保存阶段手动中断时，保存最后成功更新的完整状态和失败报告。待处理批次的采样位置与RNG回退，恢复时重试该批次。若错误发生在优化器提交过程中，则保留最近的已验证周期快照，不保存不确定的半次更新。

模型输入 `[TIR,E,0]`、seed42、数据划分/缓存、基础模型、损失权重、学习率及总训练预算均保留。本次不切换BF16、不关闭轮廓/语义/颜色分支、不降低学习率。数学边界修复和新增恢复机制记录在新run的 `numerical_revision` 与 `recovery` 元数据中。

## 从旧实验迁移

不要直接修改旧run.json绕过源码检查。新入口 `--recover-from-checkpoint` 会新建run，并保留网络、优化器动量、AMP、采样顺序、历史图池和全部随机状态，继续原始全局步数。它会核对原数据、先验、版本、计划以及已审核的原始代码哈希；与只复制权重的 `--init-tir-checkpoint` 不同。

首次迁移命令如下。`--verify-updates 2` 只运行2次成功更新并退出保存，原目标仍为27655，不会将实验标为全部完成。已建立修复run时直接用下一段的恢复命令。

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location "$thermalRoot"

& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_full_e5_seed42_stable_v1 `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --recover-from-checkpoint runs/D/unpaired/training/D_tir_unpaired_full_e5_seed42/checkpoints/step_016593 `
  --steps 27655 `
  --save-every 5531 `
  --log-every 10 `
  --preview-every 5531 `
  --recovery-every 100 `
  --verify-updates 2
```

如果该新run已经建立并验证，继续训练时使用下面命令。不要再次传迁移源或初始化参数：

```powershell
& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_full_e5_seed42_stable_v1 `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --steps 27655 `
  --save-every 5531 `
  --log-every 10 `
  --preview-every 5531 `
  --recovery-every 100 `
  --resume
```

新实验以16593为恢复起点；首次2步验证后为16595，额外恢复测试会继续前进，以 `latest.json` 的实际step为准。原16594～20236未保存的3643次更新仍需重跑。`status.json` 的 `verified` 表示短恢复检查已完成；只有 `completed` 且step=27655才表示全部训练完成。

## 新的输出位置

- `checkpoints/step_XXXXXX/`：每轮长期存档及最终权重。
- `recovery/step_XXXXXX_时间标记/`：滚动完整恢复点，保留最近两个。
- `emergency/step_XXXXXX_时间标记/`：已知异常时的最后成功状态。
- `latest.json`：最近完整状态，可能指向上述任一目录。不要再假定latest必定位于checkpoints下。
- `numerical_events.jsonl`：失败分支、图像文件名、AMP缩放前后及最先发现的非有限梯度参数。
- `failure.json`：异常退出时的恢复位置和原因。

`D.unpaired.infer` 已支持从最新滚动/异常恢复点读取，也能用 `--step` 选择已有存档。数据缓存不需要重新生成。

## 验证范围

已完成：黑/白/平坦区域有限梯度测试；分支梯度及原D回归；AMP缩放降到0.5/0.25；D失败时G保持原权重；同批次噪声重试；滚动恢复只留两个且保留轮次存档；异常存档恢复成功步数与采样位置；拒绝未审核模型/数据/计划变更。

实际第3轮完整状态隔离续训40次更新，含RGB回放，40次均通过、无重试、梯度诊断全部有限，张量峰值约5.54GiB。见 `runs/D/analysis/stability_20261028/fixed_real_updates_status.json`。另完成8步实际CLI迁移验证，并在最终源码下建立新的短验证run及恢复加载测试。具体当前步数以最终验证记录和latest为准。

这些结果验证了本次数值错误及恢复机制，不能保证任何硬件/断电/文件系统故障永不发生。周期恢复点将正常中断损失限制在最近100次未保存更新以内；保存本身中断时回退到上一个完整点。
