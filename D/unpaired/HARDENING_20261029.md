# 2026-10-29：稳定性修复与训练输出重新整合

起点为合并提交`f2e5398059315fb886f3199074ae402f5e8299ab`。零方差sqrt修复已存在，但合并时遗漏了Mac侧日志入口、加权诊断和V2可靠性平滑调用。本记录对应本地修复，不把CPU检查当作Windows/CUDA长期训练保证。

## 修复范围

- 保留sqrt前的`1e-8`下限，检查原始及实际加权目标、各分支梯度，AMP允许小于1。
- 失败重试恢复显式噪声、局部位置、历史池及全局RNG；外层仍使用同一对TIR/RGB，成功后才推进计数。
- G/D共同提交前检查完毕；提交期间或已提交但未确认计数时异常，不再标为“优化器未改变”，也不生成计数不确定的紧急存档。
- 保存前检查模型、优化器历史和AMP。恢复时先验证全部网络张量和优化器，再修改当前网络。磁盘错误保留旧latest、清理本次未完成的临时目录；原始异常不被第二次保存错误掩盖。
- 保留每100个成功更新的滚动恢复、最近两个恢复点、长期存档和安全阶段紧急存档。CPU检查不再仅因有NVIDIA显卡而初始化CUDA上下文。
- 接回V2的`quality_soft`；V1保持原平滑定义。统一clamp前数值检查，拒绝将NaN/Inf保存成普通黑图；预览失败也还原train/eval状态。
- RGB预热拒绝保存非有限适配器；评估导出前拒绝会互相覆盖的RGB参考文件名。

网络容量、V1目标及权重、学习率计划、划分和P/E/Q生成算法未变。有效语义缓存可以复用。

## 训练输出

每个`--log-every`周期、本次首次和最后一次成功更新输出三行：

```text
[TRAIN  ] 当前步/总步 (百分比) | 秒/更新 | elapsed | ETA | VRAM peak/resv/free
[LOSS   ] G=实际加权总目标 | D_RGB D_TIR | cycle_TIR cycle_RGB
[AUX    ] edge content semantic color smooth color_w replay
```

数值来自当次更新，不是区间平均；ETA按本次启动的成功更新数计算，并计入重试与后续I/O时间。显存区分张量峰值、当前保留量和整卡剩余量；未安排回放时显示`off`。

`loss.jsonl`保留完整标量、加权分项、分支残差/梯度/覆盖率、AMP、重试、速度和显存。`train.log`保存摘要和加载/保存/预览/异常事件；失败细节保存在`numerical_events.jsonl`、`failure.json`。`--log-details`额外打印全部数值。第三方库直接写stderr的提示仍需通过启动命令重定向保留。

## 旧Windows实验采用方式

先正常停止旧进程，保留最后完整状态，再更新源码。源码已变，不能直接对旧run使用`--resume`或修改旧run.json；使用新run的完整状态迁移。

`recovery_sources.json`记录三个已审核起点的完整源哈希并识别LF/CRLF。迁移仍核对配置、数据、教师/先验、依赖版本及计划，只接受已审核来源与补丁涉及文件。旧V2的`quality_soft`曾未实际启用，修正会改变其实际目标，应另开显式warm-start对照，不按纯数值修复迁移。

下面针对已有`stable_v1`，读取实际latest指向的archive/recovery/emergency目录。`stable_v2`为补丁run名，不自动切换融合配方`D_unpaired_v2`。

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location $thermalRoot
$oldRun = "runs/D/unpaired/training/D_tir_unpaired_full_e5_seed42_stable_v1"
$record = Get-Content "$oldRun/run.json" -Raw | ConvertFrom-Json
$latest = Get-Content "$oldRun/latest.json" -Raw | ConvertFrom-Json
$checkpoint = Join-Path $oldRun $latest.checkpoint

& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_full_e5_seed42_stable_v2 `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --recover-from-checkpoint $checkpoint `
  --steps $record.schedule.steps `
  --save-every $record.schedule.save_every `
  --log-every $record.schedule.log_every `
  --preview-every $record.schedule.preview_every `
  --recovery-every 100 `
  --verify-updates 2
```

通过后保持相同name/缓存/计划，去掉`--recover-from-checkpoint`和`--verify-updates`，增加`--resume`。保留三网络、两个优化器、AMP、两域采样、历史池与RNG，不变成只加载权重的重新训练。Windows运行产物未回传Mac，本轮未操作Windows任务，恢复步数以它的latest为准。

## 验证与边界

- D系列73项检查通过，新增22项；覆盖小模型完整更新、V2实际调用、全局RNG重试、提交边界、非空Adam/图池/采样/RNG恢复后的后续更新逐值一致、磁盘故障和真实日志入口。
- 共享轮廓/语义/颜色/旧入口105项通过，共178项；Python源码全量AST、D JSON、CLI与diff检查通过。
- 同种子/输入的随机小模型V1对照：推理、一次完整更新后的95项G状态、2项反向状态、4项D状态及原有23项标量逐值一致。
- 证据位于`runs/D/hardening_20261029/`；该目录被gitignore，不随源码自动上传。

检查覆盖已发现问题及已构造的故障路径，不保证任意硬件/驱动/文件系统故障均可恢复，也不是完整预训练模型在Windows/CUDA上的长期训练或质量验证。重复非有限值仍会保护性停止，不取消检查、不换图绕过错误。
