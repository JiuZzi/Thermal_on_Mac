# Windows CUDA：JR0/JR1 前五轮预览

更新时间：2026-10-10。

Mac 任务已按用户要求停止，最后日志为第 1 轮第 200 次更新，没有保存权重，JR1 未启动。Windows 使用新实验名从头训练。

## 使用

将 `runs/exports/joint_contours_windows_20261010.zip` 解压到现有 thermal 项目根目录，例如 `D:\thermal`，覆盖包内对应源码。数据、训练权重不包含在包中。保留已有 FLIR_datasets 与 CUDA Python 环境，无需重新处理数据或重新安装 PyTorch。

迁移包包括当前 CycleGAN 核心源码、JR0/JR1、提前整轮停止选项、预览脚本、相关说明和之前的固定审查 manifest。启动脚本核对 CUDA 与 trainA=6191/trainB=2110；新实验目录已存在时拒绝覆盖。

在 Windows PowerShell 中执行，按实际位置修改 thermalRoot：

```powershell
conda activate base
$thermalRoot = "D:\thermal"

powershell -NoProfile -ExecutionPolicy Bypass `
  -File "$thermalRoot\scripts\run_joint_windows_preview.ps1" `
  -ThermalRoot "$thermalRoot"
```

ExecutionPolicy Bypass 只作用于这个 PowerShell 子进程。脚本顺序执行 JR0 五轮、JR1 五轮和预览。任何训练阶段失败会停止，不继续运行下一版。若要手动中断，在运行窗口按 Ctrl+C；中途未保存的更新不能作为完整 epoch 续训。

## 预算与输出

`n_epochs=40`、`n_epochs_decay=40`、`stop_after_epoch=5`。全量数据、原始模型规模、256 裁剪、batch=1、num_threads=4、seed=2026；不会把五轮当作 80 轮正式效果。

实验名：

```text
flir_v2_jr0_c2c_joint_plain_cuda_20261010_40_40
flir_v2_jr1_c2c_joint_contrastive_cuda_20261010_40_40
```

训练网页在 `runs/checkpoints/<实验名>/web/index.html`；第五轮权重与训练状态在对应目录。两版完成后对照图在 `runs/edge_audit/joint_contours_cuda_epoch005_20261010/`。

固定审查 manifest 的文件和图像哈希一致时导出重点目标四倍放大图；若 Windows 图像字节与 Mac 审查包不一致，会提示并输出全图对照，需要在 Windows 上重新建立固定目标审查清单。训练样本预览只用于诊断，最终评价沿用 testB。

启动脚本的参数与代码已在 Mac 检查，CUDA 实机运行及 PowerShell 运行结果尚未验证。GPU 具体速度取决于 Windows 显卡，不能根据 MPS 初期速度承诺耗时。
