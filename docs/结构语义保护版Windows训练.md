# 结构＋语义保护版：Windows CUDA 训练

更新时间：2026-10-13。

新增独立入口 `--model semantic_contour_cycle_gan`。第二通道为现有保护规则的 `joint512`，第一通道为原始 TIR，第三通道为零。此轮训练固定轮廓生成规则和冻结语义模型，训练 CycleGAN 如何使用这些条件；没有加入 JR0/JR1、置信度、颜色语义分支或新的损失。

## 1. Windows 准备

本次训练接入是预览提交之后的新代码，需要在 Mac 再提交并推送，Windows 拉取这次更新。

- 使用已有 CUDA PyTorch 环境，安装 `scripts/requirements_semantic_preview.txt`（OmegaConf 元数据依赖）。
- 完整外部权重放在项目的 `runs/pretrained/segformer_fmb_b3/best_model.pth`。该文件被 Git 忽略，需单独复制。
- 复用既有 `FLIR_datasets`：trainA=6191、trainB=2110、testA=696、testB=112。
- 训练在线提取语义，不读取之前的 40 张开发缓存；启动本训练不需要 `FLIR_protocol_v2` 或预览缓存。数据划分及预处理仍须与既定协议相同。

```powershell
$thermalRoot = "D:\thermal"
Set-Location "$thermalRoot"

git pull
python -m pip install -r scripts\requirements_semantic_preview.txt

powershell -NoProfile -ExecutionPolicy Bypass `
  -File "$thermalRoot\scripts\run_semantic_contour_windows.ps1" `
  -ThermalRoot "$thermalRoot" `
  -StopAfterEpoch 5
```

修改第一行实际路径。启动器检查 CUDA、依赖、权重存在和四个划分数量；新实验目录已存在会停止，避免覆盖。默认先训练五个完整 epoch，保留 40+40 的原学习率计划，并在第五轮自动导出固定 112 张 testB。

## 2. 训练行为

训练保持原 CycleGAN ResNet-9blocks、ngf/ndf=64、batch=1、256 随机裁剪、水平翻转、40+40、seed42、identity=0。网络直接使用 `[TIR,E_joint,0]`。

语义在当前裁剪/翻转后的 TIR 上计算：灰度复制三通道，作者短边至少 512 预处理，logits 恢复当前 TIR 坐标后 softmax。沿用原细粗候选、重要性保护、D、0.5 粗候选系数、0.4 纹理惩罚及 25% 细边保底。语义输入采用舍入 uint8；Canny 保留当前连续 TIR，循环中的合成 TIR 使用相同规则。

两个循环方向为：

```text
真实 TIR -> 在线 E_joint -> [TIR,E_joint,0] -> RGB -> 重建 TIR
真实 RGB -> 合成 TIR -> 重新计算 E_joint -> [合成TIR,E_joint,0] -> 重建 RGB
```

固定条件不传梯度，原始合成 TIR 通路保留梯度。语义模型始终 eval、requires_grad=False，不参与 GAN 初始化或优化，其权重不会写进每个 GAN 检查点。检查点保存语义配置签名；测试必须提供同一冻结权重。启动会打印 `Second channel: joint512` 及语义权重 SHA256。

每次更新增加真实/合成 TIR 的冻结语义前向，耗时需在 Windows 实机观察。在线裁剪改变语义上下文，与先前全图开发预览存在差异，需要实际检查。

## 3. 五轮后的输出

默认实验名：`flir_v2_struct_sem512_protected_seed42_cuda_20261013_40_40`。

- 训练网页：`runs/checkpoints/<实验名>/web/index.html`，新增 `contour_joint` 展示实际第二通道；训练网页中的图是当前训练裁剪。
- 完整第五轮权重：`runs/checkpoints/<实验名>/5_net_G_A.pth` 及其他网络文件。
- 固定测试网页：`runs/results/<实验名>/test_5/index.html`，包含原图、实际软轮廓、生成 RGB 与循环重建。
- 语义配置：对应检查点目录下的 `semantic_train.json` / `semantic_test.json`。

测试页中的 `real_B` 是无配对 RGB 样本，不是该热红外图对应的颜色真值。`real_A` 是输入 TIR，`fake_B` 是生成 RGB。

五轮属于早期使用能力诊断，不是正式 80 epoch 质量结果。固定规则与冻结语义不会随着这五轮更新；更新的是生成器和判别器。看重要目标在生成 RGB 中的轮廓保持、伪纹理和颜色自然度，不以轮廓亮度升高证明收益。同阶段对照需要同 seed42、骨干、数据、预算和测试设置。

## 4. 完整 80 epoch

如果决定直接完成 40+40，使用新实验名并关闭提前停止：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File "$thermalRoot\scripts\run_semantic_contour_windows.ps1" `
  -ThermalRoot "$thermalRoot" `
  -RunName "flir_v2_struct_sem512_protected_seed42_cuda_full80_20261013" `
  -StopAfterEpoch 0
```

启动器从头训练，不将已有五轮作为无缝续训。原 ConditionedCycleGAN 没有保存完整 optimizer/RNG 状态，不能把只加载网络权重的续训当作严格不中断训练。

## 5. 验证范围

新增测试覆盖当前裁剪/翻转输入、batch 对齐、第二/第三通道、冻结语义与 GAN 参数隔离、合成 TIR 的梯度通路、循环重算、检查点签名及同 seed 的 GAN 初始化。原融合与保护规则测试继续保留。

本次本地检查：新增9项训练检查、15项轮廓规则检查、17项原融合检查通过。完整真实语义权重在固定全图上在线生成的 `joint512` 与既有保护版浮点预览逐像素相同；GAN初始化和train模式不会改变冻结语义，加载外部权重不改变随机状态。真实FLIR的一次随机裁剪训练、检查点保存及CLI加载/导出通过。单步GAN使用ngf/ndf=8、6blocks和pixel判别器，供实现检查；Windows正式命令保持64、9blocks和原判别器。

本地 CPU 的实现验证和单步检查不证明正式 RGB 收益。Windows PowerShell/CUDA 实机运行需在 Windows 完成，不能由 Mac CPU 检查代替。
