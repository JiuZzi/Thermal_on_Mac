# D 短程条件适配：先看是否学到结构对应

这是旧RGB预热阶段的说明。真实TIR无配对训练已实现，入口 `D.unpaired.train`，机制及命令见 [unpaired/README.md](unpaired/README.md)。本页旧入口及产物保持可用。

这一步是 RGB 自配对预热：从同一张 RGB 提取灰度和当前 clarity_v1 E，以原 RGB 为去噪目标。先让条件分支适应软 E，再用固定 TIR 验证是否迁移。它不包含真实 TIR→RGB 配对训练，也没有完成 TIR 的无配对域适配。

现有软轮廓、结构/语义重要性、纹理抑制保持。RGB 重要性使用项目原有的 SegFormer-B2 Cityscapes 教师（同一固定 commit），19 类概率显式映射至 FMB 的 15 类；不把红外 FMB 模型用于 RGB。TIR 预览继续使用原 FMB 清晰轮廓缓存。没有新增语义特征注入、颜色 GMM 或置信度。

冻结 SD1.5 UNet、VAE、文本编码器及 ControlNet 主体，只更新 ControlNet 条件编码器和 down/mid 输出投影，共 12,566,160 参数。损失为标准扩散噪声预测 MSE，梯度经过冻结 UNet 回到条件分支。VAE 缓存后验均值/方差，每步重新采样，不把一次噪声永久固定为训练目标。

固定 seed42、batch=1、梯度累积2、AdamW、lr=1e-5、前50个优化步预热、1000次优化更新、每250步存档。1000步约读取2000个训练样本（可重复），不是1000个epoch。RGB 默认按既有 train/val 视频划分，固定选择256张训练图、12张验证图；不重新划分，不使用test训练。

## Windows 本机

本机已有的 D 隔离验证环境和模型缓存可以直接使用，先从项目根目录执行：

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location "$thermalRoot"

& $dPython -m D.prepare_rgb `
  --train-limit 256 `
  --val-limit 12 `
  --allow-download

& $dPython -m D.train `
  --name D_rgb_softE_pilot_seed42 `
  --steps 1000 `
  --save-every 250
```

准备数据会缓存 RGB 教师输出与 E，默认使用CPU。`--allow-download` 仅允许下载已有项目使用的固定 RGB 教师权重，不上传图像。已存在的数据缓存和训练目录不会覆盖；相同已完成的数据缓存直接复用，跳过prepare命令。

训练完成后生成前后对照：

```powershell
& $dPython -m D.review `
  --run-dir "$thermalRoot\runs\D\training\D_rgb_softE_pilot_seed42" `
  --step 1000

Start-Process "$thermalRoot\runs\D\training\D_rgb_softE_pilot_seed42\reviews\step_001000\index.html"
```

如果数据准备曾中断，保留已有缓存，用相同选择续跑：

```powershell
& $dPython -m D.prepare_rgb `
  --train-limit 256 `
  --val-limit 12 `
  --resume
```

续跑会校验已完成图像、源文件、教师权重和轮廓设置，跳过已验证的缓存。RGB 类别合并采用 float64 累加及单位质量归一化，修正 float32 求和的微小越界；NaN、明显越界或不正确的概率总和仍报错。原轮廓模块的严格检查保持不变。

预览会在同一个模型、输入、E、提示词、seed42、640×512、strength=.3下，先生成预训练控制分支结果，再加载本次训练的部分权重生成结果。包括 RGB 验证集和原固定12张TIR；不与旧的strength=.6结果混作训练增益。

## 怎么验收

网页中两组分别是：

- RGB：参考RGB → 灰度代理 → E → 训练前 → 训练后。检查学到软轮廓控制后能否保持布局、恢复合理颜色，注意RGB灰度不是实际TIR。
- TIR：原红外 → E → 训练前 → 训练后。先看人车位置与数量、肢体、道路、建筑，记录新增、消失、错位或把树替换成建筑的情况，再看颜色。

RGB验证和TIR都改善，才支持继续扩大这一路线；只有RGB改善则TIR域差异仍未解决，需要后续域适配；只有噪声MSE下降、图像对应没改善，不能算通过。首轮1000步没有改善也不能直接否定扩散方法，但应先定位原因，不能仅靠增加epoch。

噪声损失记录在 `runs/D/training/D_rgb_softE_pilot_seed42/loss.jsonl`，每250步的检查点在 `checkpoints/step_000250` 等目录，最终为 `step_001000/adapter.safetensors`。这是针对固定基础模型的部分ControlNet权重，不能当完整模型单独加载，没有自动选出best。

## 断点恢复

```powershell
& $dPython -m D.train `
  --name D_rgb_softE_pilot_seed42 `
  --steps 1000 `
  --save-every 250 `
  --resume
```

恢复最近完整检查点，包括可训练权重、AdamW、AMP、噪声RNG、采样顺序和位置。同一软硬件的小规模真实验证中，恢复后的第2步与连续训练第2步的42个权重张量逐元素一致；跨设备和库版本不承诺逐位复现。中断预览可对同一D.review命令增加 `--resume`。

## Ubuntu

上传最新源码包并解压到服务器工程根目录，在单独的 thermal_d 环境安装 `D/requirements.txt`。首次若没有扩散模型缓存，先 `python -m D.preview --download-models`。若没有固定TIR缓存，先按D/README的D.prepare命令准备val的12张，使用服务器原FMB权重。

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate thermal_d

python -m D.prepare_rgb \
  --train-limit 256 \
  --val-limit 12 \
  --allow-download

python -m D.train \
  --name D_rgb_softE_pilot_seed42 \
  --steps 1000 \
  --save-every 250

python -m D.review \
  --run-dir runs/D/training/D_rgb_softE_pilot_seed42 \
  --step 1000
```

从Jupyter打开该run的 `reviews/step_001000/index.html`，或下载整个review目录到本地查看。该目录包含前后图片和对照页。

## 已完成验证范围

已做真实RGB教师缓存、640×512的两步优化与检查点保存、断点恢复一致性、RGB/TIR训练前后网页；真实小模型测试验证了冻结UNet仍向适配器传递梯度、只有所选参数更新、权重安全保存加载和类别映射。正式1000步尚未代跑，效果需以本轮训练结果判断。640×512两步验证中PyTorch峰值分配约3.2GiB，不包括所有桌面/驱动显存，也不保证所有设备占用相同。
