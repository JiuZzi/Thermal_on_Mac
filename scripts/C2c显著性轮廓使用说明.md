# C2c 软轮廓：当前进度

**更新：** `--condition_mode soft_saliency` 现在是内容自适应 C2c v2；下面的位置先验说明与 `audit_c2_saliency.py` 仅对应保留的旧模式 `soft_saliency_position`。新 C2c 的方法、预览和实验隔离要求见 [`docs/c2c_adaptive_contour.md`](../docs/c2c_adaptive_contour.md)。

## 已实现的范围

- `soft_saliency_position` 模式将同一张处理后的红外图分别以 `sigma=0.7,1.0,1.6` 和 C2b 相同的三个阈值生成候选软轮廓。
- 细轮廓是前两种尺度的平均，粗轮廓是后两种尺度的平均。位置重要性图 `S` 在中间三分之一区域较高，过渡平滑；输出为 `S × 细轮廓 + (1-S) × 0.5 × 粗轮廓`。
- 生成器输入始终是 `[TIR, C2c软轮廓, 0]`。第三通道保留给后续置信度。使用 `--fusion_mode direct`。新 C2c 的重要性图由红外图局部结构和纹理生成，不用检测框。
- 旧版自动 `S` **仅是位置先验**。新版内容权重也不等于真正识别人、车、建筑或树叶；它只是无需标注的结构纹理启发式。
- 提供 FLIR *训练集* COCO 目标框的“理想框”诊断预览。框不能在正式训练或测试时当作预测显著性使用；FLIR 框也没有建筑物的像素轮廓。

## 在 Windows PowerShell 上预览

以下命令只读取 `trainB`，输出图片和统计，不启动训练。按你的实际路径调整 `$thermalRoot`、`$rawRoot`。

先查看新的内容自适应 C2c；同一张对照图也含 C2b、旧位置版和保边滤波备选：

```powershell
$thermalRoot = "D:\thermal"
Set-Location "$thermalRoot"

python scripts\audit_c2_adaptive.py `
  --input-dir "$thermalRoot\FLIR_datasets\trainB" `
  --output-dir "$thermalRoot\runs\edge_audit\c2c_adaptive_crop" `
  --sample-count 12 `
  --geometry center_crop_256
```

要直接比较本轮修订前后，同一批图片再运行：

```powershell
python scripts\audit_c2_adaptive_revision.py `
  --input-dir "$thermalRoot\FLIR_datasets\trainB" `
  --output-dir "$thermalRoot\runs\edge_audit\c2c_revision_crop" `
  --sample-count 12 `
  --seed 42 `
  --geometry center_crop_256
```

修订对照的列依次是红外、C2b、修订前结构纹理版、仅改连续密度、新 C2c、重要性图、增亮差异、变暗差异。`soft_saliency` 现在对应新 C2c v2。

下面的旧脚本仍只分析保留的位置版。`soft_saliency` 正式训练需要新的实验名，不能从位置版 checkpoint 续训。

```powershell
$thermalRoot = "D:\thermal"
$rawRoot = "$thermalRoot\FLIR_ADAS_v2"
Set-Location "$thermalRoot"

python scripts\audit_c2_saliency.py `
  --input-dir "$thermalRoot\FLIR_datasets\trainB" `
  --output-dir "$thermalRoot\runs\edge_audit\c2c_center_preview" `
  --sample-count 8 `
  --geometry center_crop_256
```

另做一次只用于分析的训练集目标框预览，帮助判断有语义区域指引时的轮廓变化：

```powershell
python scripts\audit_c2_saliency.py `
  --input-dir "$thermalRoot\FLIR_datasets\trainB" `
  --output-dir "$thermalRoot\runs\edge_audit\c2c_box_diagnostic" `
  --sample-count 8 `
  --geometry center_crop_256 `
  --oracle-coco "$rawRoot\images_thermal_train\coco.json"
```

分别查看 `quicklook.png`、`overview.png` 和 `individual` 目录。列顺序是：红外图、重要性图、细轮廓、粗轮廓、C2b、C2c、差异。`samples.csv` 和 `summary.json` 中的均值/差异不代表真实轮廓准确率。

如果以后已有**从红外图预测**的逐图显著性 PNG，可用 `--saliency-dir` 替代 `--oracle-coco` 查看它产生的轮廓。PNG 要与原 `trainB` 图像同尺寸，文件名为 `<红外图文件名去掉扩展名>.png`，脚本会施加相同的中心裁剪。

## 正式结论仍缺的部分

1. 新 C2c 已在原始红外、CycleGAN 循环中的合成红外及测试图上调用同一个确定性轮廓算法，但还需完整训练验证生成 RGB 的效果。
2. 建筑物需要额外的区域定义或标注，因为现有 FLIR 目标框不提供建筑物轮廓。
3. 在少量人工核对的目标边界上比较 C2b 与 C2c 的漏边、误边，并检查树叶/纹理抑制与建筑轮廓保留的取舍。

预览只是代码与视觉诊断，尚不能证明新 C2c 着色效果优于 C2b 或旧位置版。

## 新 C2c 的 40+40 epoch 命令

仅在确认预览图后启动。命令沿用既定 FLIR 划分、裁剪、训练预算和保存频率，v2 实验名称单独设为 `flir_v2_c2c_adaptive_v2_40_40`：

```powershell
$thermalRoot = "D:\thermal"
Set-Location "$thermalRoot\pytorch-CycleGAN-and-pix2pix"

python train.py `
  --dataroot "$thermalRoot\FLIR_datasets" `
  --name flir_v2_c2c_adaptive_v2_40_40 `
  --checkpoints_dir "$thermalRoot\runs\checkpoints" `
  --model conditioned_cycle_gan `
  --condition_mode soft_saliency `
  --fusion_mode direct `
  --dataset_mode unaligned `
  --direction BtoA `
  --input_nc 1 `
  --output_nc 3 `
  --lambda_identity 0 `
  --preprocess crop `
  --crop_size 256 `
  --batch_size 1 `
  --num_threads 4 `
  --n_epochs 40 `
  --n_epochs_decay 40 `
  --print_freq 100 `
  --display_freq 400 `
  --update_html_freq 1000 `
  --save_latest_freq 5000 `
  --save_epoch_freq 5
```
