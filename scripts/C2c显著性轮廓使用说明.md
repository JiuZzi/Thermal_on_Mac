# C2c 显著性引导软轮廓：当前进度

## 已实现的范围

- `soft_saliency` 模式将同一张处理后的红外图分别以 `sigma=0.7,1.0,1.6` 和 C2b 相同的三个阈值生成候选软轮廓。
- 细轮廓是前两种尺度的平均，粗轮廓是后两种尺度的平均。位置重要性图 `S` 在中间三分之一区域较高，过渡平滑；输出为 `S × 细轮廓 + (1-S) × 0.5 × 粗轮廓`。
- 生成器输入始终是 `[TIR, C2c软轮廓, 0]`。第三通道保留给后续置信度。使用 `--fusion_mode direct`。
- 目前的自动 `S` **仅是位置先验**，不能识别人、车、建筑或树叶。代码可接收外部 `[0,1]` 显著性图进行预览，但 CycleGAN 正式训练流程尚未接入一个能在训练和测试时一致运行的红外语义预测器。
- 提供 FLIR *训练集* COCO 目标框的“理想框”诊断预览。框不能在正式训练或测试时当作预测显著性使用；FLIR 框也没有建筑物的像素轮廓。

## 在 Windows PowerShell 上预览

以下命令只读取 `trainB`，输出图片和统计，不启动训练。按你的实际路径调整 `$thermalRoot`、`$rawRoot`。

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

## 正式 C2c 仍缺的部分

1. 从红外图预测目标显著性，并在原始红外、CycleGAN 循环中的合成红外及测试图上使用同一预测器；绝不能依赖测试标签。
2. 建筑物需要额外的区域定义或标注，因为现有 FLIR 目标框不提供建筑物轮廓。
3. 在少量人工核对的目标边界上比较 C2b 与 C2c 的漏边、误边，并检查树叶/纹理抑制与建筑轮廓保留的取舍。

位置先验预览是代码与视觉诊断，尚不是证明显著性方法有效的 C2c 正式实验。
