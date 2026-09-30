# C2c 图像驱动 ROI 候选：初步实验

## 目的与边界

原位置版 C2c 现保留为 `soft_saliency_position`；新的内容自适应 C2c 使用 `soft_saliency`。`soft_object_line` 仍是独立候选，不改变 C2b 与这两版 C2c：

1. 用 FLIR `trainB` 的人车框训练热红外检测器；验证按完整视频留出，未用 `testB` 选模型或阈值。
2. 检测框生成平滑对象 ROI。ROI 内侧重较细的多尺度软边，且稍弱化缺乏中尺度邻近支持的细边。
3. ROI 外侧重较粗的软边，并使用概率 Hough 长线作为“结构”代理。长线不能识别建筑，树枝、路面标线也可能被增强。
4. CycleGAN 输入仍为 `[红外, 软轮廓, 0]`，第三通道留给后续置信度；必须用 `--fusion_mode direct`。

与 C2b/C2c 相比，这个候选**额外使用了 FLIR 人车框监督和 COCO 预训练检测器**。若以后进行严格消融，必须单独说明辅助监督、固定检测器权重和阈值；不能把结果解释为单纯替换 ROI 算法的增益。

## 本轮可复现结果

检测器：SSDLite320 MobileNetV3，COCO 预训练；在 1200 张 `trainB` 图像上训练 4 epoch，从另外两个完整视频留出 216 张验证图。权重 `runs/detectors/flir_roi_ssdlite_1200/latest.pt` 的 SHA256：

`40000abb3636c2747cd26f40dd5293f4b210dc2d51d56b02675714ae9ad4bd80`

留出视频中共有 2098 个目标框。下面的框精确率与召回率按 IoU≥0.5 计算；“中心覆盖”只表示真值框中心落入预测 ROI，不代表轮廓准确。

| 检测阈值 | 框精确率 | 框召回率 | 真值中心 ROI 覆盖 | 平均 ROI 图像面积 |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 0.077 | 0.360 | 0.940 | 0.162 |
| 0.20 | 0.218 | 0.197 | 0.692 | 0.088 |
| 0.30 | 0.354 | 0.131 | 0.416 | 0.065 |

阈值 0.20 是当前候选默认值，只用于观察，并非最优结论。阈值降低可以覆盖更多对象中心，也会产生大量误框；升高阈值则明显漏掉对象。对高度至少 20 像素的目标，阈值 0.20 的框召回率为 0.354。FLIR 和现有 MSRS 分割标签都没有直接可用的建筑像素掩码，所以长线代理无法验证建筑轮廓的准确率。

可查看：

- `runs/detectors/flir_roi_ssdlite_1200/threshold_audit_final.json`：全量留出视频框与 ROI 指标。
- `runs/edge_audit/c2c_semantic_roi_1200_t10/quicklook.png`、`..._t20/quicklook.png`、`..._t30/quicklook.png`：三种阈值的红外图、预测框、ROI、长线、C2b、位置 C2c 和新候选并排预览。
- `runs/edge_audit/c2c_semantic_roi_1200_t20/overview.png`：8 张留出视频图像。

当前预览中人车漏检、误框以及树枝/路面长线增强都很明显。**本轮不建议用它替换位置 C2c，也未启动正式 80 epoch CycleGAN 训练。** 代码通过了 17 项单元测试、1 张图训练和 1 张图推理的运行检查；这些只能证明链路可运行，不能证明生成质量。

## 复现实验命令

在仓库根目录运行。训练检测器需要先准备 FLIR v2 数据和 `FLIR_datasets/trainB`；不提供 `--initial-weights` 时 TorchVision 会下载官方 COCO 权重。

```bash
python scripts/train_flir_roi_detector.py \
  --epochs 4 \
  --batch-size 4 \
  --max-train-images 1200 \
  --output-dir runs/detectors/flir_roi_ssdlite_1200

python scripts/evaluate_flir_roi_detector.py \
  --checkpoint runs/detectors/flir_roi_ssdlite_1200/latest.pt \
  --output runs/detectors/flir_roi_ssdlite_1200/threshold_audit_final.json

python scripts/audit_c2_semantic_roi.py \
  --checkpoint runs/detectors/flir_roi_ssdlite_1200/latest.pt \
  --score-threshold 0.20 \
  --sample-count 8 \
  --output-dir runs/edge_audit/c2c_semantic_roi_1200_t20
```

若只检查新 CycleGAN 模式入口，需传入同一检测器 checkpoint 与 `--condition_mode soft_object_line --fusion_mode direct --detector_score_threshold 0.20`。检测器权重**不包含在 CycleGAN checkpoint 中**，后续测试必须使用同一文件。正式对比还需先解决人车漏检和建筑/树枝混淆，再以固定数据、80 epoch 预算及同一评测协议比较 C2b、位置 C2c 与本候选。
