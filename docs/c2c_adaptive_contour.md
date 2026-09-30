# C2c 内容自适应软轮廓

## 当前选择

`--condition_mode soft_saliency` 现为内容自适应 C2c v2。旧位置先验的完整实现保留在 `--condition_mode soft_saliency_position`，用于复现旧预览或旧权重。C2b `soft_multi`、C1 `canny`、实验性检测框分支 `soft_object_line` 均未改动。

新 C2c **不使用检测框、不使用额外标注、不添加损失函数**。第三通道仍是全 0，留给后续置信度；输入 `[TIR, 软轮廓, 0]` 经 `--fusion_mode direct` 进入生成器。它的内容权重只是“保留细节的重要性”，不是边缘存在概率或置信度。

## 轮廓生成

1. 沿用 C2b 的 Canny 高阈值 `0.12,0.16,0.20`、低高比 `0.5` 和 1 像素高斯距离软化。在尺度 `0.7,1.0,1.6` 上提取软边，形成细尺度和粗尺度候选。
2. 从当前红外图计算局部梯度的结构张量、方向一致性与热对比；从**连续软边强度**计算局部密集度和细尺度相对粗尺度的多余边，不再使用旧版的 `edge>0.45` 二值密度。组合成连续权重 `R∈[0.18,0.85]`。
3. 用 `R` 融合细、粗候选，并对密集、方向不稳定的纹理施加至多 40% 的连续抑制。v2 另设 `E_final≥0.25 E_fine` 的细边保留底线；若提边器完全没发现边，该式也不会制造新边。算法常数固定在代码中，训练时不根据 `testB` 调参。

这仍然只是**无监督的结构纹理启发式**。它不能可靠识别“树叶”或“人车建筑”；高频建筑细节也可能被减弱，强树枝边缘也可能保留。特别要在预览中检查远处小行人和车轮廓。

## 预览与选择依据

上一版候选在同一批 `trainB` 图像上的历史预览：

- `runs/edge_audit/c2c_adaptive_selected_crop/quicklook.png` 与 `overview.png`：按 256 中心裁剪，对齐训练几何大小。
- `runs/edge_audit/c2c_adaptive_selected_full/quicklook.png` 与 `overview.png`：完整 360×288 图，对齐测试几何大小。
- `runs/edge_audit/c2c_adaptive_second_sample/quicklook.png` 与 `overview.png`：另一固定随机种子，检查第一组样本之外的情况。

图中依次展示红外、C2b、旧位置 C2c、内容权重、结构纹理候选、保边滤波候选权重、保边滤波候选及两候选差异。观察到结构纹理候选在部分树冠抑制杂边，同时保留远处人车和建筑主轮廓；保边滤波候选抑制更强，但细小轮廓也变弱。因此暂选**不预滤波**的结构纹理候选作为 C2c。预览没有像素轮廓真值，不能证明其边缘 F1 或着色质量优于旧 C2c。保边滤波候选仍留在 `make_adaptive_saliency_edge(..., variant="edge_preserving")` 供对照，但不是默认训练路径。

`scripts/audit_c2_adaptive.py` 在生成图时逐像素对比正式 `ConditionedGenerator` 的 C2c 输出，并确认第三通道为 0。v2 的逐步对比使用 `scripts/audit_c2_adaptive_revision.py`。当前 21 项单元测试通过；v2 的 1 张训练图与 1 张测试图冒烟运行也通过。尚未做正式 80 epoch 训练，也未评估生成 RGB。

## v2 相对 v1 的局部改动

对同一批 `trainB` 图像，`audit_c2_adaptive_revision.py` 输出旧版、只换连续密度、新版（再加细边底线）及增亮/变暗差异。旧版预览列与上次保存的 12 张图逐像素一致，最大像素差为 0。预览目录：

- `runs/edge_audit/c2c_revision_crop_seed42/quicklook.png`、`overview.png`
- `runs/edge_audit/c2c_revision_crop_seed7/quicklook.png`、`overview.png`
- `runs/edge_audit/c2c_revision_full_seed42/quicklook.png`、`overview.png`

两组各 12 张 256 裁剪图的平均绝对变化分别为 0.0080、0.0094；变化超过 0.05 的增亮像素平均占图像 0.67%、0.77%，变暗像素占 0.13%、0.23%。在**人为定义**的细边区域（`E_fine>0.4` 且 `E_coarse<0.1`）中，输出与细候选的平均强度比由 0.223→0.287、0.236→0.304。它只证明细边保留变强，不代表这些像素都是人车轮廓；树叶细边也可能同步变亮。差异图肉眼看变化较小，不能声称轮廓质量已经提高。

## 实验隔离

旧位置 C2c 权重在推理时必须指定 `--condition_mode soft_saliency_position`，否则同样的网络权重会收到不同的第二通道。v2 的正式实验名称应与 v1 隔离，例如 `flir_v2_c2c_adaptive_v2_40_40`；训练预算仍用原定的 40+40 epoch，并保持 FLIR 划分、256 裁剪、测试集与统一评估一致。v2 不能从旧位置版或 v1 checkpoint 续训。

位置权重参数 `--saliency_inner_weight`、`--saliency_outer_weight`、`--saliency_transition_fraction` 现在只作用于旧位置版；`--saliency_sigmas`、`--saliency_background_gain` 和 Canny 参数仍作用于新 C2c。
