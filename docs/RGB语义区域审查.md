# RGB语义区域第一批审查（2026-10-20）

## 结论和当前可用范围

已复用现有冻结 FMB SegFormer-B3，在32张trainA RGB、32个不同视频上比较原始RGB输入与可见光灰度复制输入。原始RGB在这批开发图的预先记录检查点上略好，但现有模型不能直接当作全量RGB的可靠像素标签。

已从92块候选内部区域中，经Codex视觉复核保留67块稀疏种子：道路28、植被12、天空26、路侧／人行道1；另有16块剔除、9块不确定。种子来自30张图／30个视频，共39531个标记像素。它们可用于小规模管线验证与进一步人工审阅，不能代表整类的完整颜色分布。当前没有拟合GMM或修改GAN训练。

- [32图语义对照页面](../runs/semantic_audit/rgb_fmb_seed42_v1/index.html)
- [92块候选的复核状态、原图与局部](../runs/semantic_audit/rgb_fmb_seed42_v1/seed_regions/index.html)
- [三图概览](../runs/semantic_audit/rgb_fmb_seed42_v1/overview.jpg)
- [种子统计](../runs/semantic_audit/rgb_fmb_seed42_v1/seed_regions/summary.json)

## 数据与输入方式

只使用trainA，不使用testA/testB选择设置。seed42，在每视频随机选一张代表帧后按场景轮转，最终覆盖城市街道8、高速8、住宅7、其他6、停车场3；没有覆盖原数据所有罕见场景。源文件均为当前协议的360×288 RGB，记录了输入哈希和协议哈希。

权重为现有完整 `best_model.pth`，SHA256 `a3f9de85452534f7731f01ef64cebee4d62c9aa4ea133861c7bb3124212aa1fb`。不下载额外模型、不微调语义模型。两种输入都采用短边至少512、同一归一化、logits先恢复原坐标后softmax，无翻转平均：

1. **RGB探针：**保留真实RGB三个通道。这是跨输入域的实验探针，不称为作者原TIR推理流程。
2. **灰度探针：**Pillow将可见光RGB转亮度，再调用原有灰度复制预处理。可见光灰度仍不等于热红外。

生产用 `scripts/semantic_preview/inference.py`、轮廓算法、生成器和服务器正在执行的两组实验均未修改。新增脚本是独立审查工具。

## 原始标注不能直接作为所需掩膜

读取 `FLIR_ADAS_v2/images_rgb_train/coco.json` 的169174条实例记录：所有 `segmentation` 都是8个坐标，恰好对应检测框四角。这不是目标真实轮廓，也不是天空、植被、道路、人行道的全图语义标签。

另外，06和27两张的协议 `hours=day`，但RGB视觉上呈现夜间人工照明。审查对照保留这两个案例，种子候选阶段排除它们；没有修改原协议或训练数据。日间颜色先验学习前还需更系统地检查光照标签。

详见[数据来源审查](../runs/semantic_audit/rgb_fmb_seed42_v1/data_source_audit.json)。

## 点审查方式与结果

预测前，只看原RGB，在每图固定3×3位置记录五组类别：道路、路侧／人行道、植被、天空、其他。共288点，其中31个因边界、材质或类别不清楚标为不确定；257点参与对照。标签由Codex视觉判断，不是用户／专家标注，不是数据集像素真值。

注释文件在第一次预测前固定哈希，之后没有修改。固定位置及容易判断的点有采样偏差，且道路点多，整体吻合率不能代替全图像素准确率、IoU或独立测试表现。

| 输入 | 五组点吻合率 | 候选核心区中的点吻合数／保留点数 |
|---|---:|---:|
| 原始RGB | 232/257 = 90.3% | 170/179 = 95.0% |
| 可见光灰度复制 | 226/257 = 87.9% | 170/180 = 94.4% |

候选核心区规则在预测前固定为：该类最大响应≥0.9、第一与第二类别响应差≥0.2、区域向内腐蚀2像素。较高的核心区吻合率同时伴随样本拒绝，不能只报较高数字而不报保留量，也不能把阈值当作校准可靠性。

RGB探针中，各类保留下来的检查点分别为：道路106/111、植被33/37、天空31/31、路侧／人行道0点。天空31/31只是小样本结果；人行道没有足够保留检查点，不能推算其可靠性。

预测后重新看原分辨率图时，24_P1和30_P2出现建筑／树冠重叠等标注争议。原始预测前标签和分数保留，争议单独记入[标注注意事项](../runs/semantic_audit/rgb_fmb_seed42_v1/annotation_cautions.json)。这些点不能单独用来证明模型误分，也使上述小样本数值更不适合当正式精度。

完整混淆与逐点记录见[点审查JSON](../runs/semantic_audit/rgb_fmb_seed42_v1/point_audit.json)。

## 为什么还要逐块复核

- 建筑、车辆阴影等可能进入植被候选，响应仍可很高，例如08白色建筑区域、16车辆／建筑暗部候选。
- 棕榈叶、稀疏树冠的预测会吞入天空孔隙；即使取高分核心区，也可能得到混有蓝／白背景的“植被色样”。
- 路面、道路标线、路侧土石、人行铺装之间仍有混淆与标签口径问题；不能自动分开统计两类颜色。
- 内部区域筛选天然偏向常见、容易、低歧义的外观，会遗漏细叶、枯草及复杂阴影。不能据此声称已经学到了完整颜色多样性。

复核由Codex查看原RGB上下文与原像素放大后做出，属于预测后数据整理，不用来提高前面的“预测前点吻合率”。保留的种子标签只标记明确小矩形内部，其他像素为255（忽略）；它们不是全图伪标签。若后续用这些种子学习先验，应注明视觉区域复核带来的辅助语义监督，而不是声称完全无人工筛选。

## 产物与复现

```text
scripts/audit_rgb_semantics.py                固定抽样、两种输入推理和点审查
scripts/review_rgb_seed_regions.py            内部区域候选、视觉复核后导出稀疏标签
scripts/tests/test_rgb_semantic_audit.py       7项实现契约检查
runs/semantic_audit/rgb_fmb_seed42_v1/
  input_manifest.json                        抽样、输入与协议身份
  manual_points.json / annotation_lock.json   原图检查点及预测前哈希
  predictions_manifest.json                  权重、输入方式、代码和环境
  maps/                                     两种输入的float32响应、标签、候选核心区
  seed_regions/review.json                    每块的接受／拒绝／不确定及理由
  seed_regions/reviewed_masks/                67块种子合成的稀疏类标签，未标像素255
  point_audit.json                           点级开发诊断
  index.html                                全部32图对照
```

在新的输出目录复现；不要覆盖已有注释与预测：

```bash
python scripts/audit_rgb_semantics.py prepare --output-dir runs/semantic_audit/rgb_fmb_seed42_new
# 只看新目录raw中的原图，完成manual_points.json，无法判断填uncertain。
python scripts/audit_rgb_semantics.py infer --output-dir runs/semantic_audit/rgb_fmb_seed42_new
python scripts/audit_rgb_semantics.py evaluate --output-dir runs/semantic_audit/rgb_fmb_seed42_new
# 本批次06/27的RGB光照异常已记录在预测前注释中，显式排除其色样。
# 新抽样需重新查看原RGB，不能照搬帧号；无排除时省略--exclude-frames。
python scripts/review_rgb_seed_regions.py prepare --audit-dir runs/semantic_audit/rgb_fmb_seed42_new --exclude-frames 06 27
# 看全部候选，将seed_regions/review.json标为accepted/rejected/uncertain。
python scripts/review_rgb_seed_regions.py export --audit-dir runs/semantic_audit/rgb_fmb_seed42_new
```

本地CPU推理完成，7项契约检查通过，灰度探针与原接口逐像素一致的检查通过。另核对全部64组响应、输入及权重哈希、预测前注释锁、67块种子与30张稀疏标签的对应关系，以及287个本地链接，见[产物核对记录](../runs/semantic_audit/rgb_fmb_seed42_v1/artifact_verification.json)。实现和文件核对不代表全量RGB语义质量。

## 接下来可以做什么

第一批来源审查和稀疏种子准备已完成。天空、道路、植被种子可用于一个明确限定为小规模的颜色分布拟合试验；路侧／人行道只有1块，先补区域或暂缓该类。颜色先验的按视频验证划分仍需在拟合前固定。

扩展到全6191张自动提取之前，需要适配RGB的分割器或更多人工修正，并独立核查类别纯度、光照覆盖与未见视频表现。当前结果不足以证明现有FMB模型可直接承担全量RGB伪标注；也没有任何新颜色先验或着色增益结论。
