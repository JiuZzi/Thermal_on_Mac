# SegFormer-B3：固定 FLIR 开发样本的语义预览

本步骤只检查语义是否可用。输入为原 trainB 红外图；不训练、不加载 GAN、不用框生成模型输入。
第三置信度通道和 C2b/C2c 训练入口没有改变。语义响应不作为已校准的边缘置信度。

## 1. 模型与权重

采用 InfraredIR 作者提供的完整 `best_model.pth`，网络为 MiT-B3 / SegFormer，解码器宽度 256，15 类输出。
权重放到：

```text
<thermal 项目>/runs/pretrained/segformer_fmb_b3/best_model.pth
```

作者目录：[完整分割权重](https://drive.google.com/drive/folders/1spGJsgOaTfi-EHqKRh_8a2jomlLhKiLa)。
`mit_b3.pth` 只有骨干，不能替代完整分割权重。实际文件会重新计算 SHA256、严格检查全部参数名称与尺寸。
源代码固定提交、15 类映射与依赖替换见 [来源说明](../scripts/semantic_preview/UPSTREAM.md)。

## 2. Windows PowerShell

在已有 PyTorch 环境中运行。建议 PyTorch >=2.6；不要在正在训练 C2 的环境里升级 torch。
新增加的依赖只有 OmegaConf；模型不依赖 timm、mmcv 或完整 InfraredIR 的扩散环境。

```powershell
$thermalRoot = "D:\thermal"
Set-Location "$thermalRoot"

python -m pip install `
  -r scripts\requirements_semantic_preview.txt

python scripts\preview_semantics.py `
  --checkpoint "$thermalRoot\runs\pretrained\segformer_fmb_b3\best_model.pth" `
  --device cuda `
  --num-images 40 `
  --detail-short-side 1024 `
  --output-dir "$thermalRoot\runs\edge_audit\segformer_fmb_b3_semantic_preview_seed42"
```

第一行应改为本机真实项目路径。没有 CUDA 时用 `--device cpu`；Mac 终端 MPS 可用时用 `--device mps`。
默认 `auto` 优先 CUDA、再 MPS、最后 CPU；显式指定的设备不可用会报错，不静默切换。
输出目录已存在会拒绝覆盖；重做时换目录名。

只验证入口的一张图：`--num-images 1`，使用新的输出目录。这是实现检查，不是完整效果证据。

## 3. 固定样本与计算方式

使用随代码保存的 `scripts/semantic_preview/fixed_review_seed42.json`：40 张 trainB（30 常规、10 小目标挑战）与
90 个原重点裁剪。它来自旧重要轮廓审查包，保留图片 SHA256 和原坐标；并检查图片属于当前 trainB 协议。
源图片、视频或几何不匹配会停止，不能暗中替换为不同帧。若明确使用另一个完整审查包，传 `--review-manifest`。

所有预测在完整图上进行，然后裁剪同一浮点结果并最近邻放大。不将人物框裁成单独模型输入，避免改变上下文。
框仅定位审查和计算框内响应代理；框不是语义/边界真值，不用于训练。testB 不用于选择尺寸或阈值。

- **base：**与作者一致，原 uint8 灰度复制三通道，按原均值/标准差归一化，短边不足 512 时等比例 PIL 双线性放大。
- **detail：**同一冻结权重，短边至少 1024；只检查尺度是否影响小目标响应，不增加真实图像信息，也不保证更好。
- **可选翻转：**`--flip-tta` 仅对 detail 做翻转、逆对齐、类别响应平均；base 保持作者单次前向。默认关闭。
- **无较大尺寸对照：**`--detail-short-side 0`。其余因素不变。

两版 logits 都先按作者 `align_corners=True` 恢复原坐标，再 softmax；输出不逐图归一化，不混合两版或逐图选最好。

## 4. 输出与怎么看

打开输出目录的 `index.html`：

- 原红外、两种尺寸的类别叠加；展开查看完整类别图。
- 行人、车辆组、建筑、植被软响应，显示统一 [0,1]。
- 固定重点目标的并排放大图，可点击打开原尺寸。
- 可选人工评价“识别且基本覆盖 / 识别但明显遗漏 / 未识别 / 误分类 / 无法判断”，并导出 `semantic_review_ratings.json`。
  页面自动保存本机评分，可导入同一权重、样本、代码和预处理对应的评分。不同审查包会拒绝混用。

先看原图是否可辨认，再检查远处行人响应、误分类和区域膨胀。人体区域填满是语义分割的正常表现；此处不评轮廓宽度。
彩色标签只是类别显示色，不是 RGB 着色。FMB 没有单独 rider/other_vehicle 类；这些裁剪需要参照相关类别人工判断。
植被类别也不会直接指出每片叶纹应删除。

同时保存：

- `maps/*_responses.npz`：原坐标的 float32 全部 15 类响应、类别 ID、熵和最大响应；读取时不需要 pickle。
- `frames/`、`targets/`：独立 PNG，便于放大查看或导出。
- `preview_manifest.json`：模型/数据/代码哈希、几何、参数、设备、时间与限制。
- `frames.csv`、`targets.csv`：响应统计代理，不能称准确率、边界召回或 mIoU。
- `status.json`：是否完整完成；中断后保留已有输出并明确标记。

没有 FLIR 像素语义真值，不计算 FLIR mIoU。冻结外部先验在这批开发图上可用也不证明泛化或最终 RGB 提升。
截至 2026-10-13，用户已完成部分审查并授权先用作者 512 版调节重要性 R。独立轮廓预览见第 7 节。

## 5. 实现检查

```powershell
Set-Location "$thermalRoot"

python -m unittest discover `
  -s scripts\tests `
  -p test_semantic_preview.py `
  -v
```

测试关注作者预处理、恢复坐标顺序、冻结条件、SyncBN 的 eval 兼容性、完整权重检查、源图哈希与裁剪范围。
真实权重与 FLIR 推理另行运行；合成测试只能验证实现。

## 6. 本次已验证结果

2026-10-12，本地 Mac 使用 CPU、4 线程完成完整预览：40 张图 / 90 个裁剪，作者 512 与额外 1024 分开导出。
整个导出约 168 秒；这是本机这次运行记录，不是其他设备的时间承诺。
完整权重 SHA256 为 `a3f9de85452534f7731f01ef64cebee4d62c9aa4ea133861c7bb3124212aa1fb`，589 个状态张量严格匹配。
14 项单元检查通过；全部 40 个 NPZ 的 15 类 float32 响应有限、归一和为 1、类别图与 argmax 一致；770 个页面图像引用均存在。
离线评分的修改、保存、导出、导入与错误审查包拒绝已检查。CPU 真权重推理已验证；CUDA/MPS 实机尚未验证。

实际图中有小行人语义响应，但边界仍粗糙，较大尺寸不是逐目标必然更好。
这批输出供独立审查；尚未得到语义精度、小目标提升、轮廓改善或着色增益的结论。

## 7. 512 语义引导软轮廓对照（2026-10-13）

入口：`scripts/preview_semantic_contours.py`。使用同一本机生成的上述缓存中的 `base_probabilities`，
不读取 1024 响应进行融合，不需要再次训练语义模型。缓存、源图、原推理代码和划分哈希不匹配时停止。

固定采用行人、车辆组和建筑的软响应之和：

$$
S=P_{person}+P_{car}+P_{truck}+P_{bus}+P_{motorcycle}+P_{bicycle}+P_{building},\quad
R_{sem}=0.18+0.67S.
$$

这些类别等权，没有逐类调增益、硬阈值、形态学膨胀或位置 ROI。
R 仍在 C2c 原来的 [0.18,0.85] 范围，重要区域倾向细候选，背景倾向粗候选；每个位置保留粗候选通路。
沿用 C2c 的多阈值、多尺度、1像素软化、背景系数0.5、纹理惩罚 D 与25%细边保底：

$$
E_{sem}=\max\left[0.25E_{fine},\;
\left(R_{sem}E_{fine}+(1-R_{sem})0.5E_{coarse}\right)(1-0.4D)\right].
$$

唯一待验证因素是 R。语义类别错误仍会影响权重；在人被错判成车等同属重要类别时，重要性仍可较高，
但这不代表类别正确，也不支持后续直接使用该类别给颜色。人被错判成植被时仍可能减弱，保底只减少完全删边的风险。
算法不补回所有 Canny 候选都遗漏的边，不修复断裂、内部软化响应过宽或错误边缘位置。

已有完整缓存时，Windows PowerShell 多行命令为：

```powershell
Set-Location "$thermalRoot"

python scripts\preview_semantic_contours.py `
  --semantic-preview-dir "$thermalRoot\runs\edge_audit\segformer_fmb_b3_semantic_preview_seed42" `
  --num-images 40 `
  --output-dir "$thermalRoot\runs\edge_audit\semantic512_contours_seed42_v1"
```

本机尚无缓存时，先按第 2 节在本机生成缓存；只需要512时传 `--detail-short-side 0`。
重新生成须使用新输出目录，并将对应目录传给轮廓命令。

打开新目录的 `index.html`。全图展示红外、R、C2b、C2c、新候选；展开目标后查看相同裁剪与最近邻放大。
裁剪面板依次为红外、C2b、C2c、新候选、语义 R；上行灰度，下行红色叠加。
黄色框只在红外图中定位评分目标，未进入轮廓生成。所有显示统一使用[0,1]，不逐图增强对比度。
`quicklook.png` 按预先固定的01、02、31、34图排列，没有依据输出质量挑样本。

输出包括：`maps/*_contours.npz` 原坐标 float32 轮廓/候选/R/D；`frames/`、`targets/` 图片；
`frames.csv`、`targets.csv` 强度代理统计；`preview_manifest.json` 数据/缓存/代码哈希；`status.json` 完成状态。
这些框内或语义区域强度代理不能当边界召回、准确率或结构改善。

本地CPU复用缓存完成40张/90裁剪，轮廓导出循环约15秒，不含之前的语义推理。
22项检查通过（原语义14项加轮廓8项）；每张图的C2b/C2c都与原 `ConditionedGenerator` 逐像素匹配，
新轮廓通过原有 `edge_override` 的直接通道接口检查为 `[TIR,E,0]`。
正式GAN并未注册新的训练模式；后续在线裁剪/翻转、循环中合成红外的语义输入一致性，以及生成器是否有效利用它，需另行设计和验证。

初看部分人车和建筑轮廓更突出、部分植被细纹理减弱；小目标断裂、过宽和语义外溢仍然存在。
下一步用这些固定对照确认重要边界有无遗漏/误增强，再决定是否进入生成模型的消融训练。

## 8. 结构显著性与语义共同判断（2026-10-13）

用户授权在第7节的基础上提升融合判断。新增独立入口 `scripts/preview_structure_semantic_contours.py`，
结果保存到 `runs/edge_audit/semantic512_structure_joint_seed42_v2`，原语义v1图和正式C2c继续保留。

原结构显著性 `R_0` 来自C2c局部结构张量、对比度和连续边缘密度；不是新训练的显著性网络。
保持冻结512语义、候选、纹理惩罚D、融合系数和细边保底。只增加重要性的结构补充：

$$
A=\operatorname{clip}\left(\frac{E_{fine}-0.5E_{coarse}}{E_{fine}+10^{-6}},0,1\right),\quad
W=(1-S)(1-D)A,
$$

$$
R_{joint}=R_{sem}+W\max(R_0-R_{sem},0).
$$

语义支持较强时保留原效果；语义支持弱、原结构权重更高且细边占优时补充。
密集纹理区域的补充较少，粗边独有的位置不提高R，避免提高R反而损失粗轮廓。
联合R位于语义R与两者较大值之间，仍受[0.18,0.85]约束。W只是结构保护权重，不是边缘置信度。
语义分类结果本身不变，模型仍可能把小行人判断为植被；此步骤只缓解对已有候选的误抑制。

在同一本机已有语义缓存和v1轮廓结果时运行：

```powershell
Set-Location "$thermalRoot"

python scripts\preview_structure_semantic_contours.py `
  --semantic-preview-dir "$thermalRoot\runs\edge_audit\segformer_fmb_b3_semantic_preview_seed42" `
  --previous-contours-dir "$thermalRoot\runs\edge_audit\semantic512_contours_seed42_v1" `
  --num-images 40 `
  --output-dir "$thermalRoot\runs\edge_audit\semantic512_structure_joint_seed42_v2"
```

未生成缓存时依次运行第2节和第7节。已有输出会拒绝覆盖，重做应换目录名。
新入口要求原v1缓存与语义源一致；每张图都重算并逐像素核对原v1全部浮点图，以及原C2b/C2c代码。

打开新目录的 `index.html`。目标面板依次为红外、C2c、语义v1、结构＋语义、联合R。
上行灰度轮廓，下行红色叠加；黄框只定位审查对象。
另外，每帧保存 `semantic_global_match`：只用完整图均值求全局缩放系数，使v1轮廓达到联合版的相同全图平均强度，
不使用目标框或语义区域做缩放。它仅用于排查整体变亮，不作为新的轮廓算法或训练输入。

本地CPU完成40图/90裁剪，导出循环约18秒，复用原语义缓存。29项检查通过；530个页面图像引用均存在；
所有v1浮点图逐像素相同，第三输入通道保持零；新图未压弱v1像素，25%细边底线与R范围通过核对。

| 已指出的行人 | 语义v1框内强度 | 联合保护 | 相同全图均值缩放对照 |
| --- | ---: | ---: | ---: |
| 02_O03 | 0.3507 | 0.3873 | 0.3642 |
| 08_O09 | 0.3526 | 0.3767 | 0.3602 |
| 29_O02 | 0.3524 | 0.3842 | 0.3674 |

三个样例均部分恢复，也高于相同全图均值的缩放对照；但没有回到原C2c的框内强度。
预测植被区域的平均轮廓强度由v1的0.1643回升到0.1702，仍低于C2c的0.1794；36/40张图在该代理统计上低于C2c。
这些统计仅为强度变化，不是边界准确率，三个开发样例也不能证明泛化。
局部改善幅度不大，树枝/路面等杂纹可能同步恢复；不修复全部漏边、断裂或轮廓过宽。
完整汇总位于输出目录 `RESULTS.md`；当前保留为独立候选，着色训练模式、在线几何和RGB效果仍需后续验证。
