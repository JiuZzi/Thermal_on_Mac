# D-v3：RGB外观、解码范围与结构约束修订

## 当前外观优化目标

保住真实红外里的目标、位置和轮廓。外观侧重点是颜色割裂、整体或局部偏暗、人物和车辆泛白，以及可见光材质细节不足。训练与评估按来源分组，规则适用于采用同一语义映射的其他数据集。颜色和明暗目标来自各自训练 RGB，测试图像只用于冻结方案后的评估。

### 合成红外与区域 Lab 分布

`config_quality_v3_appearance.json` 启用完整外观配置。沿用 D-v3 的模型和条件缓存格式，使用新的 run 从已有完整检查点初始化。

- **外观监督**每四步复用 RGB 循环已经构造的合成 TIR、E、P、Q。外观支路对这些条件断梯度，使用同一张真实 RGB 和它的教师掩码做对齐监督。原 RGB 循环继续训练反向生成器，RGB 噪声回放继续使用原自配对。合成红外仍存在域差距，收益要看真实红外验证图。
- **明暗和色度分布**用 `D.unpaired.appearance` 从训练 RGB 缓存拟合。每类先在图内采可靠像素，再按图和来源组均衡汇总 `L*` 与四个 `ab` 投影的软累积分布。稀少类别需要至少两个训练组才启用，生成图上可靠性不足的区域跳过该项。真实 TIR 的生成分布按有效更新做滑动估计，使不同图像可以提供不同颜色模式。该项取代两项 GMM 惩罚，权重在首 1,000 次更新逐步升到 0.1。原 GMM token 仍用于模型条件。
- **颜色过渡和细节**在原对齐外观损失内增加多尺度 `ab` 差分误差，使用目标 RGB 中的真实过渡作为参考。原灰度梯度细节项、E、内容和结构精度损失继续参与训练。颜色连续性和结构指标都要在留出数据上检查。

三项可分别消融。`rgb_appearance_input` 可设为 `rgb_gray`，`appearance_chroma_detail` 可设为 `0`，`appearance_distribution_weight` 可设为 `0`。关闭分布项时省略 `--appearance-reference`。要比较旧 GMM 监督，恢复原配置中的两个 GMM 权重并关闭新分布项。每个实验使用独立 run，固定起始权重、训练预算、样本和评估协议。

#### 拟合并短训

在项目根目录执行。`D_PYTHON` 使用原 D-v3 CUDA 环境。`TRAIN_CACHE`、`VAL_CACHE`、`DATA_MANIFEST`、`COLOR_PRIOR`、`FMB` 使用原实验的路径，`INIT_CKPT` 指向 Ubuntu 上区域版的完整检查点目录。完整检查点包含 `generator.safetensors`、`reverse.safetensors`、`critics.safetensors` 及检查点记录。

```bash
REFERENCE="$PWD/runs/D/unpaired/appearance/train_lab_reference.json"

"$D_PYTHON" -u -m D.unpaired.appearance \
  --data-cache "$TRAIN_CACHE" --output "$REFERENCE"

appearance_args=(
  --name D_v3_appearance_u1000_seed42
  --config D/unpaired/config_quality_v3_appearance.json
  --data-cache "$TRAIN_CACHE" --validation-cache "$VAL_CACHE"
  --data-manifest "$DATA_MANIFEST" --prior "$COLOR_PRIOR" --semantic-checkpoint "$FMB"
  --appearance-reference "$REFERENCE"
  --steps 1000 --save-every 1000 --log-every 20 --preview-every 0
  --recovery-every 100 --validation-every 1000 --validation-device cpu --best-policy structure
)

"$D_PYTHON" -u -m D.unpaired.train "${appearance_args[@]}" \
  --init-tir-checkpoint "$INIT_CKPT" --check-only &&
"$D_PYTHON" -u -m D.unpaired.train "${appearance_args[@]}" \
  --init-tir-checkpoint "$INIT_CKPT" --verify-updates 20 &&
"$D_PYTHON" -u -m D.unpaired.train "${appearance_args[@]}" --resume
```

拟合完成显示 `[READY]`。可用 `--frames-per-group` 调整每组采样上限。更换训练缓存或可靠性阈值时重新拟合。预检显示 `[APPEAR] input=synthetic_tir` 和有效分布类别数，训练日志里的 `lab_dist` 分别记录明度和色度差异。训练恢复校验参考文件摘要并还原滑动统计。导出推理权重沿用原流程，推理读取原 GMM 条件文件。

#### 验证颜色和结构

对已完成的验证推理目录运行以下命令。`PREDICTIONS` 指包含 `prediction.json`、`records.json` 和 `images/` 的目录。可加 `--rgb-cache` 指向同一验证划分的完整 RGB 条件缓存，同时报告和真实 RGB 的区域分布距离。

```bash
"$D_PYTHON" -u -m D.unpaired.appearance_metrics \
  --prediction-dir "$PREDICTIONS" --data-cache "$VAL_CACHE" \
  --data-manifest "$DATA_MANIFEST" --output "$PREDICTIONS/appearance_metrics.json"
```

比较天空等区域的偏暗比例、人车偏白比例、色度分布与同类内部的颜色跳变。结合 Edge-F1、边缘精度、人车 AP 和同图裁剪判断，灰图也可能有很小的颜色跳变。外部已配准数据可以用 `D.unpaired.paired_metrics` 做独立的区域 ΔE00 与感知质量评估。

## 原 D-v3 质量修订

2026-11-01。针对已完成D V1五轮训练中灰白人物、块状材质、后期RGB分布退化的问题，增加显式的质量版本。工程与CPU检查不代表新训练的FID/KID已经改善；用户要求真实GPU短验证稍后进行，当前没有启动新GPU训练。

## 版本与目标

保持 `[TIR, clarity_E, 0]`、seed42、原清晰/显著性轮廓、冻结FMB与RGB语义教师、原学得颜色先验、预训练SD1.5/ControlNet及原数据划分。不新增外部预训练模型，不删除源图适配器，不启用补边/JR或轮廓置信度C。

默认 `config.json` 仍是V1；`config_review_v2.json`仍是V2。必须显式传下面的新配置才启用本次修订：

- `D/unpaired/config_quality_v3.json`：4个实际DDIM步骤，保持旧采样预算与起点，适合首先验证质量目标。
- `D/unpaired/config_quality_v3_7step.json`：7个实际步骤；20个总时间步、strength0.35。与前者均从t=301开始，避免把增加步数与更大初始噪声混淆。7步需独立新run，不能冒充4步权重的原配方测试。

原模型主干、输入通道与可训练模块容量没有扩张。新增开销来自训练目标、偶发RGB自配对生成及梯度诊断；实际速度/显存要等GPU短验证测量。

## 修改内容

### 1. RGB外观监督经过真实生成路径

原 `rgb_replay` 从真实RGB的VAE latent预测噪声，仍保留作去噪先验锚定。新增 `rgb_appearance`：缓存中训练RGB自身的灰度图、E、P、Q经过完整 `FormalGenerator.forward()`，与同一张RGB比较。源图适配器、条件分支、LoRA、语义/颜色融合都能得到该任务的信号。

默认每4步执行一次，独立建图/反传，随后释放，再进行噪声回放，避免同时保留多张扩散图。真实TIR仍是无配对；没有用另一张随机RGB作为TIR的像素目标。

外观项由三级尺度RGB L1、Lab a/b L1、三级灰度梯度细节组成。权重为1、0.5、0.25，合成后乘 `loss_weights.rgb_appearance=2`，只在执行该任务的更新使用；没有隐含按间隔补偿权重。它学习曝光/材质/颜色，不保证解决RGB灰度到颜色的一对多歧义。

#### 可选的区域均衡外观监督

训练命令加入 `--rgb-appearance-region-balance` 启用 `class_balanced_lab_v1`。沿用训练 RGB 自配对及其 P/Q 缓存，给人、车、天空、植被等可靠类别区域分别计算与同图原始 RGB 对齐的 Lab 误差。每个区域先按 `P × Q` 求加权平均，再在有效类别之间取平均。

区域监督使用 `Q >= 0.5` 且有效权重总和至少为 16 的类别。有效图像把原外观损失中全分辨率 RGB L1 的 25% 换为区域 L* 误差，把全图色度项的 25% 换为区域 a/b 误差。低分辨率重建和细节项按全图计算。没有有效类别的图像使用原全图外观损失。

首轮沿用亮度对照实验的控制组参数、原 D-v3 第 11062 步完整检查点和已有缓存，选择新 run 名，只打开区域均衡选项。预检通过时 `[APPEAR]` 显示 `region balance=class_balanced_lab_v1`。自配对更新的摘要中 `photo_classes` 记录有效类别数，亮度和色度误差写入 `loss.jsonl`。验证 20 步后的同一 run 用相同策略和 `--resume` 继续。 [亮度对照实验命令](APPEARANCE_LUMA_AB.md)

### 2. 解码饱和保留修正梯度

V3默认采用straight-through限幅：前向数值与硬clamp完全相同，反传对raw解码输出使用恒等导数。额外的Huber范围损失直接作用于raw与其合法范围目标，避免只依赖被截断的图像损失。TIR生成、RGB循环、RGB自配对三处都记录并约束范围；所有原有非有限数值检查保留。

推理继续输出合法[-1,1] RGB。该梯度是显式的替代梯度，不是硬clamp的数学导数；收益与稳定性需要GPU训练确认。`decoder_policy=hard`与 `decoder_range=0` 可作对照。

### 3. 原颜色安全约束之外增加弱的连续混合分布吸引

原 `color_prior` 的99%支持范围hinge保留。新增 `color_attraction` 使用同一冻结GMM的完整混合似然，与混合分布在各分量中心的高密度参考比较，以log1p相对代价提供连续弱吸引。保留多峰和可靠区域，不用固定类别平均色，不强迫每张图片包含所有颜色模式。

当前原先验允许中性灰色在四类内取得零hinge；新项在偏离高密度模式时可产生梯度。若某类真实高密度模式本来就是灰色，仍允许灰色，不能把这个目标理解为强制增饱和度。人/车/建筑不在原GMM支持类别中；这些区域的RGB外观主要依靠新自配对、对抗与预训练生成器学习。

### 4. 结构精度与语义边界上下文

原E召回/方向约束与内容对齐保留。新增弱 `edge_precision`，在可信、缺少TIR结构证据的语义内部，处罚生成图里的多余粗边缘。使用5/9像素尺度、源E/源梯度和语义边界豁免带，允许小位移与细RGB纹理。它不是“E以外一律不许有边”，也不提供人工人车边界GT。

V3沿用V2的先屏蔽再编码，以及可靠性软色度平滑。颜色分支继续保守使用内部Q；语义结构分支可在测得E附近读取邻近可信上下文，填充没有可信观测的局部特征。低Q的P仍不进入编码器，全未知区域不会因偏置生成语义残差。这个确定性上下文门控不等于置信度C，第三通道始终为0。

### 5. 源残差预算与可解释诊断

源适配器残差默认每图RMS上限1，小残差原样保留；FP32计算缩放，避免半精度分母溢出。`source_residual_rms_limit=0`关闭该预算。该阈值是候选设置，不是已经找到的最优值。

新增源latent/残差RMS、解码饱和比例、外观分量与全部加权损失。每100步额外观察各目标在一组固定探针参数上的梯度；即使普通日志频率不是100的因子，也会在探针步记录。字段 `objective_probe_grad_*` 只覆盖source最后一层、条件输入卷积、首组语义/颜色头，不是整个网络的梯度，也不是输入重要性分数。探针不累积到优化梯度；CPU已验证启用/关闭后权重更新及噪声流逐值一致。

V3分别裁剪扩散适配器与逆向网络，避免两组共同共享一个裁剪预算。学习率、循环权重、预热与衰减预算保持原样；没有仅凭低D loss就自动削弱判别器或改变各损失权重。新诊断用于后续有证据的平衡实验。

## 兼容与缓存

新增输入签名绑定质量政策与采样设置，V1/V2及不同采样版本的权重不能当作V3原配方静默加载。新V3使用新的run，从相同RGB预热检查点初始化；不能在旧TIR run上直接加新配置 `--resume`。V3同一配方下的完整状态恢复已用CPU逐值验证。

本次没有更改P/E/Q的数值生成算法，因此当前本地已完成、相互匹配且校验通过的V1训练/完整验证缓存可复用。新生成的V3缓存额外记录条件管线代码指纹；训练与验证必须来自同一个身份，不能混拼旧缓存与重新生成的另一配方。服务器上若尚未准备缓存，使用新配置和新目录生成两份。

任何源码更新都会影响严格resume检查。仍在运行的旧实验应使用原源码完成；新V3放在独立工作目录或等旧任务结束再上传覆盖源码。

## Windows：只读预检

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location "$thermalRoot"

& $dPython -m D.unpaired.train `
  --name D_quality_v3_seed42 `
  --config D/unpaired/config_quality_v3.json `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42 `
  --steps 27655 `
  --save-every 5531 `
  --log-every 10 `
  --preview-every 5531 `
  --recovery-every 100 `
  --validation-cache runs/D/unpaired/conditions/val_full_seed42 `
  --validation-every 5531 `
  --validation-device cpu `
  --best-policy structure `
  --check-only
```

该命令不加载GPU模型、不创建训练run。本次两种配置均需通过缓存/先验/初始化/完整验证预检。

## GPU方便后：先20步工程短验证

将上面最后的 `--check-only` 替换为 `--verify-updates 20`。保留 `--steps 27655`，避免短验证改变正式学习率衰减预算。会创建新run、执行初始12图预览、训练20个成功更新并保存完整恢复快照；它只用于检验数值、显存和速度，不足以评价质量。

继续同一个run时删除 `--init-rgb-run`，删除 `--verify-updates`，加 `--resume`，其余预算与验证参数不变。若只追加另一个有限验证区间，保留 `--verify-updates N` 即可。

查看当前短验证权重的12图结果可运行：

```powershell
& $dPython -m D.unpaired.infer `
  --run-dir runs/D/unpaired/training/D_quality_v3_seed42 `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --phase val `
  --output-dir runs/D/unpaired/predictions/D_quality_v3_seed42_preview
```

这是开发预览，不是完整690张验证，也不用于test选模。确认数值和方向后才做长训练；正式best仍按完整验证的结构优先策略，同时保留FID/KID候选。

20步时还没有 `best.json` 属于预期，首次完整验证在5531步后才运行。

## Ubuntu

先按原在线下载说明安装环境、下载同一批模型；数据盘缓存变量不变。若新做缓存，两条依次执行：

下载仍使用 `python -m D.preview --download-models` 的默认下载入口；它只下载同一套固定版本基础权重，不在该命令后传V3训练配置。

```bash
python -u -m D.unpaired.prepare \
  --config D/unpaired/config_quality_v3.json \
  --data-manifest runs/D/ubuntu_assets_20261030/cycle_manifest.json \
  --tir-limit 0 --rgb-limit 0 --val-limit 12 \
  --device cuda:0 \
  --output-dir runs/D/unpaired/conditions/full_quality_v3_seed42

python -u -m D.unpaired.prepare \
  --config D/unpaired/config_quality_v3.json \
  --data-manifest runs/D/ubuntu_assets_20261030/cycle_manifest.json \
  --roles val --domains TIR --val-limit 0 \
  --device cuda:0 \
  --output-dir runs/D/unpaired/conditions/val_full_quality_v3_seed42
```

RGB预热仍用原1000步流程。若已有的训练/验证缓存身份完全相同且仅改变模型质量政策，可统一复用原缓存，先做check-only。

```bash
python -u -m D.unpaired.train \
  --name D_quality_v3_ubuntu_seed42 \
  --config D/unpaired/config_quality_v3.json \
  --data-cache runs/D/unpaired/conditions/full_quality_v3_seed42 \
  --data-manifest runs/D/ubuntu_assets_20261030/cycle_manifest.json \
  --prior runs/D/ubuntu_assets_20261030/rgb_prior_seed42_loss_v1.json \
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42_ubuntu_20261030 \
  --steps 27655 --save-every 5531 --log-every 10 \
  --preview-every 5531 --recovery-every 100 \
  --validation-cache runs/D/unpaired/conditions/val_full_quality_v3_seed42 \
  --validation-every 5531 --validation-device cpu --best-policy structure \
  --check-only
```

GPU短验证用 `--verify-updates 20` 替换最后一项。完成真实GPU短验证前不建议直接nohup启动完整长训练。

## 对照与验证边界

每个新增目标可把对应loss权重设0；解码、源残差、上下文门控和分组裁剪也有明确开关。不同前向政策绑定不同签名，改动后开新run。先比较同一RGB起点、同一seed42、同一划分下的V1与V3，再用单变量对照定位收益；不把修改过的V3当成旧V1续训。

CPU回归覆盖灰色旧hinge零梯度/新混合目标有效梯度、合法中性模式、RGB自配对样本归属、源适配器梯度、限幅/范围稳定性、未知语义隔离、C通道零梯度、采样起点一致、合成目标加权总值、诊断无副作用、失败事务以及完整恢复后的下一次更新逐值一致。全尺寸CUDA、峰值显存、真实训练FID/KID与人车完整性仍待GPU短验证和后续实验确认。
