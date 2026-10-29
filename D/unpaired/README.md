# D：真实 TIR 无配对扩散训练

正式入口是 `D.unpaired.prepare/train/infer/evaluate`。`D.train` 是之前的 RGB 自配对预热，`D.preview` 是零训练 D0/D1。新版本保留现有清晰/显著性轮廓、冻结语义模型和颜色统计，实际训练 TIR→RGB。用户已授权新增损失；完整训练后的提升仍需验证。

2026-10-28本地审查：无配对循环与教师梯度路径通过小模型检查；新增可靠性融合候选与通用校验，完整说明见[融合、模块输入与无配对训练审查](../../docs/D系列融合与无配对训练审查_2026-10-28.md)。Windows正在运行的原V1继续作为基准，不在运行中替换源码。

## 本地审查候选V2（显式启用）

默认`config.json`与`DEFAULT_CONFIG`仍为V1。新增`config_review_v2.json`，保持E、网络容量、采样、学习率和九项损失权重，改变两项策略：

| 字段 | V2值 | 作用 |
|---|---|---|
| `fusion_policy` | `quality_masked` | P／颜色特征先由Q与支持度筛选、归一化聚合，再逐层屏蔽无支持区域；输出仍按可靠性控制强度 |
| `fusion_min_quality` | `0.1` | 过滤低质量语义上下文，需检查是否损伤弱小目标 |
| `chroma_smooth_policy` | `quality_soft` | 用软语义相似度与相邻Q调节色度平滑 |

prepare和train均需显式传`--config D/unpaired/config_review_v2.json`，使用新缓存目录、新run及相同RGB预热起点。新缓存增加P/E/Q完整计算代码和库版本指纹。V1正式权重不能直接作为V2续训，原run也不能用修改后的训练源码`--resume`；严格身份检查会拒绝代码变化。现有Windows任务尚未同步这些本地修改。

V2支持分别将两项policy设为`legacy`，用于只改融合、只改平滑和组合三组对照；不是已验证最优配置。C仍为零，显式补边与`recall_v1`均未接入。

通用修复还包括P/Q数值检查、解码clamp前非有限值检查、逐图合成条件与内容采样、重复输出名／缓存进度校验，以及关闭颜色损失时不要求无用的loss_support。日志新增实际加权损失、分支残差幅度、分开统计的裁剪前梯度范数和解码饱和比例。当前加载器仍为batch1。

## 日志显示（2026-10-28整理）

正式训练不再直接打印完整字典。每个`--log-every`周期输出三行，首次成功更新和最后更新也会输出。以下数字仅用于说明格式：

```text
[TRAIN  ] 01010/10000 (10.1%) | 7.20s/update | elapsed 1m12s | ETA ~17h58m | VRAM peak/resv/free=5.53/6.1/1.3GiB
[LOSS   ] G=15.82 | D_RGB=0.5164 D_TIR=1.121 | cycle_TIR=0.4616 cycle_RGB=0.542
[AUX    ] edge=0.5744 content=3.056 semantic=2.708 color=0.2154 smooth=0.1473 color_w=1e-05 replay=1.147
```

- `G`是当次实际加权总目标；cycle和AUX显示该次更新的原始损失，不是区间均值。`color_w`是渐增后的实际颜色系数；`replay=off`表示当次没有安排RGB回放，不把缺失误记成损失为零。
- 速度与ETA按**本次启动成功执行的更新数**计算，断点续训不会用历史1000步去除刚运行的72秒。模型加载和初始基准预览不计入；计时开始后的读数据、跳过更新、保存与预览开销计入后续估算。ETA为估计值。
- `VRAM peak/resv/free`依次为本进程张量峰值、当前分配器保留量、当前整卡剩余量（GiB）。`loss.jsonl`另存当前张量量、保留峰值和总容量；保留量包含张量量，不可相加，历史峰值与当前free也不属于同一时刻。默认batch1已有文档记录约5.53GiB张量峰值，未在本机复测；[估算范围与后续batch验证](../../docs/D显存估算与批量调整_2026-10-28.md)。
- 原完整损失及分支／梯度／覆盖率诊断继续保存在run目录的`loss.jsonl`；新增总步数、时间、ETA、本次起始步与UTC记录时间。终端摘要及训练内部事件另存UTF-8的`train.log`，续训时追加。若仍要在终端看全部数值，加`--log-details`；该显示开关不修改训练配置。
- 缓存／预览按约10%进度或样本完成时检查的30秒间隔打印，显示完成数、耗时和ETA；权重保存显示开始、完成及耗时。异常保留原traceback，并在`train.log`记录；`train.log`不收集第三方库直接写到stderr的全部提示，完整进程输出仍可用原命令重定向保存。

这是本地源码更新，当前Windows进程不会自动切换显示；无需为日志重启它。原严格resume检查包含源码，继续原任务时仍使用原代码，后续新run再采用更新版本。日志格式与模型／优化目标无关，V1和V2均可使用。

## 实际信息流

- 主条件 **X=[TIR, clarity_v1 E, 0]** 在同一坐标中。E 内现有显著性、纹理抑制原样保留，不再重复乘显著性，也没有启用 JR。
- X 进入 ControlNet 条件编码器及可训练的源内容残差编码器。**TIR 复制三通道**经冻结 RGB VAE 作为扩散起点，不把 `[TIR,E,0]` 当彩色照片送进 VAE。
- 固定 SD1.5 与 Canny ControlNet commit。可用已有 RGB 软 E 预热初始化：旧 `[E,E,E]` 首层三个卷积核求和放到 E 通道，TIR、C 核先置零，保留 E 响应，随后学习 TIR。
- 冻结 FMB 提供15类源 P；现有语义可靠性 Q 对独立语义适配器门控。现有多模态 GMM token 经颜色适配器注入各尺度残差，统计只支持 road、sidewalk、vegetation、sky，不代表每个物体真实颜色。
- **Q 不是轮廓置信度 C**。C=0 表示模块未实现，不计算 `E*C`。零输入通道未学会使用真实置信度，以后需要加入训练。
- 训练 ControlNet 条件编码/输出投影、UNet 交叉注意力 rank4 LoRA、源内容残差、语义/颜色投影。SD/ControlNet 大部分参数、VAE、文本编码器、教师和 GMM 冻结。扩散可训练约1341万参数，反向 ResNet9 和两域 PatchGAN 另行训练。

## 无配对目标

独立打乱 TIR_i 和 RGB_j。**不与随机 RGB_j 计算 TIR 生成图的像素 L1**。

| 目标 | 默认权重 | 作用 |
|---|---:|---|
| 两域 LSGAN | 1 | 域真实性 |
| TIR→RGB→TIR 循环 L1 | 10 | 保留 TIR 内容 |
| RGB→TIR→RGB 循环 L1 | 10 | 反向约束与颜色表达 |
| 软梯度支撑/方向 | 1 | 对齐源 E，允许热/可见光对比反转 |
| 同位置局部自相似对比 | 0.2 | 减少区域替换；不是 CUT/JR 复现 |
| 可信区域跨域语义一致性 | 0.1 | 冻结 RGB 教师分析生成图，与源 P 比较，保留图像梯度 |
| GMM Lab 色度超额 NLL | 0.01 | 弱颜色约束，前1000更新逐渐增加 |
| 同类低边缘区域色度平滑 | 0.02 | 减少割裂，保留边界，不强制亮度一致 |
| RGB 自配对去噪回放 | 1，每4更新一次 | 减少遗忘，条件 `[Gray(RGB),E_RGB,0]` |

两条循环及回放分图反传。合成 TIR 的 E/P/Q detach，**合成 TIR 本身保留 VAE/源路径的梯度**。生成 RGB 上的损失均可微，不用 NumPy Canny 做训练损失。

`config.json` 默认：原图360×288、batch1、10个DDIM总时间步、strength0.4，即实际 **4步去噪**；CFG1。训练与推理同一完整采样链，无截断反传。扩散lr1e-5，反向网络/判别器lr2e-4，前50更新预热、后半线性衰减。

沿用既有 `cycle_manifest.json` 视频组划分：完整train TIR1420/RGB5531，val TIR690，test TIR112/RGB696。此次使用**整图，无原GAN的随机256裁剪/翻转**，并有新损失、采样与预算，属于扩展D。不能声称与旧GAN只有骨干不同；只换骨干的另一个候选消融见 `D/BACKBONE_BASELINE.md`。

## Windows：准备条件、先做短程真实训练

当前本机已有独立验证 venv。新电脑按 `D/README.md` 建 `thermal_d` 后，用 `python` 代替 `& $dPython`。

```powershell
$thermalRoot = "E:\上海大学\项目\thermal\Thermal_on_Mac"
$dPython = "$thermalRoot\runs\D\verify_env\Scripts\python.exe"
Set-Location "$thermalRoot"

& $dPython -m D.unpaired.prepare `
  --tir-limit 256 `
  --rgb-limit 256 `
  --val-limit 12 `
  --output-dir runs/D/unpaired/conditions/clarity_seed42 `
  --allow-download
```

默认 CPU 冻结教师推理；GPU空闲可加 `--device cuda:0`。**训练与测试准备应使用相同后台**，后台写入 recipe。`--allow-download` 仅下载已有固定 RGB 教师。如果未缓存基础扩散模型：

```powershell
& $dPython -m D.preview --download-models
```

当前固定选256张TIR、256张RGB训练图及每域12张验证图，不使用test训练。缓存保存原图/P/E/Q及数据和教师SHA。已有完整缓存可以跳过prepare；准备中断用同样参数加 `--resume`，会校验并跳过已完成图。

你已有RGB预热完成1000更新，用它初始化：

```powershell
& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_clarity_pilot_seed42 `
  --data-cache runs/D/unpaired/conditions/clarity_seed42 `
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42 `
  --steps 200 `
  --save-every 50 `
  --log-every 10 `
  --preview-every 50

Start-Process "$thermalRoot\runs\D\unpaired\training\D_tir_unpaired_clarity_pilot_seed42\previews\step_000200\index.html"
```

200是**优化更新数**，不是epoch或5+5。每更新各读取一个独立TIR/RGB。先检查真实TIR是否改善，再扩大预算；200不保证收敛。没有预热文件时省略初始化参数，会记录为预训练Canny初始化。网页按行显示 TIR、E、源语义、适配前、适配后；前后使用相同初始权重/采样/每图seed42，不与早期strength0.6预览混比。

## 中断恢复与完整训练

同一run的恢复保持原配置、缓存、代码、版本、总更新数；去掉初始化参数，加 `--resume`：

```powershell
& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_clarity_pilot_seed42 `
  --data-cache runs/D/unpaired/conditions/clarity_seed42 `
  --steps 200 `
  --save-every 50 `
  --log-every 10 `
  --preview-every 50 `
  --resume
```

检查点包括三个网络、优化器、AMP、两域采样顺序、历史图池、噪声/位置采样和 Python/NumPy/Torch RNG。只从最后一个**完整保存的更新**恢复，中断时未保存的部分重跑。事务完成后才更新latest。

固定seed42，关闭TF32/cudnn benchmark，尽量用确定性算法。部分CUDA backward没有严格确定实现，因此不保证跨GPU、Windows/Ubuntu逐位一致。

扩大训练集用新缓存、新run：

```powershell
& $dPython -m D.unpaired.prepare `
  --tir-limit 0 `
  --rgb-limit 0 `
  --val-limit 12 `
  --output-dir runs/D/unpaired/conditions/full_seed42

& $dPython -m D.unpaired.train `
  --name D_tir_unpaired_clarity_full_seed42 `
  --data-cache runs/D/unpaired/conditions/full_seed42 `
  --init-tir-checkpoint runs/D/unpaired/training/D_tir_unpaired_clarity_pilot_seed42/checkpoints/step_000200 `
  --steps 10000 `
  --save-every 500 `
  --log-every 10 `
  --preview-every 500
```

10000只是可调整预算示例，不是已验证最佳长度。完整P缓存可能几十GB。`--init-tir-checkpoint` 继承三个网络，**重置优化器、采样、图池、RNG和学习率**，属于新实验热启动，与断点恢复不同。从RGB预热公平开始则改用 `--init-rgb-run`，不能同时给两个初始化。

## 完整112张测试及旧评估器

根据val选择检查点，再报告完整test，不用test调参。以下以pilot step200示例，正式训练替换run/step。

```powershell
& $dPython -m D.unpaired.prepare `
  --roles test `
  --domains TIR `
  --test-limit 0 `
  --output-dir runs/D/unpaired/conditions/test_seed42

& $dPython -m D.unpaired.infer `
  --run-dir runs/D/unpaired/training/D_tir_unpaired_clarity_pilot_seed42 `
  --data-cache runs/D/unpaired/conditions/test_seed42 `
  --phase test `
  --step 200

& $dPython -m D.unpaired.evaluate `
  --prediction-dir runs/D/unpaired/training/D_tir_unpaired_clarity_pilot_seed42/predictions/test_step_000200/adapted `
  --data-cache runs/D/unpaired/conditions/test_seed42 `
  --output-dir runs/D/unpaired/evaluations/D_tir_clarity_step200_seed42
```

测试准备使用训练相同config、FMB、RGB教师、E配方及后台。推理加 `--compare-initial` 可生成适配前对照，中断生成则同命令加 `--resume`。程序拒绝用12张验证预览算完整分布指标。

沿用 `scripts/evaluate_distribution.py`：FID、KID(mean/std，subset50×100，seed42)、APCE-Py、edge precision/recall/F1，Canny sigma1.0；输出JSON、逐阈值CSV、PNG、PDF。不做无配对图像PSNR/SSIM。APCE-Py是项目原有scikit-image实现，非PearlGAN官方MATLAB数值。test112对RGB696有小样本波动，应与相同划分和指标环境比较。

先看人车数量/位置/肢体、道路、建筑边界，再看颜色和割裂。场景错位时FID下降不算通过。E、P、Q都有误差，热图不能唯一决定真实颜色。此实现没有自动选出的best；latest指最后完整检查点。

从服务器导出后在本地测试时，保留原run的 `run.json`、`latest.json` 及所选 `checkpoints/step_XXXXXX/` **完整目录**，不要把其中某个文件改名成GAN的 `.pth`。验证校验需要该目录的 `checkpoint.json`、三个 `.safetensors` 和 `training.pt`。本地再按相同配方准备test缓存；不必下载全部中间检查点。

## Ubuntu上传与命令

源代码包 `runs/exports/D_unpaired_full_ubuntu_20261026_v2.tar.gz` 和 `.sha256` 不含数据/权重/缓存。服务器还需：

- 原 `runs/learned_color/data_seed42_v1/cycle_manifest.json` 及其引用数据；
- `runs/learned_color/rgb_prior_seed42_loss_v1.json`；
- FMB `best_model.pth`，放在 `runs/pretrained/segformer_fmb_b3/` 或 `runs/checkpoints/semantic_epoch80_for_mac/`，自动查找；
- 固定HuggingFace模型缓存；
- RGB预热初始化所需的原run `run.json`、`latest.json`、`checkpoints/step_001000/`（含权重和校验文件）。

最直接：通过Jupyter文件面板，把源码包和sha文件上传到 `JiuZzi/Thermal_on_Mac`。你的SSH `gjt` 是宿主机账号，`/home/gjt` 与Jupyter容器的 `/home/ubuntu/work` 并非同一路径；scp到宿主机不会自动进入容器，也不要在Linux运行Windows盘符命令。以下在Jupyter的Ubuntu容器终端执行：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
sha256sum -c D_unpaired_full_ubuntu_20261026_v2.tar.gz.sha256
tar -xzf D_unpaired_full_ubuntu_20261026_v2.tar.gz

conda create -n thermal_d python=3.10 -y
conda activate thermal_d
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r D/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m D.preview --download-models

python -m D.unpaired.prepare \
  --tir-limit 256 \
  --rgb-limit 256 \
  --val-limit 12 \
  --output-dir runs/D/unpaired/conditions/clarity_seed42 \
  --allow-download

python -m D.unpaired.train \
  --name D_tir_unpaired_clarity_pilot_seed42 \
  --data-cache runs/D/unpaired/conditions/clarity_seed42 \
  --init-rgb-run runs/D/training/D_rgb_softE_pilot_seed42 \
  --steps 200 \
  --save-every 50 \
  --log-every 10 \
  --preview-every 50
```

服务器无RGB预热文件时先传上述文件，或去掉初始化参数，记录为不同对照。其余测试/评估/恢复参数相同，仅换续行符。持续运行仍需使用实验室的tmux/作业管理方式。

## 已做验证

2026-10-28本地复核：原版23项、修改后45项CPU测试通过（新增22项），包含真实随机小型Diffusers模型的完整V2更新、RGB回放、冻结SegFormer对图像的梯度、低Q上下文隔离和签名／缓存检查。同输入／种子的小模型V1前后推理、一次更新后的三个网络权重及原有日志逐值一致。测试环境固定Torch2.6.0／Diffusers0.32.2／Transformers4.46.3；Mac ARM的SciPy二进制问题用隔离环境SciPy1.17.1规避，生产requirements未修改。详情及[日志](../../runs/D/review_20261028/current_tests.log)见[审查记录](../../docs/D系列融合与无配对训练审查_2026-10-28.md)。未在本机复跑真实SD权重、CUDA训练或图像质量评估。

以下为仓库此前记录的Windows／GPU验证；相应产物未同步本机，本轮不据此确认完整训练或性能：

10项正式模块测试涵盖E卷积转换、完整采样链、TIR/反向循环梯度、语义/内容梯度、C零输入、冻结参数、RGB回放、签名、采样/图池恢复及非有限梯度跳过。另有原D测试。

真实SD1.5/ControlNet、实际数据、已有step1000RGB初始化下完成4次完整GPU更新，含第4次RGB回放，产物 `runs/D/unpaired/training/D_unpaired_verified_seed42/`。RTX3070Ti Laptop8GB张量峰值约5.53GiB，不含桌面/驱动；本次热缓存约7秒/更新。这是功能测试，不是收敛、完整质量或跨设备速度结论。

另实测从第1更新完整检查点恢复到第4更新：采样顺序、噪声/位置RNG及全局RNG一致，首次恢复前向损失相同。CUDA非确定backward带来微小后续差异，本次扩散权重最大绝对差约1.9e-6，反向网络约4.1e-5，不宣称逐位恢复。重新加载生成结果与对应训练预览相同；原清晰E与新缓存同图逐值一致；12张以下预览被完整分布评估器拒绝。详见 `runs/D/analysis/unpaired_verification_20261026.json`。

研究依据：[ControlNet](https://arxiv.org/abs/2302.05543)、[CUT](https://arxiv.org/abs/2007.15651)、[One-Step Image Translation](https://arxiv.org/abs/2403.12036)。本实现保留SD1.5多步DDIM，不是CycleGAN-Turbo/CUT复现。
