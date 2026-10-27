# D：扩散 TIR→RGB

已实现正式 **[TIR,E,0] 真实TIR无配对训练**，保留清晰/显著性轮廓、语义和颜色先验，包含可微DDIM、循环、对抗与结构目标。当前主入口及训练、恢复、测试、评估命令见 **[unpaired/README.md](unpaired/README.md)**。真实GPU短测试通过，完整训练效果尚未验证。

之前RGB自配对预热仍见 [TRAINING.md](TRAINING.md)，入口 `D.train`，可作正式版本初始化。下文保留最初零训练D0/D1对照，入口 `D.preview`。

本版本先回答一个问题：在同一预训练 RGB 扩散模型上，接入现有清晰软轮廓，能否更稳定地保留人车的位置和边界？

| 对照 | 红外内容输入 | 轮廓控制 | 本轮训练 |
|---|---|---|---|
| D0 | 原红外灰度复制三通道，作为 img2img 初始图 | ControlNet 残差强度 0 | 无 |
| D1 | 与 D0 相同 | 现有 `clarity_v1` E，ControlNet 强度 1 | 无 |

两者使用同一个 `StableDiffusionControlNetImg2ImgPipeline`、同一套基础权重、提示词、DDIM、尺寸、强度与噪声。每张图、每个版本均重新建立 CPU generator，固定 seed42。D0 的零残差与普通 img2img 的等价性用真实 diffusers 小模型验证。

这是零训练接入预览，尚未训练红外域内容编码器或软轮廓适配器。D1 借用在 RGB 硬 Canny 图上预训练的 ControlNet；输入软 E 存在条件域差异。D0/D1 的差别包含接入这个预训练控制分支的作用，不能单凭本轮声称清晰 E 优于普通 Canny。

E 原样复用现有算法：保留结构/语义重要性 R、纹理抑制 D 和清晰轮廓提取。冻结 FMB 语义模型只参与 E 的生成；本轮没有独立语义特征注入、GMM 颜色先验、JR 或轮廓置信度。不会读取旧 GAN 生成器权重。

所有新产物位于 `runs/D/`。现有 GAN 代码、数据划分、权重和训练入口均不修改。

## 1. 单独环境

以下命令从项目根目录运行，即同时包含 `D/`、`FLIR_datasets/`、`scripts/` 的 `Thermal_on_Mac`。

Ubuntu / Windows PowerShell 均可逐条运行：

```text
conda create -n thermal_d python=3.10 -y
conda activate thermal_d
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r D/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

固定 CUDA PyTorch 单独安装，后续要求中的 `torch==2.6.0` 接受已安装的 `2.6.0+cu118`。无需更改现有 `c1_sem512` 或 `thermal_train` 环境。

## 2. 先缓存验证集 12 张 E

沿用 `runs/learned_color/data_seed42_v1/cycle_manifest.json` 的 train/val/test，不重新划分。默认从验证集按视频组、seed42 固定选 12 张，避免只看同一视频的连续帧。未把测试集当作调参预览集。

Ubuntu：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate thermal_d

python -m D.prepare \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --phase val \
  --limit 12 \
  --output-dir runs/D/conditions/val_seed42
```

Windows PowerShell：

```powershell
Set-Location "E:\上海大学\项目\thermal\Thermal_on_Mac"
conda activate thermal_d

python -m D.prepare `
  --semantic-checkpoint runs/checkpoints/semantic_epoch80_for_mac/best_model.pth `
  --phase val `
  --limit 12 `
  --output-dir runs/D/conditions/val_seed42
```

默认用 CPU 和 4 个线程运行冻结语义推理。空闲 GPU 可增加 `--semantic-device cuda:0`。不修改原始图像：在现有数据集原尺寸上生成 E，保存原始灰度、float32 E、原重要性和增益，并记录原图、语义权重、算法代码的 SHA256。已存在的缓存不会被覆盖。

## 3. 下载一次固定版本模型

```text
python -m D.preview --download-models
```

下载到 `runs/D/model_cache/`，首次需要数 GB 空间。基础模型和 ControlNet 均固定到配置中的完整 upstream commit，只取需要的 fp16 safetensors。此命令下载权重，不运行推理。后续推理仅使用本地缓存，不暗中再次联网下载。

模型来自：

- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet v1.1 Canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)
- [官方 ControlNet img2img 接口](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)

PyPI 清华镜像仅负责依赖安装，不负责 Hugging Face 模型下载。下载中断可重跑同一下载命令。

## 4. 跑 D0 和 D1

先检查输入与依赖（不会加载扩散权重）：

```text
python -m D.preview --check-only
```

运行两组对照：

```text
python -m D.preview
```

默认 batch=1、fp16、模型 CPU offload、关闭 VAE 分块解码。这样降低显存需求，并避免分块边界干扰割裂现象判断；是否适合当前 GPU 仍以实际占用为准。

默认配置位于 `D/configs/sd15_clarity_seed42.json`：seed42，640×512，30 个 DDIM 总采样步，img2img strength=0.6（实际约 18 个去噪步），CFG=7.5。原始 360×288 图与 640×512 保持比例；其他比例必须修改模型尺寸，程序拒绝拉伸。E 以 float32 三通道传入，只做双线性缩放，没有二值化、反色、阈值化或逐图拉满亮度。

输出保留模型尺寸图，同时将结果还原到源图尺寸，用 `_fake_B.png` 与既有评估命名对应。提示词对所有图片固定，不根据测试图或标签逐张添加人车。

结果入口：

```text
runs/D/previews/val_seed42_v1/index.html
runs/D/previews/val_seed42_v1/comparisons/
runs/D/previews/val_seed42_v1/D0/images/
runs/D/previews/val_seed42_v1/D1/images/
runs/D/previews/val_seed42_v1/run.json
runs/D/previews/val_seed42_v1/records.json
```

网页逐张展示红外 / E / D0 / D1。原尺寸与模型尺寸都保留，方便区分模型变化和缩放影响。保留模型自身安全检查，逐图记录标记；评估入口会拒绝未检查或被过滤的结果。

断电恢复同一配置：

```text
python -m D.preview --resume
```

仅跳过已记录且 SHA256 一致的图片；改了配置、缓存、版本或代码会拒绝混入旧结果。修改配置时使用新输出目录。不同设备和库版本之间不保证逐像素复现。

## 5. 何时评估、如何判断

先检查 12 张全部对照：人车位置、数量、外轮廓、细肢体、建筑几何，然后看自然颜色、纹理和割裂。出现新增/消失的人车也应记录，不能仅挑颜色好的样例。

12 张用于机制检查，不计算小样本 FID 来证明提升。确认值得继续后，准备完整验证集或完整测试集（`--limit 0`），在新条件目录和新预览目录生成，再调用同一套旧指标：

```bash
python -m D.prepare \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --phase test --limit 0 \
  --output-dir runs/D/conditions/test_seed42

python -m D.preview \
  --conditions runs/D/conditions/test_seed42 \
  --output-dir runs/D/previews/test_seed42_v1

python -m D.evaluate \
  --preview-dir runs/D/previews/test_seed42_v1 \
  --output-dir runs/D/evaluations/test_seed42_v1
```

PowerShell 将行末 `\` 换为反引号，并使用本地语义权重路径。评估默认完整 testB=112 对完整 testA=696，实际数量以原 manifest 为准；包含 FID、KID（seed42、50×100）、APCE-Py、edge precision/recall/F1，复用 `scripts/evaluate_distribution.py`，没有修改指标公式。APCE/edge 是结构代理指标，无法证明真实物体边界准确率；无配对数据不报告随机 RGB 配对的 PSNR/SSIM。

## 6. 本机已完成的首轮预览

2026-10-26：已在本地 RTX 3070 Ti Laptop 上实际完成默认 12 张验证图的 D0/D1，共 24 张输出，约 152 秒；没有被安全检查过滤的图片。已有 7 项协议与真实小模型验证通过。CUDA 启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并启用确定性算法；正式运行的全部源码与输出校验值已核对。

直接查看 `runs/D/previews/val_seed42_v1/index.html`；三页总览和审阅记录在 `runs/D/analysis/val_seed42_v1/`。本机用于验证的隔离环境为 `runs/D/verify_env/`，使用现有 thermal_train 的基础依赖并单独安装扩散库，没有修改 thermal_train；服务器仍建议按上文建立 thermal_d。

首轮现象：D1 比 D0 更接近源图的道路布局与部分目标位置，但仍有建筑、车辆、人物的形态改写；两组几乎都是灰度图。当前仅证明预训练控制分支接收并响应了 E，尚未实现目标 RGB 彩色化，也未证明优于已有 GAN 或清晰 E 优于普通 Canny。保留这套初始设置作为对照，后续先讨论条件域适配，再决定下一步。

## 7. 上传与验证

生成不含数据、权重和缓存的源码包：

```text
python -m D.package --output runs/exports/D_preview_ubuntu_20261026_v2.tar.gz
```

上传 `.tar.gz` 与 `.sha256` 到服务器工程根目录。包内有 `D/` 和当前未改动的轮廓/评估依赖；不含数据清单与 FMB 权重，服务器继续使用已有文件。它会覆盖包内列出的共享源码，因此不要在服务器共享源码已另行修改时盲目解压。

```bash
sha256sum -c D_preview_ubuntu_20261026_v2.tar.gz.sha256
tar -xzf D_preview_ubuntu_20261026_v2.tar.gz -C .
```

离线的小模型/协议验证：

```text
python -m unittest discover -s D/tests -v
```

测试随机初始化的极小模型仅检查接口、零残差等价、条件响应和重复性，不能用于判断正式模型的图像质量。当前阶段没有完成红外域无配对适配训练，不能预先承诺比 GAN 更好。
