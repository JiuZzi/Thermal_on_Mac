# 结构＋语义保护版：3080 Ubuntu 更新与完整训练

更新时间：2026-10-14。

后续部署方式更新（2026-10-20）：统一使用 tar.gz 手工上传、在项目根目录解压。当前语义／颜色两组入口见[两组 Ubuntu tar 训练](语义颜色两组Ubuntu训练.md)；下方 ZIP 部分保留为旧版部署记录。

服务器项目路径已确认是 `/home/ubuntu/work/JiuZzi/Thermal_on_Mac`，训练子目录为 `pytorch-CycleGAN-and-pix2pix`。原环境为 `c1`，本次语义训练使用你后来建立的 `c1_sem512` 环境。压缩包解压到 **`Thermal_on_Mac` 根目录**，即训练子目录的上一层。

## 1. 当前代码已经接入什么

入口为 `--model semantic_contour_cycle_gan`。数据加载器完成随机 256 裁剪、水平翻转和归一化后，冻结 FMB SegFormer-B3 在线产生 512 语义响应，结合 C2c 的细粗结构候选生成 `joint512`。生成器输入为 `[TIR, E_joint, 0]`；循环路径的合成 TIR 也重新提取条件。

固定提边规则与语义模型不更新，训练的是 CycleGAN 生成器和判别器。保留原 GAN 与 cycle 损失，identity=0；本轮第三通道为零。训练配置为 ResNet-9blocks、ngf/ndf=64、basic 判别器、instance norm、batch=1、seed42、lr=0.0002、40+40 epoch。原 C1 权重不参与初始化，本实验从头训练。

核心代码：

- `pytorch-CycleGAN-and-pix2pix/models/semantic_contour_cycle_gan_model.py`：训练入口与循环路径。
- `pytorch-CycleGAN-and-pix2pix/models/semantic_contour.py`：冻结语义加载与条件生成。
- `pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py`：结构＋语义保护规则。
- `scripts/semantic_preview/`：SegFormer 架构、预处理、完整权重加载及 vendored 编码器。

`semantic_contour.py` 会从项目根目录下的 `scripts/semantic_preview` 导入实现。因此只更新 `pytorch-CycleGAN-and-pix2pix` 会缺少依赖；还必须复制完整外部权重 `runs/pretrained/segformer_fmb_b3/best_model.pth`（约 511 MiB，不能用 encoder-only 的 `mit_b3.pth`）。

## 2. 传输哪些文件

已生成更新包：`runs/exports/semantic_contour_ubuntu_20261014.zip`，校验文件为同名 `.zip.sha256`。包内路径从项目根目录开始：

```text
pytorch-CycleGAN-and-pix2pix/           当前训练源码，含 data/models/options/util
scripts/semantic_preview/              冻结语义实现及来源许可证
scripts/run_semantic_contour_ubuntu.py  Ubuntu 启动与检查
scripts/requirements_semantic_*.txt     补充依赖
scripts/evaluate_distribution.py       后续统一指标
scripts/render_metrics_figure.py
scripts/requirements_metrics.txt
FLIR_protocol_v2/                      划分清单与协议，不包含图像
runs/pretrained/segformer_fmb_b3/best_model.pth
docs/结构语义保护版Ubuntu训练.md
semantic_ubuntu_update_manifest.json   包内文件大小与 SHA256
```

服务器继续使用原 `FLIR_datasets`：trainA=6191、trainB=2110、testA=696、testB=112。更新包不带原始数据、训练图像、旧 GAN 检查点、预览结果、Mac 环境或缓存。完整训练在线计算语义，不依赖之前 40 张开发样本的缓存。包内的 protocol 清单只校验划分文件名；图像应复用原来同一份预处理结果。

未来代码改动后，在 Mac 项目根目录重新打包，选择新文件名：

```bash
python3 scripts/package_semantic_contour_ubuntu.py \
  --output runs/exports/semantic_contour_ubuntu_下一版本.zip
```

## 3. 上传和解压

将 ZIP 和 `.sha256` 上传到服务器 **`Thermal_on_Mac` 根目录**。可以继续用现有压缩包传输工具；如果使用 scp，在 Mac 运行下面的模板，并替换服务器登录地址、端口及实际绝对目录：

```bash
scp -P 22 \
  /Users/tanwenjie/代码/thermal/runs/exports/semantic_contour_ubuntu_20261014.zip \
  /Users/tanwenjie/代码/thermal/runs/exports/semantic_contour_ubuntu_20261014.zip.sha256 \
  服务器用户名@服务器IP:/实际绝对路径/Thermal_on_Mac/
```

在服务器终端检查位置。这里应能看到 `FLIR_datasets` 和 `pytorch-CycleGAN-and-pix2pix`：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
pwd
ls -d FLIR_datasets pytorch-CycleGAN-and-pix2pix
pgrep -af '[p]ython.*train.py'
```

如果还有旧训练正在运行，等它结束后更新源码。确认没有训练进程后，可先保留旧代码，再解压覆盖包内对应文件：

```bash
cp -a pytorch-CycleGAN-and-pix2pix "pytorch-CycleGAN-and-pix2pix_backup_$(date +%Y%m%d_%H%M%S)"
sha256sum -c semantic_contour_ubuntu_20261014.zip.sha256
unzip -o semantic_contour_ubuntu_20261014.zip -d .
```

本包不包含 `runs/checkpoints` 和训练数据，所以解压覆盖不会替换已有实验权重或图像。不要把包解压进训练子目录，否则会产生嵌套目录。

## 4. 检查 c1 环境

先激活你原来的环境，只安装补充依赖；这份 requirements 不包含 torch/torchvision：

```bash
conda activate c1
python -m pip install -r scripts/requirements_semantic_train.txt
nvidia-smi
python -u scripts/run_semantic_contour_ubuntu.py --check-only
```

检查会输出当前 Python、PyTorch、CUDA、GPU、四个划分数量和语义权重 SHA256，核对清单文件名，并在一张真实 trainB 的 256 裁剪上执行完整冻结语义与 `joint512` 前向。`preflight passed` 表示这些启动条件通过；正式 GAN 的峰值显存和训练速度仍由实际训练确定。

本启动器支持 PyTorch>=2.6，以匹配当前 checkpoint 加载使用的类型白名单接口。仓库原 `environment.yml` 固定为 PyTorch 2.4，不能直接用它重建本次环境。如果 `c1` 的检查明确提示版本过旧，可建立独立环境：

```bash
conda create -n c1_sem512 python=3.11 -y
conda activate c1_sem512
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r scripts/requirements_semantic_train.txt
python -u scripts/run_semantic_contour_ubuntu.py --check-only
```

上面的 CUDA 11.8 安装组合来自 [PyTorch 官方历史版本安装表](https://pytorch.org/get-started/previous-versions/#v260)。GPU 能否使用还需上述服务器检查通过；checkpoint 白名单接口见 [PyTorch 2.6 serialization 文档](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch.serialization.safe_globals)。不同 PyTorch/CUDA 环境下的 baseline 结果应记录环境差异；严格比较时需统一环境。

若数据实际不在项目根目录，检查和训练命令都加同一个 `--dataroot /实际/FLIR_datasets`。选择某张物理 GPU 时，在命令前设置 `CUDA_VISIBLE_DEVICES=卡编号`，程序会使用这张可见卡的 cuda:0。

## 5. 从头跑完整 80 epoch

在项目根目录、已激活且通过检查的 CUDA 环境中运行。默认就是完整 40+40，未设置五轮提前停止：

```bash
mkdir -p runs/training
nohup python -u scripts/run_semantic_contour_ubuntu.py \
  > runs/training/struct_sem512_ubuntu_20261014.log 2>&1 &
tail -f runs/training/struct_sem512_ubuntu_20261014.log
```

`Ctrl+C` 只退出当前 `tail`。启动器先用独立进程检查并释放该检查所占 GPU 内存，再启动训练；训练成功结束后自动使用第 80 轮编号检查点，按 `preprocess=none`、`no_flip`、`eval` 导出前 112 个固定 testB 输入。训练或测试子进程失败会返回失败退出码，不会继续报告成功。

默认实验名为：

```text
flir_v2_struct_sem512_protected_seed42_ubuntu_20261014_40_40
```

实验目录已经存在时启动器停止，防止误覆盖。重跑时加新的 `--name`，并使用新的日志文件。例如 `--name flir_v2_struct_sem512_protected_seed42_ubuntu_run2_40_40`。若只做五轮诊断可加 `--stop-after-epoch 5` 和独立实验名；它保持原 40+40 学习率计划，但不是完整训练。

原 conditioned/semantic 训练没有保存完整 optimizer 和 RNG 状态；本启动器不提供严格续训，也不接着旧 C1 或五轮网络权重训练。完整预算实验应从头启动。被中断后的网络权重续训不能当作严格等价的不中断实验。

## 6. 看哪些输出

```text
runs/training/struct_sem512_ubuntu_20261014.log
runs/checkpoints/<实验名>/loss_log.txt
runs/checkpoints/<实验名>/web/index.html
runs/checkpoints/<实验名>/80_net_G_A.pth
runs/checkpoints/<实验名>/80_net_G_B.pth
runs/checkpoints/<实验名>/semantic_train.json
runs/results/<实验名>/test_80/index.html
runs/results/<实验名>/test_80/images/*_fake_B.png
```

每五轮保存编号检查点。训练页的 `contour_joint` 是送入生成器的实际第二通道；测试页的 `real_A` 是 TIR，`fake_B` 是生成 RGB。`real_B` 是无配对 RGB 样本，不能作为同一 TIR 的颜色真值。

完整测试应有 112 张 `_fake_B.png`；服务器检查命令：

```bash
python -c 'from pathlib import Path; p=Path("runs/results/flir_v2_struct_sem512_protected_seed42_ubuntu_20261014_40_40/test_80/images"); n=len(list(p.glob("*_fake_B.png"))); print("Generated TIR-to-RGB images:",n); assert n==112'
```

若第 80 轮训练已完成，但测试导出失败，修复失败原因后可单独导出，不重新训练：

```bash
python -u scripts/run_semantic_contour_ubuntu.py --test-only --epoch 80
```

已有 `test_80` 目录时先将失败的输出目录改名保留，再运行此命令。

## 7. 训练后统一评价

包内包含既有的统一评估脚本及画图依赖，完成训练后单独运行：

```bash
python -m pip install -r scripts/requirements_metrics.txt
python scripts/evaluate_distribution.py \
  --tir-dir FLIR_datasets/testB \
  --generated-dir runs/results/flir_v2_struct_sem512_protected_seed42_ubuntu_20261014_40_40/test_80/images \
  --generated-suffix _fake_B.png \
  --reference-rgb-dir FLIR_datasets/testA \
  --output-dir runs/evaluations/flir_v2_struct_sem512_protected_seed42_ubuntu_20261014_40_40 \
  --device cuda --batch-size 8
```

首次 FID/KID 评价可能需要下载 Inception 特征权重。对照组应使用相同 testB、训练预算、种子、测试尺寸与评价参数；包里保留原 C1/C2 入口。当前轮廓预览用于开发诊断，正式 RGB 效果和重要目标边界是否改善由训练后的对照结果判断。

## 8. 本次验证范围

本次未改动语义融合、GAN 架构、损失或原训练循环。已验证 Ubuntu 启动入口的参数解析、包内训练/测试 CLI 导入、真实外部权重与一个真实 TIR 裁剪的 CPU 在线轮廓，并检查压缩包包含文件、校验和及解压后的依赖导入。3080 的 CUDA 环境、完整 80 轮训练、显存峰值、耗时和 RGB 收益需在服务器运行后确认。

## 9. 训练效率更新：删除未使用的计算

增量更新包为 `runs/exports/semantic_training_fastpath_20261014.zip`，配套同名 `.zip.sha256`。它用于已经解压过上面完整语义更新包的服务器，只包含修改后的代码、本文和验证记录，不包含训练数据、GAN 检查点或语义权重。

本次训练路径直接生成所需的 `joint512`，不再计算 C2b 对照轮廓、C2c 对照输出或 semantic-v1 对照输出；细粗候选、结构重要度和纹理图只计算一次。冻结语义推理只返回类别概率，不再计算标签图、熵或最大类别分数。离线预览工具仍能生成原来的对照和诊断结果。

每次条件生成的 Canny 调用由 12 次降为 9 次，结构重要度/纹理计算由 2 次降为 1 次。512 语义推理、预处理、插值、细粗候选、所有阈值及融合公式、当前裁剪/翻转、合成 TIR 的循环条件、GAN 架构/损失/优化器均保持原样；训练命令、seed42 和 40+40 计划无需改变。

将更新 ZIP 和校验文件上传到项目根目录，当前训练结束后运行：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
sha256sum -c semantic_training_fastpath_20261014.zip.sha256
unzip -o semantic_training_fastpath_20261014.zip -d .
```

服务器必须同时更新训练子目录和 `scripts/semantic_preview/inference.py`，不能只复制其中一处。无需重装环境或再次上传 `best_model.pth`；下一次启动原来的多行训练命令即可使用新路径。已经运行的 Python 进程不会自动切换源码；不要为了应用优化贸然中断正式实验。代码中的语义配置签名保持一致，现有网络权重格式仍兼容，但原训练器没有保存完整优化器与随机数状态，网络权重续训不能视为严格等价的连续训练。

本地验证记录在 `runs/smoke/semantic_training_fastpath_20261014/verification.json`：

- 60 项相关单元测试通过；连续 3 次 CPU 小网络更新的新旧损失、输出、梯度、参数及优化器状态完全一致。
- 与修改前源码独立对比：真实冻结权重在 256 裁剪、水平翻转、整图和连续合成 TIR 上的概率图、`joint512` 均逐像素一致，最大绝对差为 0。
- 真实语义权重参与的一次 CPU 小网络 GAN 更新，包括合成 TIR 的循环路径，新旧损失、输出、梯度、参数和优化器状态完全一致。
- C2c 的 5 个既有变体在 15 个输入/变体组合上输出完全一致；预览接口和语义配置签名保持一致。

这些验证确认计算结果没有改变；尚未测量 RTX 3080 Ti 上的加速比例，也没有代替完整训练后的质量评价。每步两次 SegFormer-B3 推理仍是必要计算，因此不能按 Canny 次数的减少比例估计整步加速。
