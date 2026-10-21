# 语义与颜色：两组 Ubuntu tar 更新与试跑

日期：2026-10-20。用户选择今后沿用 **tar.gz 上传、解压更新**；本次由用户自行上传，不通过远程登录代为操作。

## 本轮比较什么

| 组别 | 实验名 | 颜色约束 |
|---|---|---|
| 仅语义 | `flir_v2_semonly_from80_seed42_20261020_preview5` | 完整15类条件；人工颜色特征置零，颜色损失0 |
| 语义＋人工颜色先验 | `flir_v2_semcolor_from80_seed42_20261020_preview5` | 完整15类＋人工颜色范围；颜色损失权重1 |

两组均保留原始 TIR、joint512、第三零通道，从同一旧第80轮 G_A/G_B 初始化。每组新增5轮，seed42、学习率0.00002、256裁剪、batch1；判别器和优化器重新开始。两组顺序运行，每组结束自动导出112张固定 testB。

这是相同起点的短期着色诊断，不能把它称作从头5轮、完整新训练或严格续训。人工先验可能造成类别套色或强化语义错误，本轮就要与仅语义组比较这些问题。旧版80轮结果可作参考；若要排除额外训练本身的影响，仍需后续同预算 `none` 对照。

## 本地包与以后打包方式

本次重新生成包：`runs/exports/semantic_color_ubuntu_20261020_repack1.tar.gz`，配套 `.tar.gz.sha256`。它恢复了误删的更新包和根目录清单，训练算法与两组各5轮的配置不变。从项目根目录重新生成时使用新文件名：

```bash
python3 scripts/package_semantic_color_ubuntu.py --output runs/exports/semantic_color_ubuntu_下一版本.tar.gz
```

包包含当前训练源码、完整 `scripts/semantic_preview/`、新旧启动器、补充依赖清单、说明和来源许可证。不带数据图像、预览缓存、语义权重、GAN检查点或登录信息。

服务器应已有：

```text
FLIR_datasets/{trainA,trainB,testA,testB}
FLIR_protocol_v2/{manifest.csv,protocol.json}
runs/pretrained/segformer_fmb_b3/best_model.pth
runs/checkpoints/flir_v2_struct_sem512_protected_seed42_ubuntu_aligned_40_40/80_net_G_A.pth
runs/checkpoints/flir_v2_struct_sem512_protected_seed42_ubuntu_aligned_40_40/80_net_G_B.pth
```

包内 `semantic_color_ubuntu_update_manifest.json` 记录源码与这些已有权重／协议的 SHA256。安装校验将检查它们，避免从另一组旧权重开始；不写入已有权重。解压后将这个 JSON 保留在项目根目录，批量启动脚本会读取它。

## 上传和解压

用户自行把 **tar.gz 与 sha256 两个文件**上传到 JupyterLab 链接中的 `JiuZzi/Thermal_on_Mac` 项目文件夹。

解压位置必须是包含 `FLIR_datasets`、`scripts` 和 `pytorch-CycleGAN-and-pix2pix` 的项目根目录。2026-10-20 的终端输出已确认当前服务器根目录为 `/home/ubuntu/work/JiuZzi/Thermal_on_Mac`。

当前仍有训练运行时，等其结束再覆盖源码。在项目根目录终端执行：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
ls -d FLIR_datasets pytorch-CycleGAN-and-pix2pix
sha256sum -c semantic_color_ubuntu_20261020_repack1.tar.gz.sha256 &&
tar -xzf semantic_color_ubuntu_20261020_repack1.tar.gz -C .
```

## 检查和顺序运行两组

当前服务器有两个同名 `c1_sem512`。用户已确认项目路径下的环境具有 PyTorch 2.6.0+cu118、CUDA可用，RTX3080Ti 预检通过；`/opt/conda/envs/c1_sem512` 缺少 torch。使用完整路径激活已经通过检查的环境：

```bash
conda activate /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512
python scripts/package_semantic_color_ubuntu.py --verify-installed
python -u scripts/run_semantic_color.py --check-only
```

校验和 CUDA 检查通过后：

```bash
mkdir -p runs/training
PYTHON_BIN=/home/ubuntu/work/JiuZzi/conda_envs/c1_sem512/bin/python \
nohup bash scripts/run_semantic_color_pair_ubuntu.sh > runs/training/semantic_color_pair_20261020_repack1.log 2>&1 &
tail -f runs/training/semantic_color_pair_20261020_repack1.log
```

脚本在运行第一组前检查两组目录均未存在；第一组训练并导出完成后才启动第二组。任一步失败立即停止。它不会自动删除已有实验或在失败后从头重跑。`Ctrl+C` 退出 `tail`，后台训练继续。

## 查看和取回结果

```text
runs/training/semantic_color_pair_20261020_repack1.log
runs/results/flir_v2_semonly_from80_seed42_20261020_preview5/test_5/index.html
runs/results/flir_v2_semcolor_from80_seed42_20261020_preview5/test_5/index.html
```

测试页中 `fake_B` 才是生成 RGB，`color_hint` 是人工颜色提示。仅语义组虽然也记录提示诊断图，但颜色特征和颜色损失已关闭，不能据诊断图误判它使用了人工颜色输入。`real_B` 是无配对 RGB，不能当作该 TIR 的真值。

两组完成后，服务器根目录打包结果供下载回 Mac：

```bash
tar -czf semantic_color_pair_results_20261020.tar.gz \
  runs/training/semantic_color_pair_20261020_repack1.log \
  runs/results/flir_v2_semonly_from80_seed42_20261020_preview5/test_5 \
  runs/results/flir_v2_semcolor_from80_seed42_20261020_preview5/test_5 \
  runs/checkpoints/flir_v2_semonly_from80_seed42_20261020_preview5/color_train.json \
  runs/checkpoints/flir_v2_semonly_from80_seed42_20261020_preview5/loss_log.txt \
  runs/checkpoints/flir_v2_semcolor_from80_seed42_20261020_preview5/color_train.json \
  runs/checkpoints/flir_v2_semcolor_from80_seed42_20261020_preview5/loss_log.txt
sha256sum semantic_color_pair_results_20261020.tar.gz > semantic_color_pair_results_20261020.tar.gz.sha256
```

结果包不包含大检查点；完成后可先据同一批测试图和日志判断是否值得延长训练。

2026-10-20回传更新：用户上传包已校验，两组各5轮和112图导出均完整。[全部对照与结果报告](../runs/evaluations/semantic_color_pair_20261020/index.html)已完成；人工颜色版尚未呈现清晰着色收益，详见报告中的结构代理、颜色统计及具体案例。
