# 从零训练：语义、RGB学得颜色先验与微量颜色损失

日期：2026-10-21。以下只训练一个主实验：`rgb_prior_scratch80_loss001_seed42_v1`。

G_A、G_B及判别器重新初始化，不读取旧第80轮或仅语义第5轮生成器，也不使用continue_train。两个语义分割器仍使用预训练权重并冻结；GMM从当前RGB训练区域离线拟合后冻结。“从零”指彩色化网络重新训练，并不表示放弃已有分割知识。

本次80轮是原项目40+40训练预算的起点：前40轮初始学习率0.0002，后40轮线性衰减；batch1、裁剪256、seed42。微量颜色损失目标权重0.01、前1000次更新渐增，仅Q>=0.5的支持类别内部像素生效。预算不代表已经证明收敛或不会过拟合。

流程：更新代码 → 环境 → 按视频划分 → RGB语义缓存 → 审查及GMM拟合 → 预检 → 从零训练 → val选择检查点 → test导出。

若已训练后因`/tmp/pymp-*`空间不足而停滞，已有检查点请使用[权重与更新计数恢复流程](临时目录故障后恢复学得颜色训练.md)，不要再次以`--from-scratch`启动同名实验。

## 1. 上传并解压最新源码包

将以下两个文件上传到项目根目录：

- `learned_color_scratch80_ubuntu_20261021.tar.gz`
- `learned_color_scratch80_ubuntu_20261021.tar.gz.sha256`

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512
sha256sum -c learned_color_scratch80_ubuntu_20261021.tar.gz.sha256 &&
tar -xzf learned_color_scratch80_ubuntu_20261021.tar.gz -C .
python scripts/package_learned_color_ubuntu.py --verify-installed
mkdir -p runs/training
```

新包包含从零训练入口，无需依次解压之前的包。它不包含数据、RGB教师、颜色先验或FMB权重；保留解压后的`learned_color_update_manifest.json`。已有`FLIR_datasets`、`FLIR_protocol_v2`和`runs/pretrained/segformer_fmb_b3/best_model.pth`复用。

## 2. 检查环境并补依赖

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pip install -r scripts/requirements_semantic_train.txt -r scripts/requirements_learned_color_teacher.txt
python -c "from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation; import scipy, skimage, omegaconf; print('Dependencies imported')"
python -m pip freeze > runs/training/scratch80_environment_20261021.txt
sha256sum runs/pretrained/segformer_fmb_b3/best_model.pth
```

应为正确完整路径的Python、PyTorch>=2.6且CUDA=True。FMB权重沿用已核对版本，SHA256应为`a3f9de85452534f7731f01ef64cebee4d62c9aa4ea133861c7bb3124212aa1fb`。依赖清单不主动重装原CUDA torch/torchvision；若当前环境缺少torch，先解决环境，不在错误同名环境启动。新RGB教师真实权重仍需在服务器执行下一阶段下载与推理，本机未替代该验证。

## 3. 生成共享的视频分组清单

```bash
python scripts/prepare_grouped_color_data.py \
  --csv FLIR_protocol_v2/manifest.csv --image-root FLIR_datasets \
  --image-column destination_name --group-column video_id \
  --role-column destination_split --role-is-subdir \
  --train-a-role trainA --train-b-role trainB \
  --test-a-role testA --test-b-role testB \
  --dataset-id flir_v2 --classes road sidewalk vegetation sky \
  --val-fraction 0.2 --seed 42 \
  --output-dir runs/learned_color/data_seed42_v1
```

输出`cycle_manifest.json`供GAN使用、`rgb_manifest.json`供RGB缓存使用。当前协议应产生RGB5531/660/696、TIR1420/690/112张train/val/test。同视频／场景不会跨角色，RGB与TIR共享组划分。若已存在同一份有效清单，直接复用，勿反复换seed挑结果；这不涉及旧彩色化权重。

## 4. 离线生成RGB语义缓存

若服务器连接Hub失败，可改用已准备的[官方权重离线tar包与本地目录命令](RGB教师离线权重与网络错误排查.md)，无需重新打包训练代码。该方案用`rgb_teacher_seed42_offline_v1`目录，后续fit／evaluate的manifest随之改为该目录；GAN使用的先验JSON路径保持一致。

```bash
nohup python -u scripts/learn_color_prior.py cache \
  --manifest runs/learned_color/data_seed42_v1/rgb_manifest.json \
  --teacher nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
  --revision main --allow-download --device cuda:0 --flip \
  --min-probability 0.9 --min-margin 0.2 --min-agreement 0.9 --erosion 2 \
  --output-dir runs/learned_color/rgb_teacher_seed42_v1 \
  > runs/training/rgb_teacher_seed42_v1.log 2>&1 &
```

```bash
tail -f runs/training/rgb_teacher_seed42_v1.log
```

等待末尾出现`Cached manifest:`；此时按Ctrl+C只是退出tail，随后检查：

```bash
ls -lh runs/learned_color/rgb_teacher_seed42_v1/manifest.json
```

默认只缓存RGB train/val共6191张，跳过test的696张。进度分母包含原test条目，最后6191/6887并不表示遗漏训练数据。`index.html`包含按组的原RGB、预测类别与筛选后内部区域；在Jupyter文件列表中查看，重点检查植被／天空混淆以及保留像素是否过少。

首次`main`解析为具体snapshot，清单记录`teacher.resolved_commit`、权重哈希、预处理和筛选参数；复现时固定该commit。高分伪标签仍可能错，显示色不参与拟合。已有完整且与上述清单、参数一致的缓存可复用，不必重跑教师；程序拒绝覆盖已有输出目录，半成品不能当作完成缓存。

这里缓存的是**RGB训练样本的类别标签**。红外FMB响应、Q和joint512在GAN中对每次裁剪／翻转后的TIR及循环生成TIR在线计算，没有另一个必须先执行的TIR离线缓存步骤。

## 5. 拟合并审查颜色先验

```bash
python -u scripts/learn_color_prior.py fit \
  --manifest runs/learned_color/rgb_teacher_seed42_v1/manifest.json \
  --config configs/learned_color/regularized_v1.json \
  --output runs/learned_color/rgb_prior_seed42_loss_v1.json

python -u scripts/learn_color_prior.py evaluate \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --manifest runs/learned_color/rgb_teacher_seed42_v1/manifest.json \
  --role val \
  --output runs/learned_color/rgb_prior_seed42_loss_v1_val.json
```

GMM中心、方差、比例及损失支持阈值只从组平衡RGB train色样得到，val选择小容量模式数；test不读取。查看支持组数、验证NLL、训练—验证差距、bootstrap及`loss_support`覆盖比例。val本身参与选择，因此上述评价是开发诊断。类别样本不足时保持unsupported，不填人工颜色。已有新版正式先验可以复用；不得将67块开发种子产物当作全量先验，也不要手工给旧JSON填一个阈值。

## 6. 从零训练预检

```bash
python -u scripts/run_learned_color.py --check-only --from-scratch \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_scratch80_loss001_seed42_v1 --mode learned --gate-semantic \
  --epochs 80 --decay-epochs 40 --lr 0.0002 --save-every 5 \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000
```

应显示`initialization: fresh GAN networks`，总80轮、constant40、decay40、lr0.0002。它检查清单、重复图像、先验与验证／测试集交叉、文件存在和配置；不执行完整FMB前向，也不是性能测试。`--from-scratch`会拒绝任何旧生成器初始化参数。

## 7. 正式启动80轮

```bash
nohup python -u scripts/run_learned_color.py --train --from-scratch \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_scratch80_loss001_seed42_v1 --mode learned --gate-semantic \
  --epochs 80 --decay-epochs 40 --lr 0.0002 --save-every 5 \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000 \
  > runs/training/rgb_prior_scratch80_loss001_seed42_v1.log 2>&1 &
```

```bash
tail -f runs/training/rgb_prior_scratch80_loss001_seed42_v1.log
```

不传`init-condition-dir`、`init-contour-dir`或continue_train。每5轮保留编号检查点，完整第80轮也会保存，训练与RGB缓存依次占用GPU。不要对已启动的同名实验重复执行命令；启动器会保护已有目录。日志含`color_prior_raw`、实际加权`color_prior`、`color_coverage`和渐增`color_weight`；微小值显示0.000可能是三位小数舍入。

完成时检查`End of epoch 80 / 80`及新实验目录下`80_net_G_A.pth`。`color_train.json`应记录`initialization: fresh`且没有旧G_A/G_B来源；`train_opt.txt`记录实际参数。G_A/G_B/D从新初始化开始，原有语义分支和新颜色分支随GAN训练，FMB与GMM冻结。

## 8. 导出val并选择检查点

训练结束后，可一次导出40／60／80轮，逐组审查多场景下的错色、边界串色、结构、服装斑驳和自然度，再决定最终epoch：

```bash
for epoch in 40 60 80; do
  python -u scripts/run_learned_color.py --test-only --phase val --epoch "$epoch" \
    --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
    --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
    --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
    --name rgb_prior_scratch80_loss001_seed42_v1 --mode learned --gate-semantic \
    --color-loss-weight 0.01 --color-loss-warmup-steps 1000 || break
done
```

对应`runs/results/rgb_prior_scratch80_loss001_seed42_v1/val_40/`、`val_60/`、`val_80/`，当前每组690张TIR。没有自动早停或自动最优检查点评分。先验变动／颜色损失配置变动后不能静默加载同一签名；导出保持与训练一致。

## 9. 固定选择后导出test

下面以选定80轮为例；如果验证选择40或60，替换epoch。测试用于最终报告，不按测试结果反复修改颜色规则：

```bash
python -u scripts/run_learned_color.py --test-only --phase test --epoch 80 \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_loss_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_scratch80_loss001_seed42_v1 --mode learned --gate-semantic \
  --color-loss-weight 0.01 --color-loss-warmup-steps 1000
```

输出`runs/results/rgb_prior_scratch80_loss001_seed42_v1/test_80/index.html`与images，当前112张TIR。RGB与TIR非配对，不计算逐像素PSNR／SSIM作为真值比较。原test已经用于前期开发分析，严格最终泛化评价仍需新视频或外部数据；从零训练消除旧着色权重见过新val的影响，不会自动消除开发测试集偏差或分割器域偏差。

## 10. 打包结果回传

```bash
mkdir -p runs/exports
tar -czf runs/exports/rgb_prior_scratch80_loss001_seed42_v1_results.tar.gz \
  runs/results/rgb_prior_scratch80_loss001_seed42_v1 \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/color_train.json \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/train_opt.txt \
  runs/learned_color/rgb_prior_seed42_loss_v1.json \
  runs/training/rgb_prior_scratch80_loss001_seed42_v1.log
cd runs/exports
sha256sum rgb_prior_scratch80_loss001_seed42_v1_results.tar.gz \
  > rgb_prior_scratch80_loss001_seed42_v1_results.tar.gz.sha256
cd ../..
```

下载结果包及校验文件，包含图像、训练日志、先验JSON与color_train.json，无需打包整个checkpoints目录。若要分离颜色损失的作用，再用同样80轮、相同划分和seed从零训练无损失对照，设置新name和`--color-loss-weight 0`；本流程不会自动启动第二个实验。
