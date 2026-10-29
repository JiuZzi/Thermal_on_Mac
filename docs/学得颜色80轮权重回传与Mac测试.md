# Ubuntu学得颜色80轮：只回传所需权重，在Mac测试

日期：2026-10-28。对应`rgb_prior_scratch80_loss001_seed42_v1`，是此前CycleGAN＋语义＋学得GMM＋0.01微量颜色损失实验，不是Windows正在训练的D扩散模型。

## 1. 本次完成证据

用户回传的`rgb_prior_scratch80_loss001_seed42_v1_resume1.log`末尾为：

```text
saving the model at the end of epoch 80, iters 442480
End of epoch 80 / 80     Time Taken: 2574 sec
Stopped after complete epoch 80; original 40+40 schedule preserved.
```

这说明恢复后的训练循环正常达到原80轮预算；原`.log`中的ENOSPC是此前中断记录，不应误判为本次最终退出原因。服务器实际权重尚未回传本机，文件完整性及最终图像仍待检查。此前恢复没有原Adam／RNG／图池完整状态，实验记录保留这项限制。

## 2. Ubuntu只打包生成器与测试必要信息

选40、60、80轮的G_A/G_B用于验证比较。现有CycleGAN测试路径会加载两份G并计算循环，所以G_B也要带；D_A/D_B、latest、全部中间权重、训练图像、语义缓存和巨大的recovery备份权重不需要打包。

先确认六份文件存在：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac

ls -lh runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/{40,60,80}_net_G_{A,B}.pth
```

打包命令（Ubuntu Bash）：

```bash
mkdir -p runs/exports

sha256sum runs/pretrained/segformer_fmb_b3/best_model.pth \
  > runs/exports/rgb_prior_scratch80_fmb_20261028.sha256 &&
tar -czf runs/exports/rgb_prior_scratch80_weights_20261028.tar.gz \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/{40,60,80}_net_G_{A,B}.pth \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/train_opt.txt \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/color_train.json \
  runs/checkpoints/rgb_prior_scratch80_loss001_seed42_v1/loss_log.txt \
  runs/learned_color/rgb_prior_seed42_loss_v1.json \
  runs/learned_color/data_seed42_v1/cycle_manifest.json \
  runs/learned_color/data_seed42_v1/rgb_manifest.json \
  runs/training/rgb_prior_scratch80_loss001_seed42_v1.log \
  runs/training/rgb_prior_scratch80_loss001_seed42_v1_resume1.log \
  runs/exports/rgb_prior_scratch80_fmb_20261028.sha256 &&
(
  cd runs/exports &&
  sha256sum rgb_prior_scratch80_weights_20261028.tar.gz \
    > rgb_prior_scratch80_weights_20261028.tar.gz.sha256
)
```

下载`runs/exports/`下这两个文件：

- `rgb_prior_scratch80_weights_20261028.tar.gz`
- `rgb_prior_scratch80_weights_20261028.tar.gz.sha256`

如果40或60轮已经被清理，先说明实际保留情况再调整列表；不要忽略tar缺文件错误后使用部分包。若服务器FMB不在上述路径，以`train_opt.txt`记录的位置为准，生成对应SHA清单，不随便换语义权重。

## 3. Mac在独立目录接收

将下载的两文件放入本地项目`/Users/tanwenjie/代码/thermal/runs/checkpoints/`。以下在Mac终端执行：

```zsh
cd /Users/tanwenjie/代码/thermal/runs/checkpoints
shasum -a 256 -c rgb_prior_scratch80_weights_20261028.tar.gz.sha256
```

校验为OK后解压到新目录，保留原本地数据／先验：

```zsh
cd /Users/tanwenjie/代码/thermal
received_run=runs/imports/rgb_prior_scratch80_20261028
mkdir -p runs/imports
mkdir "$received_run" &&
tar -xzf runs/checkpoints/rgb_prior_scratch80_weights_20261028.tar.gz -C "$received_run"
```

接收目录若已存在，选择另一个新名字并同步修改后续`received_run`，不直接覆盖已导入权重。

## 4. 核对数据划分和本地FMB

```zsh
cmp runs/learned_color/data_seed42_v1/cycle_manifest.json \
  "$received_run/runs/learned_color/data_seed42_v1/cycle_manifest.json" &&
shasum -a 256 -c "$received_run/runs/exports/rgb_prior_scratch80_fmb_20261028.sha256"
```

`cmp`成功时无输出；FMB应为OK。若不一致，先核对，不能重新划分数据或替换模型来绕过。使用本地已有清单是因为相对图像路径基于原清单目录；不要把嵌在接收目录中的清单直接当数据入口，否则它会去错误位置寻找FLIR图像。

本地目前已有原FMB和完整数据清单／图像，因此包内只需携带FMB的SHA，不重复传大文件。实际测试也会严格检查模型／先验／条件签名。

## 5. 使用已经准备好的本地CPU环境

本轮隔离环境已具备Torch2.6.0、Torchvision0.21.0、OmegaConf、SciPy／skimage及HTML依赖dominate。源码新增支持：`--test-only --device cpu`、`--checkpoints-dir`和测试专用`--limit`。训练默认行为与模型／先验签名不变；不要把新源码覆盖正在跑的Windows训练。

```zsh
source runs/D/review_env_20261028/bin/activate
```

CPU避开旧test.py自动选择MPS的问题。它会较慢，先少量预览确认，再导出完整集合。本轮actual CPU检查使用随机完整ResNet9／ngf64生成器、真实冻结FMB，实际通过launcher和test.py生成1张图及HTML；不是服务器80轮结果，也不是质量验证。入口／恢复13项及D原45项检查通过。

## 6. 先看80轮的8张验证图

保持原实验条件：`joint512`、learned、Q翻转一致性、语义门控、颜色损失签名0.01／最低Q0.5／预热1000。这里不能把颜色损失权重改成0，否则检查点签名改变；测试本身不会反传或更新权重。

```zsh
python -u scripts/run_learned_color.py --test-only \
  --device cpu --phase val --epoch 80 --limit 8 \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior "$received_run/runs/learned_color/rgb_prior_seed42_loss_v1.json" \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --checkpoints-dir "$received_run/runs/checkpoints" \
  --name rgb_prior_scratch80_loss001_seed42_v1 \
  --mode learned --contour-variant joint512 --gate-semantic \
  --color-loss-weight 0.01 --color-loss-min-quality 0.5 --color-loss-warmup-steps 1000 \
  --results-dir runs/results/rgb_prior_scratch80_preview8_20261028
```

网页：`runs/results/rgb_prior_scratch80_preview8_20261028/rgb_prior_scratch80_loss001_seed42_v1/val_80/index.html`。完整分辨率PNG在同目录`images/`下；HTML缩略图宽度不代表输出PNG被缩到该尺寸。重点看`real_A`（此接口下是TIR）、`fake_B`（目标RGB）、`contour_joint`、`semantic_quality`。`color_hint`是展示用提示，不是生成结果；独立RGB参考并非该TIR的真值。

8张仅用于入口和视觉排错，不拿来计算完整分布指标或宣布80轮最佳。重复运行须更换结果根目录，程序保护已有导出。

## 7. 正式比较40／60／80轮验证结果

去掉limit会导出完整val（当前690张TIR／每个检查点）：

```zsh
for epoch in 40 60 80; do
  python -u scripts/run_learned_color.py --test-only \
    --device cpu --phase val --epoch "$epoch" \
    --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
    --prior "$received_run/runs/learned_color/rgb_prior_seed42_loss_v1.json" \
    --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
    --checkpoints-dir "$received_run/runs/checkpoints" \
    --name rgb_prior_scratch80_loss001_seed42_v1 \
    --mode learned --contour-variant joint512 --gate-semantic \
    --color-loss-weight 0.01 --color-loss-min-quality 0.5 --color-loss-warmup-steps 1000 \
    --results-dir runs/results/rgb_prior_scratch80_validation_20261028 || break
done
```

三个网页分别为对应结果根目录下`<name>/val_40/index.html`、`val_60/index.html`、`val_80/index.html`。CPU完整导出量较大，可以先每轮相同8张排错，再决定完整导出；少量图的视觉优劣仍不代表整体验证结论。

## 8. 固定权重后导出完整112张test

以下以验证后选中80轮为例；若选40／60，改`--epoch`。不要用test反复选择检查点。

```zsh
python -u scripts/run_learned_color.py --test-only \
  --device cpu --phase test --epoch 80 \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior "$received_run/runs/learned_color/rgb_prior_seed42_loss_v1.json" \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --checkpoints-dir "$received_run/runs/checkpoints" \
  --name rgb_prior_scratch80_loss001_seed42_v1 \
  --mode learned --contour-variant joint512 --gate-semantic \
  --color-loss-weight 0.01 --color-loss-min-quality 0.5 --color-loss-warmup-steps 1000 \
  --results-dir runs/results/rgb_prior_scratch80_test_20261028
```

网页在`runs/results/rgb_prior_scratch80_test_20261028/rgb_prior_scratch80_loss001_seed42_v1/test_80/index.html`；生成RGB为`images/*_fake_B.png`。后续如算FID/KID，使用完整112张生成图与同协议696张RGB参考及统一评估环境，不计算无配对逐图PSNR／SSIM当准确率。

本轮没有执行上述服务器打包、下载或80轮实际测试；需要用户先回传包。收到后再核对真实权重、训练签名与生成效果。
