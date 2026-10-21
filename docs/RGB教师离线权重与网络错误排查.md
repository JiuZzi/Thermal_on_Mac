# RGB教师下载失败：离线包与继续缓存

日期：2026-10-21。服务器在`huggingface_hub.snapshot_download`阶段遇到`ConnectError [Errno 99] Cannot assign requested address`，随后报告本地snapshot不存在。这个日志证明Hub连接失败且无可用本地权重，不能单凭它确定是代理、IPv6、容器网络或其他原因。尚未进入图像语义缓存；无需因为该报错重新训练GAN或重建数据划分。

已从NVIDIA官方模型仓库固定commit `d633b2072669ca68d8f8e309de9b52bfdbf6bf72`下载config、预处理、PyTorch权重和README。官方权重SHA256为`584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe`，本地一致。外部文件均作为模型数据读取，不执行远程自定义代码。

## 上传、校验、解压

将`runs/exports/segformer_rgb_teacher_offline_20261021.tar.gz`和同名`.sha256`手工上传到服务器项目根目录。它只包含模型文件与来源校验，不替换GAN训练源码或现有模型。按顺序执行：

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512
mkdir -p runs/training
sha256sum -c segformer_rgb_teacher_offline_20261021.tar.gz.sha256 &&
tar -xzf segformer_rgb_teacher_offline_20261021.tar.gz -C .
```

确认旧缓存任务已经退出，再启动新任务：

```bash
pgrep -af '[l]earn_color_prior.py cache'
```

没有输出表示未发现该缓存进程；有输出时先确认旧进程状态，避免叠加运行。包内`SHA256SUMS`还可用于逐文件校验。

## 直接从本地模型目录缓存

当前脚本已经支持本地`--teacher`目录，无需改源码。不传`--allow-download`，并对该进程启用离线环境变量。使用新缓存目录／日志，避免与此前任务的半成品和旧报错混淆：

```bash
nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512/bin/python \
  -u scripts/learn_color_prior.py cache \
  --manifest runs/learned_color/data_seed42_v1/rgb_manifest.json \
  --teacher runs/pretrained/segformer_cityscapes_b2/d633b2072669ca68d8f8e309de9b52bfdbf6bf72 \
  --revision d633b2072669ca68d8f8e309de9b52bfdbf6bf72 \
  --device cuda:0 --flip \
  --min-probability 0.9 --min-margin 0.2 --min-agreement 0.9 --erosion 2 \
  --output-dir runs/learned_color/rgb_teacher_seed42_offline_v1 \
  > runs/training/rgb_teacher_seed42_offline_v1.log 2>&1 &
```

```bash
tail -f /home/ubuntu/work/JiuZzi/Thermal_on_Mac/runs/training/rgb_teacher_seed42_offline_v1.log
```

等待`Cached manifest:`出现，查看新输出目录的`index.html`审查伪标签，然后拟合。由于缓存目录已改名，下方manifest也使用offline目录：

```bash
python -u scripts/learn_color_prior.py fit \
  --manifest runs/learned_color/rgb_teacher_seed42_offline_v1/manifest.json \
  --config configs/learned_color/regularized_v1.json \
  --output runs/learned_color/rgb_prior_seed42_loss_v1.json
```

正式先验路径保持不变，因此后续预检、从零80轮训练和导出继续使用[完整流程步骤6及以后](从零训练学得颜色先验Ubuntu完整流程.md)。若需评价先验val密度，也将evaluate命令的manifest替换为offline缓存的manifest。此方案只处理网络下载依赖，不代表实际分割或着色质量已验证。

来源：[官方固定版本](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/tree/d633b2072669ca68d8f8e309de9b52bfdbf6bf72)、[官方权重哈希](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/blob/main/pytorch_model.bin)、[Hub离线配置](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline)。
