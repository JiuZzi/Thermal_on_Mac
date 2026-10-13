# 来源与兼容范围

来源：[csmty/InfraredIR](https://github.com/csmty/InfraredIR)，固定提交
`060a3c2d6b0407f1af1429efbbb643887fbcce64`。源文件路径、Git blob 哈希和修改说明保存在 `upstream.json`。

- MiT 编码器保留作者文件及参数名；仅将 timm 的三个工具函数替换为 PyTorch 实现。
- 分割头按作者的 MLP、特征拼接顺序、两处插值参数及 ConvModule 操作实现。
- 作者分割头采用 SyncBatchNorm；本实现采用相同 eps/状态键的 BatchNorm2d，仅支持冻结 eval 推理。这不是训练版替代。
- 完整权重严格加载，包括作者 forward 未使用的 classifier；不自动下载或随机补齐缺失参数。
- 灰度复制、RGB 通道均值/标准差、短边至少 512 的放大和 logits 恢复顺序均遵循该提交的 `test.py` / `imutils.py`。
- 512 基线保持单次前向。可选较大尺寸/翻转是额外诊断，分开导出，不能称作者原推理结果。

编码器与分割头源文件带有 NVIDIA 2021 版权声明，许可证完整保存在 `_vendor/LICENSE_NVIDIA`。
原始 SegFormer 来源：[NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)。本目录服务于研究预览。

作者提供的 `best_model.pth` 含优化器及配置对象，因此只读加载采用有限元数据类型白名单与
`weights_only=True`，不切换到任意 pickle 反序列化。OmegaConf 仅用于这些元数据。

这个模型是外部数据训练的语义先验，不是本项目无标签 FLIR 上新训练的贡献。
权重内部迭代数和 best_miou 是原训练记录，不是 FLIR 预览指标。
