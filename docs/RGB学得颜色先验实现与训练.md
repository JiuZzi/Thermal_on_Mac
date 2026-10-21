# RGB学得颜色先验第一版：实现与训练

日期：2026-10-21。独立模型入口为 `learned_color_cycle_gan`。原来的轮廓版、仅语义版和人工颜色版源码没有改动。

本版实现的是 **RGB训练区域 → 小容量多峰Lab色度分布 → 冻结颜色模式 → 生成器条件分支**。GMM用EM从真实RGB拟合均值、对角方差和频率，不是人工色表，也不是另外预训练一个大型颜色神经网络。生成器继续通过原GAN和cycle目标学习生成RGB。

2026-10-21补充：用户决定试跑**微量学得颜色损失**。当前入口默认权重0.01，原条件输入对照请显式设置`--color-loss-weight 0`。新增损失定义、需要重新拟合的支持阈值和完整启动命令见[微量颜色损失Ubuntu训练](微量颜色损失Ubuntu训练.md)。下方旧两组命令已显式关闭损失，保留原对照含义。

## 当前完成与证据范围

- 已实现通用分组清单、RGB教师缓存、RGB先验拟合与独立评价、模型接入、检查点身份约束、分组GAN数据加载和启动器。
- RGB先验的拟合函数只读取train/val角色，val用于选择小模型；不读取test色样。数据加载器会拒绝跨角色的视频／场景、重复图像，以及先验训练区域进入GAN验证集的情况。
- 模式数只从预先声明的1/2/3/5中选择；按验证组平均NLL和一标准误差规则偏向较小模型；限制每图／每组样本量、方差下限，输出按组bootstrap稳定性和训练—验证差距。当前fit还记录组平衡训练样本NLL的99%分位作为颜色损失支持阈值，不使用val/test拟合该阈值。
- 真实RGB的既有67块稀疏复核区域已用于**开发检查**，得到道路2模式、植被1模式、天空1模式；人行道因独立组不足被标为不支持。这份开发先验不打入生产更新包，不能替代全量RGB来源审查。
- 真实完整FMB权重、旧第80轮G_A/G_B、resnet9/ngf64、128裁剪及两步更新通过：新增分支初始输出最大差0、投影和编码器梯度非零、语义模型保持冻结。检查使用小型pixel判别器，只验证实现。[真实检查记录](../runs/smoke/learned_color_20261021/real_weight_two_steps/smoke_report.json)
- **本机没有下载或执行新的RGB教师真实权重，也没有完成全量RGB缓存、新版GPU正式训练或外部数据集评价。**RGB教师适配器已用替身模型检查坐标恢复、冻结状态、类别映射、腐蚀和test默认跳过，不能据此声称真实分割质量。

这些防护不构成“不会过拟合”的保证。模型容量小、验证损失低，都不能代替跨数据集验证。

## 与数据集无关的边界

核心算法不含FLIR图号、360×288限制、棕榈树个例、手工RGB色样或固定测试数量。CSV列名、源划分值、图像根目录、类别名称均由参数提供；像素标签采用清单中的0..C-1，255忽略。RGB教师的类别按名称对齐。

当前TIR主干仍复用既有FMB 15类语义模型，训练数据的语义／相机域差异依然存在。更换数据集时可以复用代码、固定先验评测，也可以明确重新拟合；后者必须单列为目标域适配，不能称为未经适配的跨域泛化。

## 语义可靠度与结构

Q是固定的启发式：类别分差、熵、可选翻转预测一致性、类别交界附近减弱。它不是经过校准的正确率，也不是未来补边的置信度C。一致但错误的预测仍可能通过，单元检查明确保留这个反例。

Q作用在新增颜色残差的最终注入处，避免编码器bias绕过门控；支持类别之外关闭新增颜色残差。多种颜色模式分别编码，不压成单一均值。显示用颜色提示采用固定L*=65和混合均值，**只作诊断展示，未作为模型输入**。

`--gate-semantic`还可对旧15类语义分支贡献乘Q；需要两个训练对照同时开启或关闭。默认只对新增颜色分支门控。启用旧语义门控会改变已训练语义分支的贡献，因此不保证初始输出与原仅语义模型逐像素相同。原始TIR、原joint512及第三零通道继续保留；本轮没有训练语义修正头或实施补边，不能承诺修好高分误判的叶片。

## Ubuntu操作

代码更新继续采用tar手工上传。更新包不包含RGB教师、开发先验、数据、已有GAN权重或登录信息。

```bash
cd /home/ubuntu/work/JiuZzi/Thermal_on_Mac
conda activate /home/ubuntu/work/JiuZzi/conda_envs/c1_sem512
sha256sum -c learned_color_ubuntu_20261021.tar.gz.sha256 &&
tar -xzf learned_color_ubuntu_20261021.tar.gz -C .
python scripts/package_learned_color_ubuntu.py --verify-installed
```

### 1. 先固定分组

以下是FLIR的**数据适配命令**，不是算法里的特殊规则。换数据集时替换CSV、列名、根目录和源角色值。CSV必须提供真正的视频／场景组，不要把每帧当成独立组。

```bash
python scripts/prepare_grouped_color_data.py \
  --csv FLIR_protocol_v2/manifest.csv --image-root FLIR_datasets \
  --image-column destination_name --group-column video_id \
  --role-column destination_split --role-is-subdir \
  --train-a-role trainA --train-b-role trainB \
  --test-a-role testA --test-b-role testB \
  --dataset-id flir_v2 --classes road sidewalk vegetation sky \
  --output-dir runs/learned_color/data_seed42_v1
```

一次生成共享分组的 `cycle_manifest.json` 与 `rgb_manifest.json`。本地协议得到RGB train5531/val660/test696，TIR train1420/val690/test112。分组长度不一，因此20%组不等于20%图像；不能为得到好看的结果不断换seed。路径为清单相对路径，可随同项目迁移。

### 2. 冻结RGB教师并缓存

只在离线RGB阶段需要Transformers；GMM和GAN不导入它。先在当前环境安装可选依赖，使用RGB道路场景模型候选：

```bash
python -m pip install -r scripts/requirements_learned_color_teacher.txt
python -u scripts/learn_color_prior.py cache \
  --manifest runs/learned_color/data_seed42_v1/rgb_manifest.json \
  --teacher nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
  --revision main --allow-download --device cuda:0 --flip \
  --output-dir runs/learned_color/rgb_teacher_seed42_v1
```

首次`main`会解析为一个具体snapshot，预处理与权重从同一snapshot加载，记录其身份、张量哈希和软件版本。复现时把revision换成缓存manifest中保存的 `teacher.resolved_commit`。不加 `--allow-download` 就只使用已有本地缓存；不执行远程自定义代码。

默认只缓存train/val，跳过test；外部RGB先验评价需要显式 `--include-test`。原坐标响应恢复后，使用预声明的响应／分差／翻转一致性和内部腐蚀取区域。高分仍可能错，须看输出 `index.html` 的按组预览：原RGB／类别预测／保留区域。显示色不参与先验学习。

### 3. 只从RGB训练区域拟合

```bash
python -u scripts/learn_color_prior.py fit \
  --manifest runs/learned_color/rgb_teacher_seed42_v1/manifest.json \
  --config configs/learned_color/regularized_v1.json \
  --output runs/learned_color/rgb_prior_seed42_v1.json
```

先查看每类的独立组支持、验证NLL、训练—验证差距与bootstrap结果；数据不足的类保持“不支持”。不要把原先67块开发种子拟合的先验当作正式产物，也不要按test结果改颜色参数。

### 4. 同起点、同预算的两个训练对照

下面两组都从服务器已有**仅语义第5轮**生成器开始；判别器和优化器重新初始化。每组新增5轮是开发试验，正式训练预算需由验证观察决定。

`zero_tokens`保留相同分支容量、Q和训练配置，把颜色模式输入置零；不是删除整个分支。下面两组显式关闭新增颜色损失，日志中的`color_prior=0`符合此对照设计。

```bash
python -u scripts/run_learned_color.py --train \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --init-condition-dir runs/checkpoints/flir_v2_semonly_from80_seed42_20261020_preview5 \
  --name rgb_prior_zero_tokens_seed42_v1 --mode zero_tokens --gate-semantic --epochs 5 --color-loss-weight 0

python -u scripts/run_learned_color.py --train \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --init-condition-dir runs/checkpoints/flir_v2_semonly_from80_seed42_20261020_preview5 \
  --name rgb_prior_learned_seed42_v1 --mode learned --gate-semantic --epochs 5 --color-loss-weight 0
```

启动前可以将 `--train` 改为 `--check-only`：检查分组、重复图像、先验与验证集交叉、源文件存在等；该检查不加载完整语义权重、不检查训练质量。旧检查点原先见过新划出的trainA/trainB验证视频，这个事实无法撤回；上述val只对新阶段和新先验划分有效。整个系统严格未见验证需重新训练或另用完全未见的视频。

翻转Q增加冻结语义推理开销；`--no-reliability-flip`可关闭，但两个对照必须一致，而且改变该设置需要新实验身份。模型默认启用颜色条件dropout与新分支权重衰减，不能据此保证泛化。

### 5. 先导出验证集

训练不会自动测试。按保存的epoch导出val；数量由清单中的TIR记录决定，没有112的硬编码。

```bash
python -u scripts/run_learned_color.py --test-only --phase val --epoch 5 \
  --data-manifest runs/learned_color/data_seed42_v1/cycle_manifest.json \
  --prior runs/learned_color/rgb_prior_seed42_v1.json \
  --semantic-checkpoint runs/pretrained/segformer_fmb_b3/best_model.pth \
  --name rgb_prior_learned_seed42_v1 --mode learned --gate-semantic --color-loss-weight 0
```

另一个控制组使用相应name和mode导出。需要检查颜色条件是否被用到时，保持训练mode不变，加 `--feature-override off` 或 `shuffle`，并指定新的 `--results-dir`，防止覆盖原导出。off关闭新增颜色残差，shuffle平移颜色token，仅作依赖性诊断，不能单独证明颜色正确。

## 外部评价与剩余工作

- 在独立RGB清单上用 `learn_color_prior.py evaluate --prior ... --manifest ... --role test --output ...` 评价冻结先验。该清单需用同一规范类名、完成RGB标签缓存；脚本拒绝拟合／选择数据冒充外部测试。密度评价只说明RGB先验，不证明TIR生成质量。
- 外部TIR着色评价需固定先验、生成器和所有阈值，再提供另一份兼容的双域分组清单导出。新数据上重新拟合或调阈值应记录为适配。
- 另行验证叶片／天空串色、人体服装斑驳、结构损伤和自然度。不能让同一个伪标签模型独自给自己打正确性分数。
- 新增微量损失约束每个可靠像素的GMM颜色支持范围；尚未实现整区域颜色直方图匹配、监督式TIR语义修正、补边置信度C和GAN自动早停。现有启动器保存每轮检查点并支持val导出；先验的验证选择已实现，不等于GAN自动早停已实现。

方法依据：[SegFormer官方接口](https://huggingface.co/docs/transformers/model_doc/segformer)、[RGB候选模型](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024)、[多种颜色可能性的建模](https://richzhang.github.io/colorization/)。本文的组平衡、多峰token与Q接入方式属于本项目方案，尚无实际着色增益或跨域结论。
