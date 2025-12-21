**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar  
**工作流（最重要）：** `EXECUTION_PLAN.md` 里每完成一个步骤（阶段）→ 立刻 `git commit` → `git push` 到 `origin/main`。

---

根据你提供的《思路.docx》里“思路1”的原始设想：**用 DINOv3 + LoRA 微调来减少光学预训练与 SAR 图像之间的域差异；主数据集用 RSAR（DOTA 格式，做目标检测算 mAP），如果不好做就切换到 HRSID；另外可参考 RHINO（mmdetection 里集成了 DINO detector，但不是 v3）**。

下面我给你一个**“思路1”的完整 Proposal**（尽量把你后续真正要做的工程/实验全部拆开），并且**每一阶段都明确：实现目标、验收步骤、需要注意的点**。我会把“主线方案（RSAR+DOTA/旋转框）”作为默认推进，同时给“兜底方案（HRSID/水平框快速验证）”。

---

## 0. Proposal 总览（你要交付什么，怎么判定做成了）

### 项目名称

**DINOv3 + LoRA 适配 SAR 船只检测（DOTA 旋转框）**

### 背景与动机

* DINOv3 是 Meta 发布的视觉基础模型（SSL 训练，提供强 dense features，官方代码与模型可用）。([Meta AI][1])
* 你的核心问题是：**光学域预训练 → SAR 域检测** 存在明显域差异；希望通过 **LoRA 这种参数高效微调** 在不大规模改网络/不全量微调的前提下实现“域对齐”。

### 目标定义（可验收的）

1. 在 **RSAR（DOTA/OBB）** 上跑通训练+评估闭环，输出标准 mAP（DOTA 口径）。
2. 在同一训练设定下，**DINOv3+LoRA** 的结果要能：

   * 至少不弱于强基线（如 Rotated RetinaNet / Rotated Faster R-CNN 等），并提供对比表；
   * 或者在相同/更少可训练参数量下达到更优或相当的 mAP（体现“参数高效迁移”的价值）。
3. 给出最少一份**消融**：Frozen backbone vs Full fine-tune vs LoRA（以及 LoRA 插入位置/秩等）。

### 两条实现路径（主线 + 兜底）

* **主线（推荐）**：MMRotate/MMDetection 体系（旋转框原生），DOTA 格式直接对齐，最终结果最“正宗”。
* **兜底（快速验证）**：用 `dinov3_stack` 先跑 HRSID 或把 RSAR 转成 HBB 做 quick check（它的检测管线支持 SSD/RetinaNet，RetinaNet 默认且效果更好；并支持 last/multi layer 特征抽取）。([GitHub][2])

---

## 1. 阶段1：项目初始化与“成功标准”固化

### 实现目标

* 把“你到底要证明什么”写成一页纸：

  * 任务：船只检测（OBB/DOTA）
  * 数据：RSAR（主）/HRSID（备）
  * 指标：mAP（DOTA 评估脚本/或 MMRotate 内置 DOTA mAP）
  * 对比对象：ResNet50/101-FPN 的 rotated detector baseline + DINOv3 frozen + DINOv3 LoRA
* 明确最终交付物清单：训练代码/配置、模型权重、日志、可复现实验说明、结果表、可视化。

### 验收步骤

1. 产出 `README_proposal.md`（或 notion 文档）包含：

   * 数据集来源与格式（RSAR DOTA；必要时切 HRSID）
   * 评估口径（mAP@0.5? 或 DOTA 的 mAP@0.5 等，写清楚）
   * 实验矩阵（至少 3 条：baseline / frozen / LoRA）
2. 设定随机种子策略、训练/评估脚本命名、输出目录规范（比如 `outputs/exp_name/`）。

### 需要注意的点

* **不要先改模型**。先把“评估口径”和“实验矩阵”写死，否则后面会陷入“跑了很多但不知道比较什么”。
* DOTA/OBB 指标要写清楚：DOTA 常见是 AP@0.5（也有人做多阈值），你必须选定一个并贯穿全文，否则对比会失效。

---

## 2. 阶段2：环境与代码基座搭建（主线 + 兜底都准备好）

### 实现目标

* 主线环境：MMDetection + MMRotate（用于 DOTA/旋转框）。
* 兜底环境：`dinov3_stack` + 官方 DINOv3 repo & 权重（用于快速下游验证）。

  * `dinov3_stack` 需要你在 `.env` 里配置 DINOv3 仓库路径与权重路径。([GitHub][2])

### 验收步骤

**主线验收（MMRotate）**

1. 能跑通官方一个 rotated detector 的 toy demo（随机数据/小样本），确认 CUDA、mmcv、mmengine 无版本地狱。
2. 能加载 DOTA 格式数据的 dataloader（即使先用空数据也行）。

**兜底验收（dinov3_stack）**

1. clone `dinov3_stack`，按 README 配好 `.env`：

   * `DINOv3_REPO="/abs/path/to/cloned/dinov3"`
   * `DINOv3_WEIGHTS="/abs/path/to/dinov3/weights"` ([GitHub][2])
2. 运行 `train_detection.py --help` 能打印出参数（说明环境 OK）。([GitHub][2])

### 需要注意的点

* DINOv3 官方仓库可能要求以 `PYTHONPATH=.` 的方式运行内部模块（官方 README 常见写法），确保你加载权重/模型时路径正确。([GitHub][3])
* `dinov3_stack` 的检测头支持 **SSD/RetinaNet**，且 README 说明 RetinaNet 默认且更好。([GitHub][2])
* 代码基座尽量“少改动”，后续所有实验用配置文件/参数切换，避免改出不可复现的状态。

---

## 3. 阶段3：数据集获取、格式对齐与数据体检（RSAR 主线）

### 实现目标

* 下载 RSAR 数据集并整理成你训练框架能吃的格式。你给出的信息是：

  * RSAR 链接（Baidu）
  * **DOTA 格式**
  * 做目标检测，计算 mAP

（我把原始链接放代码块里，避免明文 URL 规则问题）

```text
RSAR: https://pan.baidu.com/s/1g2NGfzf7Xgk_K9euKVjFEA?pwd=rsar
格式：DOTA
任务：目标检测，计算 mAP
```

### 验收步骤

1. **文件完整性检查**

   * 图片数量、标注数量一致
   * 每张图片都能找到对应标注（或明确允许缺标注）
2. **标注几何正确性检查（关键）**

   * 随机抽 200 张图，把 DOTA 的 8 点/旋转框画出来（保存到 `debug_vis/`）
   * 肉眼确认：框的顺序正确、没有“蝴蝶结框”、没有大面积偏移
3. **统计与分布检查**

   * 每张图 ship 实例数分布
   * 框面积/长宽比分布
   * 图像分辨率分布（决定后面切片策略）

### 需要注意的点

* DOTA 标注的点顺序/旋转框表示法非常容易踩坑：

  * 点顺序错 → IoU 计算错 → mAP 虚假偏低
* 如果 RSAR 原图很大（DOTA 常见），你大概率需要 **切片（tiling）**：

  * 切片要保证：切片后的框正确裁剪/保留；小目标不要被裁掉过多；
  * 训练和测试切片策略要一致（或明确不一致的理由）。
* SAR 常是灰度图：后面接 DINOv3（通常按 3 通道输入）时要提前决定：

  * **复制成 3 通道**（最简单），或
  * **改第一层让它吃 1 通道**（更干净，但工程更多）。

---

## 4. 阶段4：强基线（Baseline）先跑通并“锁死”结果

> 这一步是整个项目的地基：**没有强 baseline，你后面所有 DINOv3/LoRA 的结论都站不住**。

### 实现目标

* 在 RSAR(DOTA) 上训练至少 1-2 个成熟 rotated detector baseline（建议从 Rotated RetinaNet 开始，因为论文/工程生态都很成熟；而且你上传的 OFA-Net 也是基于 RetinaNet 的鲁棒检测范式）。
* 产出 baseline 的：config、权重、训练日志、验证集 mAP、测试集 mAP、可视化结果。

### 验收步骤

1. Baseline 训练能稳定收敛（loss 不爆炸、不 NaN）。
2. 评估脚本能正常跑出 mAP（不是 0，不是 NaN）。
3. 产出“可复现证据包”：

   * `config.yaml/py`
   * `metrics.json`
   * `log.txt`
   * `vis_samples/`（至少 50 张预测可视化）

### 需要注意的点

* baseline 的数据增强、输入尺度、训练轮数，要尽量合理且**固定**，后面 DINOv3/LoRA 都以此为参照。
* 如果 baseline 太弱，你的 LoRA 提升可能只是“把 pipeline 修好了”，不是真提升。
* 旋转框检测经常受 NMS/阈值影响，baseline 和后续模型推理参数必须保持一致，否则对比不公平。

---

## 5. 阶段5：DINOv3 Backbone 接入（先 Frozen，不上 LoRA）

### 实现目标

* 把检测器 backbone 换成 DINOv3，但先**冻结 backbone**，只训练 neck + head：

  * 目的：验证“DINOv3 特征能不能在 SAR 上工作”，以及你的“模型拼装/特征维度/多尺度”是否正确。
* 你可以参考两种实现方式：

  1. **ConvNeXt 系** DINOv3 backbone（天然多 stage，接 FPN 更直接）
  2. **ViT 系** DINOv3 backbone（需要把 token 特征变成多尺度特征：用多层输出 / 适配器 / FPN-like neck）

### 验收步骤

1. 模型 forward 形状检查通过（各层 feature map 尺寸与 FPN/head 输入一致）。
2. 冻结 backbone 的情况下训练能跑完，mAP 不至于崩到“明显比 baseline 低很多”（允许略低，但不能完全不可用）。
3. 记录：

   * trainable params 数量（应该显著小于全量）
   * 显存占用、吞吐（作为工程可行性证据）

### 需要注意的点

* DINOv3 是“通用 backbone”，但你要把它变成检测用多尺度特征。`dinov3_stack` 的检测脚本就提供了一个思路：

  * 支持 `--feautre-extractor {last,multi}`（取最后一层或多层做特征），并支持 SSD/RetinaNet 检测头。([GitHub][2])
* 灰度 SAR → DINOv3 输入：

  * 如果你用复制通道，注意归一化统计（不要盲目用 ImageNet mean/std；至少要对比“用 DINOv3 官方推荐预处理 vs 自己的 SAR 归一化”的差异）。
* Frozen 阶段的结果是“管线正确性验证”，不必追求最优，但必须能跑通且有合理预测。

---

## 6. 阶段6：LoRA 插入与参数高效微调（核心阶段）

### 实现目标

* 在 DINOv3 backbone 上插入 LoRA，只训练 LoRA 参数（+ 检测 head/neck），实现“域对齐”。
* 你需要明确 LoRA 设计 choices（写进 config）：

  * 插入位置：Attention 的 q/k/v/o 线性层、MLP 的两层线性层（ViT），或 ConvNeXt block 的 pointwise/linear 层（如存在）。
  * rank `r`、alpha、dropout
  * 是否训练 LayerNorm / bias（常见 trick：LN 可训练能提升稳定性）

### 验收步骤

1. **参数量验收（必须做）**

   * 打印总参数量、可训练参数量
   * 可训练参数占比应明显下降（例如 <10%，最好 <5%）——这才叫“LoRA”
2. **功能验收**

   * LoRA on 后训练能跑完，不 NaN
   * 与 Frozen 版本相比，验证集 mAP 有稳定提升（哪怕小幅提升也算“对齐有效”）
3. **对比验收**

   * baseline（ResNet） vs frozen DINOv3 vs LoRA DINOv3
   * 同一评估脚本、同一 NMS、同一阈值

### 需要注意的点

* LoRA learning rate 通常要比 head 小心：

  * 太大容易把预训练表示“扯坏”；太小又不动。
  * 推荐做 **分组学习率**：`lr_head > lr_lora` 或相反都要试，但要固定实验矩阵。
* 如果你用 ViT backbone：

  * 多尺度特征是关键瓶颈，LoRA 再强，多尺度做错了也没用。
* 你必须做一个“full fine-tune（小轮数/小学习率）”作对照，否则别人会问：

  * “是不是全量微调更好？”
  * 你的回答必须用数据支撑（哪怕是“全量更好但成本更高；LoRA 更划算”）。

---

## 7. 阶段7：训练策略与数据增强（专门为 SAR 域服务）

> 这一步是“让 LoRA 真正值钱”的地方：很多时候域差异不是靠 LoRA 一刀切就能解决，SAR 的成像特性决定你要做一些特化增强/预处理。

### 实现目标

* 建立一套**可控**的 SAR 专用增强组合，并做消融：

  * Speckle noise 模拟
  * 对比度/动态范围变化（log/γ）
  * 模糊（聚焦变化）
  * 轻微几何增强（但要保证旋转框同步变换正确）
* 目标不是“增强越多越好”，而是找到 **对 mAP 和泛化最稳定** 的组合。

### 验收步骤

1. 每个增强组合都跑至少一个短实验（相同 seed/epoch/配置），记录 mAP。
2. 输出一个消融表：augmentation set → mAP/Recall/Precision。
3. 输出增强可视化样例（增强后图像 + 目标框，确认标注仍对齐）。

### 需要注意的点

* 旋转框增强最容易出错的是“旋转/裁剪后框变形”，必须做可视化验收。
* SAR 的强噪声/强对比变化可能导致 detector 学到“背景纹理”，需要关注 false positive 的类型（港口结构、条纹、海杂波）。

---

## 8. 阶段8：鲁棒性评估（可选但强烈建议，尤其你关心干扰）

虽然这是你“思路2”的主战场，但你现在做的是“思路1”，也可以把它作为**扩展验收项**：看看 LoRA 域对齐是否也能提升对干扰的鲁棒性。

你上传的 OFA-Net 论文明确指出：**频移干扰幅度增大时，检测精度与置信度下降，false positives 上升，甚至干扰条纹会被误检成船**。

### 实现目标

* 在“无干扰 → 有干扰”的条件下评估你的模型（baseline / frozen / LoRA）：

  * mAP 随干扰强度变化曲线
  * FP 类型统计（条纹误检、岸线误检等）

### 验收步骤

1. 有一套可复现的干扰注入脚本（如果你没有真实干扰数据，也可以先做合成干扰）。
2. 输出表格：干扰强度 → mAP / FP 数量 / Recall。
3. 产出对比可视化（同一张图：baseline vs LoRA 的预测框差异）。

### 需要注意的点

* 干扰注入一定要可控（强度参数、频率、方向），否则你解释不了结论。
* 如果发现 LoRA 对干扰帮助有限，这是正常的：论文也强调“干扰会严重破坏 SAR 表征质量，需要专门理解干扰信号”。

---

## 9. 阶段9：结果整理、误差分析与最终交付（论文/报告级）

### 实现目标

* 把结果从“我跑出来了”升级到“我能解释为什么这样”：

  * 典型成功案例/失败案例
  * 按场景分组：近岸/远海、密集/稀疏、小船/大船
  * 分析 LoRA 改善的是哪类样本（或恶化了哪类）

### 验收步骤

1. 输出最终结果表：

   * baseline（至少 1-2 个）
   * DINOv3 frozen
   * DINOv3 + LoRA（主结果）
     -（可选）full fine-tune
2. 输出可视化册：

   * Top-50 正例
   * Top-50 假阳性
   * Top-50 漏检
3. 输出可复现包：

   * 一键训练脚本（run.sh）
   * 一键评估脚本
   * 环境导出（requirements 或 conda env）
   * 权重文件命名规范 + 模型卡

### 需要注意的点

* 复现性是最容易被忽略的“验收项”。你必须能回答：

  * “别人拿你的 repo 怎么复现？”
* 误差分析不要只贴图，要归因：

  * 是阈值问题？NMS？多尺度不足？还是 SAR 纹理导致？

---

# 兜底路线（当 RSAR/DOTA 卡住时：先用 HRSID 快速验证“DINOv3+LoRA 有用”）

你原文写了：如果 DINO 的检测不好做，就切换 HRSID。
我建议把它当作“工程风险兜底”，不是替代主线。

### 兜底阶段A：HRSID 数据准备

**实现目标**：把 HRSID 整成 COCO/VOC/HBB 格式，先跑通水平框检测。
（原链接放代码块）

```text
HRSID: https://github.com/Ycc1999/modified-HRSID
```

**验收步骤**

* 画框可视化检查
* 训练一个 HBB baseline（RetinaNet/FCOS）跑通 mAP

**注意点**

* HRSID 与 RSAR 标注口径可能不同，兜底结果只能用于验证“迁移是否有效”，不能直接替代 RSAR(DOTA) 主结果。

### 兜底阶段B：用 dinov3_stack 做快速检测验证

`dinov3_stack` 的 README 明确：

* 支持 SSD/RetinaNet 检测头，RetinaNet 默认更好；
* `train_detection.py` 提供训练示例命令，参数包括 `--head {ssd,retinanet}`、`--feautre-extractor {last,multi}`、`--fine-tune` 等。([GitHub][2])

**实现目标**：用同一份 HRSID(HBB) 数据对比：

* 普通 backbone（比如 torchvision resnet）baseline
* DINOv3 frozen
* DINOv3 + LoRA（你需要自己加 LoRA，但工程量相对可控）

**验收步骤**

* 训练脚本能跑完并出 mAP
* 结果趋势符合预期（LoRA ≥ frozen）

**注意点**

* dinov3_stack 的评估口径可能是 VOC/COCO 风格，与你 RSAR DOTA mAP 口径不同；这条路的意义是“证明方法值得继续”，不是最终论文结果。

---

## 参考与借鉴（你文档里提到的 RHINO）

你也写了：RHINO 是一个把 DINO 作为 detector 集成到 mmdetection 的参考项目，但不是 v3。
**它在“工程结构”上很有参考价值**：如何在 OpenMMLab 体系里接 transformer/特征/neck/head。([GitHub][4])

---

# 最终你会得到的“完整交付物清单”（用于最终验收）

1. **数据处理**

* RSAR 原始数据 → 训练可用格式（含切片脚本/说明）
* 数据体检报告（统计 + 可视化）

2. **模型与实验**

* Baseline rotated detector 配置/权重/日志
* DINOv3 frozen 配置/权重/日志
* DINOv3 + LoRA 配置/权重/日志
  -（可选）full fine-tune 配置/权重/日志
* 消融表（LoRA rank/位置/可训练 LN 与否）

3. **结果与分析**

* mAP 表格（统一口径）
* 失败案例分析（FP/FN 分类）
  -（可选）干扰鲁棒性曲线与可视化对比

4. **复现包**

* 一键训练/评估脚本
* 环境文件
* 说明文档（让别人照着跑）

---

如果你希望我把这个 Proposal 进一步“落到可执行的配置级别”（例如：**你选哪个 DINOv3 backbone、MMRotate 用哪个 detector、LoRA 具体插哪些层、默认 lr/epoch/augment 组合怎么设**），我也可以直接给你一套“默认推荐配置 + 消融矩阵”，你拿去就能开跑。

[1]: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/?utm_source=chatgpt.com "DINOv3: Self-supervised learning for vision at ..."
[2]: https://github.com/sovit-123/dinov3_stack "GitHub - sovit-123/dinov3_stack: A repository to apply DINOv3 models for different downstream tasks: image classification, semantic segmentation, object detection."
[3]: https://github.com/facebookresearch/dinov3?utm_source=chatgpt.com "Reference PyTorch implementation and models for DINOv3"
[4]: https://github.com/SIAnalytics/RHINO?utm_source=chatgpt.com "SIAnalytics/RHINO"
