# 执行计划（强制：每步完成都要 push）

**GitHub 仓库（唯一远端）：** https://github.com/shatianming5/dino_sar  
**分支：** `main`（默认）  
**强制规则：** 下面每个“阶段/步骤”完成后，必须执行一次 `commit + push`（同一步骤内可多次小提交，但结束时至少 1 次 push）。

```bash
git status
git add -A
git commit -m "stageX: <本阶段一句话总结>"
git push origin main
```

---

## 0. 对 `plan.md` 的完整结构解析（你要做的事都在这里）

`plan.md` 的主线是：**RSAR（DOTA/OBB）上跑通 rotated detector 基线 → 接入 DINOv3 backbone（Frozen）→ 插入 LoRA 做参数高效微调 → 做 SAR 专用增强/可选干扰鲁棒性 → 整理结果与复现包**。

对应章节如下：

1. **阶段1：项目初始化与成功标准固化**（先写清楚“证明什么/怎么验收/怎么复现”）
2. **阶段2：环境与代码基座**（主线 MMRotate；兜底 dinov3_stack）
3. **阶段3：数据集获取与体检**（RSAR DOTA；几何/统计/可视化）
4. **阶段4：强基线**（Rotated RetinaNet 等，锁死对比口径）
5. **阶段5：DINOv3 接入（Frozen）**（先验证特征与多尺度拼装正确）
6. **阶段6：LoRA 核心阶段**（插入位置/秩/可训练参数比例/对比实验）
7. **阶段7：SAR 专用训练策略与增强**（speckle/对比度/log/γ 等消融）
8. **阶段8：鲁棒性评估（可选）**（干扰注入→mAP 曲线/FP 类型）
9. **阶段9：结果整理与最终交付**（表格/误差分析/一键脚本/环境文件）
10. **兜底路线：HRSID 快速验证**（当 RSAR/DOTA 工程卡住时启用）

---

## 1. 可执行步骤（每步结束必须 push）

下面把 `plan.md` 的“阶段”落成你每天可以照着做的 checklist；默认以 **主线（RSAR+DOTA/OBB）** 推进，兜底作为风险控制。

### Step 1（对应 plan 阶段1）：固化成功标准 + 交付物

**目标**
- 明确任务/数据/指标/对比矩阵（baseline / DINOv3 frozen / DINOv3+LoRA / 可选 full fine-tune）

**产出物（仓库内）**
- `README_proposal.md`：一页纸写清楚成功标准、评估口径、实验矩阵、目录规范、复现方式
- `scripts/`（可先留空占位）：后续一键训练/评估脚本统一放这里
- `outputs/` 只用于本地输出（已被 `.gitignore` 忽略）

**验收**
- 任何人打开 `README_proposal.md` 就知道：跑哪个命令、看哪个指标、怎么对比、结果放哪

**结束必须 push**
- Commit message 建议：`stage1: freeze success criteria and deliverables`

---

### Step 2（对应 plan 阶段2）：环境与代码基座（主线 + 兜底）

**目标**
- 主线：MMDetection + MMRotate 能启动、能加载 DOTA 数据
- 兜底：`dinov3_stack` 能跑 `train_detection.py --help`

**产出物（仓库内）**
- `env/` 或 `requirements*.txt`/`environment.yml`（二选一，按你实际习惯）
- `docs/ENV_SETUP.md`：记录版本组合（cuda/torch/mmcv/mmengine/mmrotate）、安装命令、踩坑

**验收**
- 主线：跑通 rotated detector 的最小 demo（哪怕用极小样本/假数据）
- 兜底：能成功导入 DINOv3 权重（路径写到本机 `.env`，不要提交）

**结束必须 push**
- Commit message 建议：`stage2: environment baseline ready (mmrotate + fallback)`

---

### Step 3（对应 plan 阶段3）：RSAR 数据体检与格式对齐

**目标**
- 确认 RSAR 的 DOTA 标注几何正确；必要时制定切片策略

**产出物（仓库内）**
- `tools/data_audit/`：数据体检脚本（数量核对、统计、可视化）
- `docs/DATA.md`：数据来源、目录约定、DOTA 口径说明、切片策略说明

**验收**
- 随机抽样可视化（例如 200 张）确认旋转框无“蝴蝶结/错序/偏移”
- 统计报表生成成功（实例数/面积/长宽比/分辨率）

**结束必须 push**
- Commit message 建议：`stage3: rsar data audit and dota alignment`

---

### Step 4（对应 plan 阶段4）：强 Baseline 锁死

**目标**
- 至少 1-2 个成熟 rotated detector baseline 在 RSAR(DOTA) 上跑通并锁死配置

**产出物（仓库内）**
- `configs/baselines/`：baseline 配置
- `scripts/smoke_train_baseline.sh`：20 iter 冒烟训练（先验证链路）
- `scripts/train_baseline.sh`、`scripts/eval_baseline.sh`：正式训练/评估一键脚本
- `docs/RESULTS.md`：baseline 结果表（统一评估口径）

**验收**
- 先跑通冒烟训练：`configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_smoke.py`
- loss 收敛、评估 mAP 非 0 且稳定
- 保存最少 50 张预测可视化（本地；不要提交大图）

**结束必须 push**
- Commit message 建议：`stage4: baseline rotated detector locked`

---

### Step 5（对应 plan 阶段5）：DINOv3 Backbone 接入（Frozen）

**目标**
- 用 DINOv3 替换 backbone；先冻结 backbone，仅训练 neck/head，验证拼装正确

**产出物（仓库内）**
- `models/`：DINOv3 backbone 适配器（多尺度输出/neck 对接）
- `configs/dinov3_frozen/`：冻结版本配置
- `docs/RESULTS.md`：追加 frozen 对比行

**验收**
- 形状/多尺度对齐正确；训练能跑完；mAP 合理（不要求最优）
- 记录可训练参数量与显存占用

**结束必须 push**
- Commit message 建议：`stage5: dinov3 backbone integrated (frozen)`

---

### Step 6（对应 plan 阶段6）：LoRA 参数高效微调（核心）

**目标**
- 在 DINOv3 上插入 LoRA（明确插入层/秩/alpha/dropout）；验证“可训练参数占比显著下降且 mAP 提升”

**产出物（仓库内）**
- `models/lora/`：LoRA 注入实现
- `configs/dinov3_lora/`：LoRA 配置（含可训练参数统计日志输出）
- `docs/RESULTS.md`：追加 LoRA 主结果 + 消融矩阵入口

**验收**
- 可训练参数占比明显下降（例如 <10%，最好 <5%）
- LoRA ≥ Frozen（同口径对比）
- 可选：full fine-tune 小学习率对照

**结束必须 push**
- Commit message 建议：`stage6: lora fine-tuning for sar domain alignment`

---

### Step 7（对应 plan 阶段7）：SAR 专用增强与训练策略消融

**目标**
- 引入 speckle/对比度/log/γ 等增强，做可控消融，找到稳定组合

**产出物（仓库内）**
- `configs/aug_ablation/`：增强组合配置
- `docs/RESULTS.md`：增强消融表

**验收**
- 每个增强组合至少一个短跑实验；增强可视化确认框对齐

**结束必须 push**
- Commit message 建议：`stage7: sar augmentation ablations`

---

### Step 8（对应 plan 阶段8，可选）：干扰鲁棒性评估

**目标**
- 干扰注入（合成或真实）→ 输出 mAP-干扰强度曲线 + FP 类型统计

**产出物（仓库内）**
- `tools/jamming/`：干扰注入脚本
- `docs/ROBUSTNESS.md`：曲线与分析方法

**验收**
- 干扰参数可控、复现实验可跑；对比 baseline/frozen/LoRA 的趋势

**结束必须 push**
- Commit message 建议：`stage8: robustness eval under jamming`

---

### Step 9（对应 plan 阶段9）：最终结果整理与复现包

**目标**
- 把“能跑”升级为“可复现 + 可解释”：结果表、误差分析、一键脚本、环境文件

**产出物（仓库内）**
- `scripts/`：一键训练/评估/可视化脚本齐全
- `docs/RESULTS.md`：最终对比表 + 误差分析入口
- `docs/REPRODUCE.md`：从零复现说明

**验收**
- 新环境按 `docs/REPRODUCE.md` 能复现实验并得到同口径指标（允许小波动）

**结束必须 push**
- Commit message 建议：`stage9: final results, analysis, and reproducibility`

---

### Step 10（扩展实验）：LoRA r=16 更长 schedule（5k iters）

**目标**
- 在当前最优 LoRA r=16 的基础上延长训练（5k iters），验证 mAP 是否继续提升

**产出物（仓库内）**
- `configs/dinov3_lora/*_r16_5kiter.py`：更长 schedule 配置
- `scripts/reproduce_lora_r16_5k.sh`：一键复现脚本
- `docs/RESULTS.md`：追加 `E3-5k-r16` 的 mAP

**验收**
- 训练完成保存 `iter_5000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage10: lora r16 longer schedule (5k)`

---

### Step 11（扩展实验）：5k iters，但保持 2k 的早学习率衰减

**动机**
- Step10 将学习率衰减按迭代数比例后移（`step=[4000,4750]`）导致性能明显下降；这里验证“延长训练但不延后衰减”（即更长的低 LR 尾训练）是否更稳。

**产出物（仓库内）**
- `configs/dinov3_lora/*_r16_5kiter_early_decay.py`：早衰减的 5k 配置
- `scripts/reproduce_lora_r16_5k_early_decay.sh`：一键脚本
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 保存 `iter_5000.pth` 并完成 RSAR val mAP 评估

**结束必须 push**
- Commit message 建议：`stage11: lora r16 5k early lr decay`

---

### Step 12（扩展实验）：LoRA rank=32 消融（2k iters）

**目标**
- 补全 LoRA rank ablation（r=8/16/32），观察是否继续提升

**产出物（仓库内）**
- `configs/dinov3_lora/*_r32_2kiter.py`
- `scripts/reproduce_lora_r32_2k.sh`
- `docs/RESULTS.md`：追加 `E3-2k-r32` 的 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage12: lora rank=32 ablation`

---

### Step 13（对照实验）：DINOv3 全量微调（非 LoRA, 2k iters）

**目标**
- 补齐 ablation：Frozen vs LoRA vs Full fine-tune（验证 LoRA 是否在更少可训练参数下达到接近/更好的 mAP）

**产出物（仓库内）**
- `configs/dinov3_full/*_full_2kiter.py`
- `scripts/reproduce_dinov3_full_2k.sh`
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage13: dinov3 full fine-tune baseline`

---

### Step 14（对照实验）：全量微调（低学习率）

**动机**
- Step13 使用 baseline 同款学习率（`lr=2.5e-3`）出现灾难性遗忘；这里用更小学习率做更公平的 full fine-tune 对照。

**产出物（仓库内）**
- `configs/dinov3_full/*_lr0p00025.py`
- `scripts/reproduce_dinov3_full_2k_lr0p00025.sh`
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage14: full fine-tune with lower lr`

---

### Step 15（扩展实验）：ConvNeXt-Base（Frozen, 2k）

**目标**
- 验证更强 backbone（ConvNeXt-Base DINOv3）在 Frozen 情况下的上限

**产出物（仓库内）**
- `configs/dinov3_frozen/*convnext_base*_2kiter.py`
- `scripts/reproduce_dinov3_frozen_convnext_base_2k.sh`
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage15: convnext-base frozen`

---

### Step 16（扩展实验）：ConvNeXt-Base（LoRA r=16, 2k）

**目标**
- 在 ConvNeXt-Base 上跑 LoRA（r=16），对比 ConvNeXt-S 的 LoRA 结果

**产出物（仓库内）**
- `configs/dinov3_lora/*convnext_base*_r16_2kiter.py`
- `scripts/reproduce_lora_r16_convnext_base_2k.sh`
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage16: convnext-base lora r16`

---

### Step 17（对照实验）：ConvNeXt-Base（Full fine-tune, low LR, 2k）

**目标**
- 在 ConvNeXt-Base 上做低学习率全量微调，对比 LoRA 的参数效率与性能

**产出物（仓库内）**
- `configs/dinov3_full/*convnext_base*_lr0p00025.py`
- `scripts/reproduce_dinov3_full_convnext_base_2k_lr0p00025.sh`
- `docs/RESULTS.md`：追加对应 mAP

**验收**
- 训练完成保存 `iter_2000.pth`；在 RSAR val 上评估得到 mAP

**结束必须 push**
- Commit message 建议：`stage17: convnext-base full fine-tune low lr`

## 2. 兜底路线（当 Step 3/4 在 RSAR/DOTA 卡住时启用）

**启用条件（建议）**
- DOTA/OBB 的数据对齐或训练链路卡住超过 3 天且无明确解决路径

**兜底步骤**
- 先整理 HRSID 到 HBB（COCO/VOC）→ 训练 baseline → DINOv3 frozen → LoRA
- 意义：先证明“DINOv3+LoRA 在 SAR 迁移上是有效的”，再回到 RSAR 主线

同样：每个兜底步骤结束也必须 `commit + push`。
