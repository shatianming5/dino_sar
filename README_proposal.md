# DINOv3 + LoRA 适配 SAR 船只检测（RSAR / DOTA-OBB）

**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar  
**强制工作流：** 每完成一个阶段（Step）→ `git commit` → `git push origin main`（详见 `EXECUTION_PLAN.md`）

---

## 1) 任务定义（固定不改）

- **任务**：SAR 船只目标检测（**旋转框 OBB**）
- **主数据集**：RSAR（DOTA 标注格式：8 点 polygon + 类别 + difficult）
- **备选数据集（兜底）**：HRSID（先做水平框 HBB 快速验证）

---

## 2) 评估口径（固定不改）

- **主指标**：DOTA 风格 **OBB mAP@IoU=0.5**（简称 `mAP(0.5)`）
- **对比规则**：baseline / frozen / LoRA 使用同一评估脚本、同一 NMS/阈值策略

---

## 3) 最小实验矩阵（必须完成）

| ID | 模型 | Backbone 训练方式 | 目的 |
|---:|---|---|---|
| E1 | Rotated detector baseline | 传统 CNN backbone（如 ResNet50-FPN）全量训练 | 锁死强基线 |
| E2 | DINOv3 detector（Frozen） | **冻结** DINOv3，只训 neck/head | 验证管线与特征多尺度对接正确 |
| E3 | DINOv3 + LoRA | 只训 **LoRA + neck/head**（可选 LN/bias） | 证明“参数高效域对齐”有效 |
| E4（可选） | DINOv3 full fine-tune | 小学习率全量微调 | 回答“为什么不用全量微调” |

---

## 4) 交付物（最终验收）

- 可复现代码与配置：`configs/`、`scripts/`
- 数据体检脚本与报告：`tools/data_audit/`、`docs/DATA.md`
- 结果表与误差分析：`docs/RESULTS.md`
- 复现说明：`docs/REPRODUCE.md`
- 训练输出（本地）：`outputs/`（不进 Git，已在 `.gitignore`）

---

## 5) 目录约定（从现在开始遵守）

- `scripts/`：一键训练/评估脚本（入口统一在这里）
- `configs/`：训练配置（baseline / frozen / lora / ablation）
- `tools/`：数据处理、体检、可视化、鲁棒性脚本
- `docs/`：环境、数据、结果、复现说明
- `outputs/`：本地实验输出（忽略，不提交）

