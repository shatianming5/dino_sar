# dino_sar

**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar

**工作流（最重要）：** `EXECUTION_PLAN.md` 里每完成一个步骤（阶段）→ 立刻 `git commit` → `git push` 到 `origin/main`。

## 这是什么

用 **DINOv3 + LoRA** 做 **SAR 目标检测（OBB）** 的工程与实验记录；主线数据集 **RSAR（6 类，DOTA/OBB）**，兜底验证 **HRSID（HBB）**。

## 文档入口

- `plan.md`：原始 Proposal（分阶段目标/验收/注意事项）
- `EXECUTION_PLAN.md`：把 `plan.md` 落到“可执行步骤 + 每步结束必须 push”的执行清单
- `docs/REPRODUCE.md`：从零复现（环境→训练→评估）
- `docs/RESULTS.md`：结果表（统一口径）
- `docs/ROBUSTNESS.md`：鲁棒性曲线（干扰注入）

## 结果速览（RSAR test，mAP@0.5）

> 完整结果见 `docs/RESULTS.md`。  
> 注意：`scripts/eval_baseline.sh` 默认评估 config 的 `data.test`（本仓库 RSAR 配置默认指向 `test/`）。

**RSAR 6-class 最新复现（test，mAP@0.5）**
- Baseline R50-FPN (10k, mean±std)：0.1008±0.0203（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_baseline_r50_10k.md`）
- LoRA r=16 (2k, mean±std)：0.0320±0.0058（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_2k.md`）
- LoRA r=16 (10k, full targets, mean±std)：0.0881±0.0031（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_10k.md`）
- LoRA r=16 (10k, ConvNeXt-Base, mean±std)：0.0892±0.0055（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_convnext_base_10k.md`）
- LoRA r=16 target=stages2+3 (10k, mean±std)：0.0846±0.0023（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_stage23_10k.md`）

## 数据说明（不进 Git）

本仓库不提交数据集文件（`train/`、`val/`、`test/` 已在 `.gitignore` 中忽略）。在本机准备好数据后再进行训练/评估。

RSAR 默认 6 类（标注文本全小写）：`aircraft` / `bridge` / `car` / `harbor` / `ship` / `tank`（详见 `docs/DATA.md`）。
