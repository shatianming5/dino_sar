# 执行计划检查清单（对齐 `EXECUTION_PLAN.md`）

> 目的：逐条核对 `EXECUTION_PLAN.md` 的“产出物/验收”，并给出当前仓库状态与下一步建议。

## 已完成（核心链路）

- Step 1：成功标准与交付物
  - `README_proposal.md`
  - `docs/RESULTS.md`
- Step 2：环境与代码基座
  - `environment.yml`
  - `docs/ENV_SETUP.md`
  - `tools/smoke_test/check_openmmlab.py`
- Step 3：RSAR 数据体检与格式对齐
  - `tools/data_audit/rsar_dota_audit.py`
  - `docs/DATA.md`
- Step 4：强 Baseline 锁死
  - 训练/评估脚本：`scripts/train_baseline.sh`、`scripts/eval_baseline.sh`
  - Baseline 配置：`configs/baselines/`
- Step 5：DINOv3 Frozen 接入
  - `configs/dinov3_frozen/`
- Step 6：LoRA 参数高效微调
  - `configs/dinov3_lora/`
  - `dino_sar/models/lora.py`
  - `dino_sar/models/backbones/dinov3_timm_convnext_lora.py`
- Step 7：SAR 增强/策略消融
  - `configs/aug_ablation/`
- Step 8：鲁棒性评估（可选）
  - `configs/robustness/`
  - `tools/jamming/`
  - `docs/ROBUSTNESS.md`
- Step 9：结果整理与复现
  - 复现记录：`docs/repro_runs/`
  - 统一结果表：`docs/RESULTS.md`
  - 复现说明：`docs/REPRODUCE.md`

## 已完成（稳定性与对齐口径：Step 21~26）

- Step 21：LoRA r=16（2k）多 seed
  - `scripts/run_multiseed_lora_r16_2k_6class.sh`
  - `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_2k.md`
- Step 22：LoRA r=16（10k, full targets）多 seed
  - `scripts/run_multiseed_lora_r16_10k_6class.sh`
  - `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_10k.md`
- Step 23/24：Baseline R50-FPN（10k）+ 多 seed
  - `scripts/run_multiseed_baseline_r50_10k_6class.sh`
  - `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_baseline_r50_10k.md`
- Step 25：ConvNeXt-Base + LoRA r=16（10k）多 seed
  - `scripts/run_multiseed_lora_r16_convnext_base_10k_6class.sh`
  - `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_convnext_base_10k.md`
- Step 26：LoRA target 扩展（10k 口径）
  - `scripts/run_multiseed_lora_r16_stage23_10k_6class.sh`
  - `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_stage23_10k.md`

## 额外已完成（计划外，但用于解释/提升低分）

- ClassBalancedDataset（10k 对比）
  - 结果汇总：`docs/RESULTS.md`
  - 复现记录：`docs/repro_runs/2025-12-25_6class_cb0p1_scorethr0p05_multiseed_baseline_r50_cb0p1_10k.md`
  - 复现记录：`docs/repro_runs/2025-12-25_6class_cb0p1_scorethr0p05_multiseed_lora_r16_cb0p1_10k.md`
  - 复现记录：`docs/repro_runs/2025-12-25_6class_cb0p1_scorethr0p05_multiseed_lora_r16_stage23_cb0p1_10k.md`
  - 复现记录：`docs/repro_runs/2025-12-25_6class_cb0p1_scorethr0p05_multiseed_lora_r16_convnext_base_cb0p1_10k.md`
- Baseline 100k（显著提升）
  - 复现记录：`docs/repro_runs/2025-12-25_6class_scorethr0p05_multiseed_baseline_r50_100k.md`
  - 低分根因分析：`docs/LOW_MAP_ANALYSIS.md`

## 待做（如果你要求“完全跑完所有扩展实验”）

- Step 10~20 的单次实验结果目前未在 `docs/repro_runs/` 与 `docs/RESULTS.md` 中形成完整记录（对应脚本与配置已具备：`scripts/reproduce_*.sh` / `configs/*`）。

