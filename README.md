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

> 完整结果与更多消融见 `docs/RESULTS.md`；默认都是 2k iters（除非特别说明）。  
> 注意：`scripts/eval_baseline.sh` 默认评估 config 的 `data.test`（本仓库 RSAR 配置默认指向 `test/`）。
> 重要：下表为历史结果（曾使用 ship-only/1 类口径）；当前仓库默认 RSAR 6 类后需重新跑实验，mAP 不可直接对比。

**RSAR 6-class 最新复现（test，mAP@0.5）**
- Baseline R50-FPN (10k, mean±std)：0.1008±0.0203（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_baseline_r50_10k.md`）
- LoRA r=16 (2k, mean±std)：0.0320±0.0058（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_2k.md`）
- LoRA r=16 (10k, full targets, mean±std)：0.0881±0.0031（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_10k.md`）
- LoRA r=16 (10k, ConvNeXt-Base, mean±std)：0.0892±0.0055（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_convnext_base_10k.md`）
- LoRA r=16 target=stages2+3 (10k, mean±std)：0.0846±0.0023（见 `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_stage23_10k.md`）

| Setting | mAP(0.5) | Config |
|---|---:|---|
| Baseline (Rotated RetinaNet R50-FPN, 10k, mean±std) | 0.4769±0.0127 | `configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_10kiter.py` |
| Baseline (Rotated RetinaNet R50-FPN, 2k, quick) | 0.0618 | `configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py` |
| DINOv3 Frozen (ConvNeXt-S, 2k) | 0.0400 | `configs/dinov3_frozen/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_frozen_2kiter.py` |
| DINOv3 Frozen (ConvNeXt-Base, 2k) | 0.0562 | `configs/dinov3_frozen/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_frozen_2kiter.py` |
| DINOv3 Full FT (ConvNeXt-S, lr=2.5e-4, 2k) | 0.2681 | `configs/dinov3_full/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter_lr0p00025.py` |
| DINOv3 Full FT (ConvNeXt-Base, lr=2.5e-4, 2k) | 0.2570 | `configs/dinov3_full/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_full_2kiter_lr0p00025.py` |
| **DINOv3 + LoRA（target=stages2+3，多 seed mean±std）** (ConvNeXt-S, r=16, 10k) | **0.5192±0.0102** | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py` |
| DINOv3 + LoRA（target=stage2 only，seed0） (ConvNeXt-S, r=16, 10k) | 0.5246 | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage2_10kiter.py` |
| DINOv3 + LoRA（target=stages1+2+3，seed0） (ConvNeXt-S, r=16, 10k) | 0.5172 | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage123_10kiter.py` |
| DINOv3 + LoRA（full targets，多 seed mean±std） (ConvNeXt-S, r=16, 10k) | 0.5169±0.0111 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_10kiter.py` |
| DINOv3 + LoRA（ConvNeXt-Base, r=16, 10k mean±std） | 0.5071±0.0232 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_10kiter.py` |
| DINOv3 + LoRA（单次 best） (ConvNeXt-S, r=16, 2k) | 0.3293 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py` |

**LoRA 关键消融（ConvNeXt-S, 2k）**
- Rank：r=8→0.2375，r=16→0.3293，r=32→0.3153
- Target：`stages_2+3`→0.3231（接近全量），`mlp.fc2` only→0.2320，`stages_3` only→0.1035
- 5k iters：按比例延后 LR step→0.1771；保持早衰减→0.2904（仍不如 2k 单次 best）

**多 seed 稳定性（RSAR test）**
- Baseline 10k（seed=0/1/2）：0.4625 / 0.4813 / 0.4867；mean=0.4769，std=0.0127
- LoRA stages_2+3 10k（seed=0/1/2）：0.5245 / 0.5257 / 0.5075；mean=0.5192，std=0.0102
- LoRA stages_2+3 + convdown 10k（seed=0/1/2）：0.5374 / 0.5091 / 0.4661；mean=0.5042，std=0.0359（不稳定）
- LoRA ConvNeXt-Base 10k（seed=0/1/2）：0.5286 / 0.4825 / 0.5102；mean=0.5071，std=0.0232
- 2k（seed=0/1/2）：0.2004 / 0.1866 / 0.2739；mean=0.2203，std=0.0469
- 10k（seed=0/1/2）：0.5150 / 0.5069 / 0.5287；mean=0.5169，std=0.0111（见 `docs/RESULTS.md`）

## 数据说明（不进 Git）

本仓库不提交数据集文件（`train/`、`val/`、`test/` 已在 `.gitignore` 中忽略）。在本机准备好数据后再进行训练/评估。

RSAR 默认 6 类（标注文本全小写）：`aircraft` / `bridge` / `car` / `harbor` / `ship` / `tank`（详见 `docs/DATA.md`）。
