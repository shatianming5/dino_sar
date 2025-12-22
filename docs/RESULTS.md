# 实验结果（统一口径）

**GitHub 仓库：** https://github.com/shatianming5/dino_sar  
**强制规则：** 每个阶段（Step）结束必须 `commit + push`

---

## 指标口径

- RSAR（DOTA-OBB）使用 **mAP@IoU=0.5**

---

## 主结果表

| ID | Setting | mAP(0.5) | 备注 |
|---:|---|---:|---|
| E1 | Baseline (Rotated RetinaNet R50-FPN) | - | `configs/baselines/rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py` |
| E1-2k | Baseline (train split, 2k iters) | 0.0618 | `configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py` |
| E1-debug | Baseline (train=val, 2k iters) | 0.1510 | `configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_val_2kiter.py` |
| E2 | DINOv3 Frozen | - | TBD |
| E2-2k | DINOv3 Frozen (ConvNeXt-S, 2k iters) | 0.0400 | `configs/dinov3_frozen/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_frozen_2kiter.py` |
| E2-2k-cnbase | DINOv3 Frozen (ConvNeXt-Base, 2k iters) | 0.0562 | `configs/dinov3_frozen/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_frozen_2kiter.py` |
| E2-2k-full | DINOv3 Full fine-tune (ConvNeXt-S, 2k iters, lr=2.5e-3) | 0.0001 | `configs/dinov3_full/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter.py` |
| E2-2k-full-lr0p00025 | DINOv3 Full fine-tune (ConvNeXt-S, 2k iters, lr=2.5e-4) | 0.2681 | `configs/dinov3_full/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter_lr0p00025.py` |
| E2-2k-full-cnbase-lr0p00025 | DINOv3 Full fine-tune (ConvNeXt-Base, 2k iters, lr=2.5e-4) | 0.2570 | `configs/dinov3_full/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_full_2kiter_lr0p00025.py` |
| E3 | DINOv3 + LoRA | - | TBD |
| E3-2k | DINOv3 + LoRA (ConvNeXt-S, r=8, 2k iters) | 0.2375 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py` |
| E3-2k-r16 | DINOv3 + LoRA (ConvNeXt-S, r=16, 2k iters) | 0.3293 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py` |
| E3-2k-r16-fc2 | DINOv3 + LoRA (ConvNeXt-S, r=16, target=fc2 only, 2k iters) | 0.2320 | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_fc2only_2kiter.py` |
| E3-2k-r16-stage3 | DINOv3 + LoRA (ConvNeXt-S, r=16, target=stage3 only, 2k iters) | 0.1035 | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage3_2kiter.py` |
| E3-2k-r16-stage23 | DINOv3 + LoRA (ConvNeXt-S, r=16, target=stages2+3, 2k iters) | 0.3231 | `configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_2kiter.py` |
| E3-2k-cnbase-r16 | DINOv3 + LoRA (ConvNeXt-Base, r=16, 2k iters) | 0.1359 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_2kiter.py` |
| E3-2k-r32 | DINOv3 + LoRA (ConvNeXt-S, r=32, 2k iters) | 0.3153 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r32_2kiter.py` |
| E3-5k-r16 | DINOv3 + LoRA (ConvNeXt-S, r=16, 5k iters, scaled LR steps) | 0.1771 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_5kiter.py` |
| E3-5k-r16-early | DINOv3 + LoRA (ConvNeXt-S, r=16, 5k iters, early LR decay) | 0.2904 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_5kiter_early_decay.py` |
| E4-2k | LoRA + SARAug (speckle+gamma, 2k iters) | 0.1845 | `configs/aug_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_saraug_speckle_gamma_2kiter.py` |
| E5-2k | LoRA + SARAug (log, 2k iters) | 0.1521 | `configs/aug_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_saraug_log_2kiter.py` |

## 多 seed 稳定性

| Setting | seed0 | seed1 | seed2 | mean | std (sample) |
|---|---:|---:|---:|---:|---:|
| LoRA r=16 (2k iters) `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py` | 0.2004 | 0.1866 | 0.2739 | 0.2203 | 0.0469 |
| LoRA r=16 (10k iters, step=1600/9000) `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_10kiter.py` | - | - | - | - | - |

> 备注：mmrotate `0.3.4` 在 CUDA 推理阶段 `multiclass_nms_rotated` 存在 device mismatch 问题，本仓库用 `dino_sar/patches/mmrotate_nms_device.py` 做了运行时修复。
