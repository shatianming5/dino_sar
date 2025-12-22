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
| E3 | DINOv3 + LoRA | - | TBD |
| E3-2k | DINOv3 + LoRA (ConvNeXt-S, r=8, 2k iters) | 0.2375 | `configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py` |
| E4-2k | LoRA + SARAug (speckle+gamma, 2k iters) | 0.1845 | `configs/aug_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_saraug_speckle_gamma_2kiter.py` |
| E5-2k | LoRA + SARAug (log, 2k iters) | 0.1521 | `configs/aug_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_saraug_log_2kiter.py` |

> 备注：mmrotate `0.3.4` 在 CUDA 推理阶段 `multiclass_nms_rotated` 存在 device mismatch 问题，本仓库用 `dino_sar/patches/mmrotate_nms_device.py` 做了运行时修复。
