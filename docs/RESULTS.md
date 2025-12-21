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
| E2 | DINOv3 Frozen | - | TBD |
| E3 | DINOv3 + LoRA | - | TBD |

