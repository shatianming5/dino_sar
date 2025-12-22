# 鲁棒性评估（Step8：干扰注入）

**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar  
**强制规则：** 每个阶段（Step）结束必须 `commit + push`

---

## 目标

- 在验证集子集上注入可控干扰（合成），输出 **mAP-干扰强度曲线**
- 对比 baseline vs DINOv3+LoRA 的趋势

---

## 干扰类型

- **Gaussian（确定性）**：`dino_sar/pipelines/jamming.py` 中 `DeterministicGaussianJamming`

---

## 复现命令（示例）

### 1) 生成 val 子集（例如 200 张）

```bash
conda run -n dino_sar python tools/jamming/make_subset.py --split val --n 200
```

### 2) 跑曲线（示例：baseline）

```bash
conda run -n dino_sar python tools/jamming/run_curve.py \\
  --config configs/robustness/retinanet_r50_fpn_rsar_le90_jam_gaussian.py \\
  --ckpt outputs/baselines/retinanet_r50_fpn_rsar_le90_train_2kiter/iter_2000.pth \\
  --subset-dir outputs/datasets/rsar_val_200 \\
  --sigmas 0,0.02,0.05,0.1,0.2
```

### 3) 跑曲线（示例：LoRA）

```bash
conda run -n dino_sar python tools/jamming/run_curve.py \\
  --config configs/robustness/retinanet_dinov3_lora_r8_jam_gaussian.py \\
  --ckpt outputs/dinov3_lora/retinanet_timm_convnext_small_dinov3_lora_r8_train_2kiter/iter_2000.pth \\
  --subset-dir outputs/datasets/rsar_val_200 \\
  --sigmas 0,0.02,0.05,0.1,0.2
```

---

## 结果（待补充）

**评估集：** `val` 子集 200 张（`tools/jamming/make_subset.py --split val --n 200`）  
**干扰：** `DeterministicGaussianJamming`（对每张图按 filename hash 固定噪声；强度由 `DINO_SAR_JAM_SIGMA` 控制）

| Setting | sigma=0 | sigma=0.05 | sigma=0.1 | sigma=0.2 |
|---|---:|---:|---:|---:|
| Baseline (R50-FPN, iter_2000) | 0.1080 | 0.1099 | 0.1182 | 0.1155 |
| LoRA (DINOv3 ConvNeXt-S, r=8, iter_2000) | 0.2445 | 0.2434 | 0.2423 | 0.2350 |
