# 实验结果（统一口径）

**GitHub 仓库：** https://github.com/shatianming5/dino_sar  
**强制规则：** 每个阶段（Step）结束必须 `commit + push`

---

## 指标口径

- RSAR（DOTA-OBB）使用 **mAP@IoU=0.5**
- RSAR 默认 6 类：`aircraft` / `bridge` / `car` / `harbor` / `ship` / `tank`
- 默认使用 `mim test` 的 `cfg.data.test` 作为评估 split（本仓库 RSAR 配置默认指向 `test/`）
- 评估默认阈值：`cfg.model.test_cfg.score_thr=0.05`（避免产生过多候选框导致评估极慢）

---

## RSAR 6-class（当前）

| Setting | seed0 | seed1 | seed2 | mean | std (sample) | 复现记录 |
|---|---:|---:|---:|---:|---:|---|
| Baseline R50-FPN (10k iters) | 0.0776 | 0.1095 | 0.1152 | 0.1008 | 0.0203 | `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_baseline_r50_10k.md` |
| LoRA r=16 (2k iters) | 0.0253 | 0.0354 | 0.0352 | 0.0320 | 0.0058 | `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_2k.md` |
| LoRA r=16 (10k iters, full targets) | 0.0882 | 0.0912 | 0.0850 | 0.0881 | 0.0031 | `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_10k.md` |
| LoRA r=16 (ConvNeXt-Base, 10k iters) | 0.0829 | 0.0925 | 0.0923 | 0.0892 | 0.0055 | `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_convnext_base_10k.md` |
| LoRA r=16 target=stages2+3 (10k iters) | 0.0828 | 0.0839 | 0.0871 | 0.0846 | 0.0023 | `docs/repro_runs/2025-12-24_6class_scorethr0p05_multiseed_lora_r16_stage23_10k.md` |
