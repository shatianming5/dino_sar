# 复现指南（Step9）

**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar  
**强制规则：** 每个阶段（Step）结束必须 `commit + push`

---

## 0) 前置条件

- 已安装 Conda（推荐 Miniconda）
- 有可用 NVIDIA GPU（本项目主线以 CUDA 训练/评估）
- RSAR 数据已放到仓库根目录（不提交到 Git）：
  - `train/`
  - `val/`
  - `test/`

目录结构参考：`docs/DATA.md`

---

## 1) 环境安装（主线 MMRotate + timm 权重）

按 `docs/ENV_SETUP.md` 执行即可；核心命令如下：

```bash
conda env create -f environment.yml
conda activate dino_sar

pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip install -U openmim
mim install -U "mmcv-full==1.7.2"
pip install -U "mmdet>=2.25.1,<3.0.0" "mmrotate==0.3.4"
```

冒烟检查：

```bash
python tools/smoke_test/check_openmmlab.py
```

---

## 2) 数据体检（可选但建议）

```bash
conda run -n dino_sar python tools/data_audit/rsar_dota_audit.py --root . --ann-samples 200
```

---

## 3) Baseline（R50-FPN, 2k iters）

训练：

```bash
bash scripts/train_baseline.sh \
  configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py \
  outputs/baselines/retinanet_r50_fpn_rsar_le90_train_2kiter \
  1
```

测试集评估（默认读取 config 的 `data.test`）：

```bash
bash scripts/eval_baseline.sh \
  configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py \
  outputs/baselines/retinanet_r50_fpn_rsar_le90_train_2kiter/iter_2000.pth \
  1 \
  outputs/eval/baseline_r50_2k_test
```

---

## 3b) Baseline（R50-FPN, 10k iters，可选但更合理）

```bash
bash scripts/reproduce_baseline_r50_10k.sh 1 0
```

---

## 4) DINOv3 + LoRA（ConvNeXt-S, r=8, 2k iters）

说明：
- 本仓库默认使用 `timm/convnext_small.dinov3_lvd1689m`（HuggingFace 下载，非 gated）
- LoRA 可训练参数比例会在日志里输出（约 2%）

训练：

```bash
bash scripts/train_baseline.sh \
  configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py \
  outputs/dinov3_lora/retinanet_timm_convnext_small_dinov3_lora_r8_train_2kiter \
  1
```

测试集评估（默认读取 config 的 `data.test`）：

```bash
bash scripts/eval_baseline.sh \
  configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py \
  outputs/dinov3_lora/retinanet_timm_convnext_small_dinov3_lora_r8_train_2kiter/iter_2000.pth \
  1 \
  outputs/eval/dinov3_lora_r8_test
```

结果表：`docs/RESULTS.md`

---

## 5) 鲁棒性（Step8，可选）

参考 `docs/ROBUSTNESS.md`，一键生成子集并跑曲线：

```bash
conda run -n dino_sar python tools/jamming/make_subset.py --split val --n 200

conda run -n dino_sar python tools/jamming/run_curve.py \
  --config configs/robustness/retinanet_dinov3_lora_r8_jam_gaussian.py \
  --ckpt outputs/dinov3_lora/retinanet_timm_convnext_small_dinov3_lora_r8_train_2kiter/iter_2000.pth \
  --subset-dir outputs/datasets/rsar_val_200 \
  --sigmas 0,0.05,0.1,0.2 \
  --out outputs/robustness/lora_r8_gauss
```

---

## 6) 推荐配置（当前最优）

- LoRA rank=16：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py`
- 一键脚本：`scripts/reproduce_lora_r16_2k.sh`

## 7) 更长训练（可选）

- LoRA rank=16, 5k iters：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_5kiter.py`
- 一键脚本：`scripts/reproduce_lora_r16_5k.sh`

## 8) ConvNeXt-Base（可选）

- Frozen 2k：`scripts/reproduce_dinov3_frozen_convnext_base_2k.sh`
- LoRA r16 2k：`scripts/reproduce_lora_r16_convnext_base_2k.sh`
- Full fine-tune (lr=2.5e-4) 2k：`scripts/reproduce_dinov3_full_convnext_base_2k_lr0p00025.sh`

## 9) LoRA target 消融（可选）

- fc2 only：`scripts/reproduce_lora_r16_fc2only_2k.sh`
- stages_3 only：`scripts/reproduce_lora_r16_stage3_2k.sh`
- stages_2+3：`scripts/reproduce_lora_r16_stage23_2k.sh`

## 10) 多 seed 稳定性（可选）

- LoRA r16 2k（seed=0/1/2）：`scripts/run_multiseed_lora_r16_2k.sh`

## 11) 更长训练 + 多 seed（可选）

- LoRA r16 10k（seed=0/1/2）：`scripts/run_multiseed_lora_r16_10k.sh`
