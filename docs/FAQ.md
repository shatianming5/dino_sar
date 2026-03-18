# FAQ

## 你是什么模型 / What model does this repository implement?

本项目实现的是 **SAR-LoRA-DINO**，即在 **DINOv3（ConvNeXt）** 骨干网络上加入 **LoRA（低秩适配）** 微调模块，用于 **SAR（合成孔径雷达）目标检测**任务，基于 SARDet-100K 数据集和 MMDetection 3.x 框架。

**This repository implements SAR-LoRA-DINO**, a detection model combining:

- **Backbone**: ConvNeXt pretrained with DINOv3 self-supervised learning (`convnext_small.dinov3_lvd1689m` via `timm`).
- **PEFT**: LoRA (Low-Rank Adaptation) adapters injected into the backbone's MLP linear layers, allowing efficient fine-tuning with far fewer trainable parameters.
- **Detector head**: RetinaNet (or other MMDetection heads) on top of the frozen + LoRA-adapted ConvNeXt features.
- **Task / Dataset**: SAR object detection on [SARDet-100K](https://github.com/zcablii/SARDet_100K).

The key idea is to keep the DINOv3-pretrained ConvNeXt backbone frozen and only train lightweight LoRA modules (~1–2 % of total parameters), which yields strong SAR detection performance while reducing GPU memory and training time.

## `ImportError: cannot import name ...` / `mmcv.ops` not found

This usually means MMCV was installed with an incompatible PyTorch/CUDA.

- Reinstall following the versions in `docs/GETTING_STARTED.md`.
- Run `bash scripts/verify_env.sh` to sanity check imports.

## Dataset path not found

Set:

```bash
export SARDET100K_ROOT=/path/to/SARDet_100K
```

or symlink:

```bash
ln -s /path/to/SARDet_100K data/SARDet_100K
```

## CUDA OOM

Try one or more of:

- Reduce `TRAIN_BATCH_SIZE` (for `scripts/run_sardet_full_cfg.sh`).
- Reduce input size in the dataset pipeline (config change).
- Use fewer `num_workers`.

## NCCL multi-GPU errors

Some hosts have global NCCL env vars that break distributed training.
`scripts/run_sardet_full_cfg.sh` already unsets common NCCL variables before launch.

