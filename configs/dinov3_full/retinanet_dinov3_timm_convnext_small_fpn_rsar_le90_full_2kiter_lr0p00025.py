_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter.py',
]

# Lower LR for full fine-tuning to avoid catastrophic forgetting.
optimizer = dict(lr=0.00025)

