_base_ = [
    '../dinov3_frozen/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_frozen_2kiter.py',
]

model = dict(
    backbone=dict(
        frozen=False,
    ),
)

# Lower LR for full fine-tuning to avoid catastrophic forgetting.
optimizer = dict(lr=0.00025)

