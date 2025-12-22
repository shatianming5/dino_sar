_base_ = [
    '../dinov3_frozen/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_frozen_2kiter.py',
]

model = dict(
    backbone=dict(
        frozen=False,
    ),
)

