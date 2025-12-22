_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py',
]

model = dict(
    backbone=dict(
        lora_r=16,
        lora_alpha=32.0,
    ),
)

