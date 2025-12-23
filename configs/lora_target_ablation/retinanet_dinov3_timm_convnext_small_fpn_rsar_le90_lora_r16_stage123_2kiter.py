_base_ = [
    '../dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py',
]

model = dict(
    backbone=dict(
        lora_target_keywords=('stages_1', 'stages_2', 'stages_3'),
    ),
)

