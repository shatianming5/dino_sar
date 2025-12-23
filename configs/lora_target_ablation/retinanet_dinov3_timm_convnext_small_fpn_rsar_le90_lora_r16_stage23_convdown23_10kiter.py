_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py',
]

# Add Conv2d LoRA on ConvNeXt *later* downsample convs (stages_2/3 only).
model = dict(
    backbone=dict(
        lora_target_conv_keywords=('stages_2.downsample.1', 'stages_3.downsample.1'),
    ),
)

