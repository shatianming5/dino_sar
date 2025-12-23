_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py',
]

# Add Conv2d LoRA on ConvNeXt downsample/stem convs (minimal Conv2d coverage).
model = dict(
    backbone=dict(
        lora_target_conv_keywords=('stem_0', 'downsample.1'),
    ),
)

