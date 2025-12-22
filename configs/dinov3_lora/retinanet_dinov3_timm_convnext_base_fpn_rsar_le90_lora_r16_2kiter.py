_base_ = [
    '../baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py',
]

custom_imports = dict(
    imports=[
        'dino_sar.datasets.rsar_dota',
        'dino_sar.models.lora',
        'dino_sar.models.backbones.dinov3_timm_convnext_lora',
        'dino_sar.patches.mmrotate_nms_device',
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='Dinov3TimmConvNeXtLoRA',
        model_name='convnext_base.dinov3_lvd1689m',
        pretrained=True,
        out_indices=(0, 1, 2, 3),
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
        lora_target_keywords=('mlp.fc1', 'mlp.fc2'),
    ),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        start_level=1,
    ),
)

