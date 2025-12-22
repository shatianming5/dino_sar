_base_ = [
    '../baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py',
]

custom_imports = dict(
    imports=[
        'dino_sar.datasets.rsar_dota',
        'dino_sar.models.backbones.dinov3_timm_convnext',
        'dino_sar.patches.mmrotate_nms_device',
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='Dinov3TimmConvNeXt',
        model_name='convnext_base.dinov3_lvd1689m',
        pretrained=True,
        frozen=True,
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        start_level=1,
    ),
)

