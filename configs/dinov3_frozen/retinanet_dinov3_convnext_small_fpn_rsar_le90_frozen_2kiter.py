import os

_base_ = [
    '../baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py',
]

custom_imports = dict(
    imports=[
        'dino_sar.datasets.rsar_dota',
        'dino_sar.models.backbones.dinov3_convnext',
        'dino_sar.patches.mmrotate_nms_device',
    ],
    allow_failed_imports=False,
)

dinov3_repo_dir = os.environ.get('DINOV3_REPO_DIR') or os.environ.get('DINOV3_REPO')

model = dict(
    backbone=dict(
        _delete_=True,
        type='Dinov3ConvNeXt',
        model_name='dinov3_convnext_small',
        pretrained=True,
        repo_dir=dinov3_repo_dir if dinov3_repo_dir else None,
        out_indices=(0, 1, 2, 3),
        frozen=True,
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        start_level=1,
    ),
)

