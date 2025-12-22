_base_ = [
    '../dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r8_2kiter.py',
]

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

custom_imports = dict(
    imports=[
        'dino_sar.datasets.rsar_dota',
        'dino_sar.models.lora',
        'dino_sar.models.backbones.dinov3_timm_convnext_lora',
        'dino_sar.pipelines.sar_aug',
        'dino_sar.patches.mmrotate_nms_device',
    ],
    allow_failed_imports=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version,
    ),
    dict(type='RandomSpeckleNoise', p=0.5, sigma_range=(0.05, 0.25), mode='lognormal'),
    dict(type='RandomGamma', p=0.5, gamma_range=(0.7, 1.5)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(pipeline=train_pipeline),
)
