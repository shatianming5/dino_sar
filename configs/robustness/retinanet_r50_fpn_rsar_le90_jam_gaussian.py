_base_ = [
    '../baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

jamming = dict(type='DeterministicGaussianJamming', sigma=0.0, seed_offset=0)

custom_imports = dict(
    imports=[
        'dino_sar.datasets.rsar_dota',
        'dino_sar.pipelines.jamming',
        'dino_sar.patches.mmrotate_nms_device',
    ],
    allow_failed_imports=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            jamming,
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

