_base_ = [
    './rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py',
]

# Smoke test config: run a few iterations to validate the training pipeline.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file='./val/annfiles',
        img_prefix='./val/images',
    ),
    val=dict(
        ann_file='./val/annfiles',
        img_prefix='./val/images',
    ),
    test=dict(
        ann_file='./val/annfiles',
        img_prefix='./val/images',
    ),
)

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=20)
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 3,
    step=[16],
)

evaluation = dict(interval=1000000, metric='mAP', iou_thr=0.5)
checkpoint_config = dict(interval=20)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
