_base_ = [
    './rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py',
]

# Quick baseline: train 2k iterations on train split, then evaluate separately.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=2000)
lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[1600, 1900],
)

evaluation = dict(interval=1000000, metric='mAP', iou_thr=0.5)
checkpoint_config = dict(interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

