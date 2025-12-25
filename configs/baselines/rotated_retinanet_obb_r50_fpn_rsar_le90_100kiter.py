_base_ = [
    './rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py',
]

# Long schedule: train 100k iterations on train split, then evaluate separately.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=100000)
lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[80000, 95000],
)

evaluation = dict(interval=1000000, metric='mAP', iou_thr=0.5)
checkpoint_config = dict(interval=20000, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

