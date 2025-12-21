_base_ = [
    './rotated_retinanet_obb_r50_fpn_rsar_le90_smoke.py',
]

# Debug baseline: train & eval on val split (fast sanity check).

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

