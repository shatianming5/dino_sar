_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py',
]

# Longer schedule (5k iters), but keep the early LR decays from the 2k schedule.

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=5000)
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
checkpoint_config = dict(interval=1000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

