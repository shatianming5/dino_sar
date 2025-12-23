_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage123_2kiter.py',
]

# Longer schedule: 10k iterations.

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=10000)
lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[1600, 9000],
)

evaluation = dict(interval=1000000, metric='mAP', iou_thr=0.5)
checkpoint_config = dict(interval=2000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

