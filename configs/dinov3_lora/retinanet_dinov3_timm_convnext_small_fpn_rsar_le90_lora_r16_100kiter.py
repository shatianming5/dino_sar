_base_ = [
    './retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_10kiter.py',
]

# Long schedule: 100k iterations.
# LR strategy: early decay at 16k, then keep mid LR until late, then final decay.

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=100000)
lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[16000, 90000],
)

evaluation = dict(interval=1000000, metric='mAP', iou_thr=0.5)
checkpoint_config = dict(interval=20000, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

