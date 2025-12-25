_base_ = ["./rotated_retinanet_obb_r50_fpn_1x_rsar_le90_cb0p1.py"]

# Class-balanced baseline: train 10k iterations on train split, then evaluate separately.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=10000)
lr_config = dict(
    _delete_=True,
    policy="step",
    by_epoch=False,
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8000, 9500],
)

evaluation = dict(interval=1000000, metric="mAP", iou_thr=0.5)
checkpoint_config = dict(interval=2000)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

