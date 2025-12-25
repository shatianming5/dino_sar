# dataset settings (RSAR / DOTA-OBB) with class-balanced oversampling

dataset_type = "RSARDOTADataset"
data_root = "./"

angle_version = "le90"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=(800, 800)),
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25],
        direction=["horizontal", "vertical", "diagonal"],
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# Oversample rare classes following LVIS repeat-factor strategy:
# r(c) = max(1, sqrt(t / f(c))) where f(c) is the fraction of images containing class c.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="ClassBalancedDataset",
        oversample_thr=0.1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train/annfiles",
            img_prefix=data_root + "train/images",
            cache_dir=data_root + "outputs/cache/rsar_dota",
            pipeline=train_pipeline,
            version=angle_version,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val/annfiles",
        img_prefix=data_root + "val/images",
        cache_dir=data_root + "outputs/cache/rsar_dota",
        pipeline=test_pipeline,
        version=angle_version,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test/annfiles",
        img_prefix=data_root + "test/images",
        cache_dir=data_root + "outputs/cache/rsar_dota",
        pipeline=test_pipeline,
        version=angle_version,
    ),
)
