# dataset settings (RSAR / DOTA-OBB) with class-balanced oversampling

_base_ = ["./rsar_dota_le90.py"]

# Oversample rare classes following LVIS repeat-factor strategy.
# r(c) = max(1, sqrt(t / f(c))) where f(c) is the fraction of images containing class c.
# With RSAR, aircraft/car/tank/harbor are rare by image-frequency, so this helps them show up.
data = dict(
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
    )
)

