_base_ = ["./retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_10kiter.py"]

# Use class-balanced oversampling for rare classes on RSAR 6-class.
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

