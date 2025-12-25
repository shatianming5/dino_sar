_base_ = [
    "./retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_10kiter.py",
    "../_base_/datasets/rsar_dota_le90_cb0p1.py",
]

# Keep the same batch config as default RSAR iter-based runs.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)
