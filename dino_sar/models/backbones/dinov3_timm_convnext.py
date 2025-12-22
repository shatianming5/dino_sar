from __future__ import annotations

from typing import Sequence

from torch import nn

from mmrotate.models.builder import ROTATED_BACKBONES


@ROTATED_BACKBONES.register_module()
class Dinov3TimmConvNeXt(nn.Module):
    """ConvNeXt backbone with DINOv3 weights via timm HuggingFace checkpoints.

    Why: official DINOv3 weight URLs may be blocked in some networks; timm hosts
    compatible checkpoints on HuggingFace (non-gated).
    """

    def __init__(
        self,
        model_name: str = "convnext_small.dinov3_lvd1689m",
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)
        self.frozen = frozen

        try:
            import timm
        except Exception as e:
            raise RuntimeError(
                "Dinov3TimmConvNeXt requires `timm`. Install it in the conda env: `pip install timm>=1.0.17`."
            ) from e

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        outs = self.backbone(x)
        return list(outs)
