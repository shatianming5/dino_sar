from __future__ import annotations

import os
from typing import Sequence

import torch
from torch import nn

from mmrotate.models.builder import ROTATED_BACKBONES


@ROTATED_BACKBONES.register_module()
class Dinov3ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt backbone adapter for MMRotate/MMDet2 FPN-style detectors."""

    def __init__(
        self,
        model_name: str = "dinov3_convnext_small",
        pretrained: bool = True,
        repo: str = "facebookresearch/dinov3",
        repo_dir: str | None = None,
        source: str = "auto",
        out_indices: Sequence[int] = (0, 1, 2, 3),
        reshape: bool = True,
        norm: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.repo = repo
        self.repo_dir = repo_dir
        self.source = source
        self.out_indices = tuple(sorted(out_indices))
        self.reshape = reshape
        self.norm = norm
        self.frozen = frozen

        self.backbone = self._load_backbone()

        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _load_backbone(self) -> nn.Module:
        repo_dir = self.repo_dir or os.environ.get("DINOV3_REPO_DIR") or os.environ.get("DINOV3_REPO")
        if self.source not in {"auto", "github", "local"}:
            raise ValueError(f"Unsupported source={self.source!r}; expected auto/github/local")

        if repo_dir:
            return torch.hub.load(repo_dir, self.model_name, pretrained=self.pretrained, source="local")

        if self.source == "local":
            raise ValueError(
                "source='local' requires repo_dir or env DINOV3_REPO_DIR/DINOV3_REPO pointing to a cloned dinov3 repo."
            )

        return torch.hub.load(self.repo, self.model_name, pretrained=self.pretrained, source="github")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        outs = self.backbone.get_intermediate_layers(
            x,
            n=list(self.out_indices),
            reshape=self.reshape,
            norm=self.norm,
        )
        return list(outs)
