from __future__ import annotations

import importlib.util
import os
import os.path as osp
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
        repo_dir: str | None = None,
        weights: str = "LVD1689M",
        out_indices: Sequence[int] = (0, 1, 2, 3),
        reshape: bool = True,
        norm: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.repo_dir = repo_dir
        self.weights = weights
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
        if not repo_dir:
            raise ValueError(
                "Missing dinov3 repo path. Clone https://github.com/facebookresearch/dinov3 and set "
                "repo_dir=... or env DINOV3_REPO_DIR/DINOV3_REPO."
            )

        convnext_path = osp.join(repo_dir, "dinov3", "models", "convnext.py")
        if not osp.exists(convnext_path):
            raise FileNotFoundError(f"dinov3 ConvNeXt source not found: {convnext_path}")

        module_name = "_dinov3_convnext_local"
        spec = importlib.util.spec_from_file_location(module_name, convnext_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from: {convnext_path}")
        convnext_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convnext_mod)

        if not self.model_name.startswith("dinov3_convnext_"):
            raise ValueError(f"Unsupported model_name={self.model_name!r}; expected dinov3_convnext_*")

        size = self.model_name.replace("dinov3_convnext_", "", 1)
        if size not in convnext_mod.convnext_sizes:
            raise ValueError(f"Unsupported ConvNeXt size={size!r}; expected one of {list(convnext_mod.convnext_sizes)}")

        size_dict = convnext_mod.convnext_sizes[size]
        compact_arch_name = f"convnext_{size}"

        model = convnext_mod.ConvNeXt(
            in_chans=3,
            depths=size_dict["depths"],
            dims=size_dict["dims"],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
        )

        if self.pretrained:
            weights_name = self.weights.strip().upper()
            if weights_name in {"LVD1689M", "SAT493M"}:
                weights_name = weights_name.lower()
                hash_map = {
                    "tiny": "21b726bb",
                    "small": "296db49d",
                    "base": "801f2ba9",
                    "large": "61fa432d",
                }
                url_hash = hash_map[size]
                url = (
                    f"https://dl.fbaipublicfiles.com/dinov3/dinov3_{compact_arch_name}/"
                    f"dinov3_{compact_arch_name}_pretrain_{weights_name}-{url_hash}.pth"
                )
            else:
                url = self.weights

            try:
                if osp.exists(url):
                    state_dict = torch.load(url, map_location="cpu")
                else:
                    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv3 ConvNeXt weights from {url}. "
                    "If the official URL is blocked (e.g. 403), download the .pth locally and set `weights` to that path."
                ) from e
        else:
            model.init_weights()

        return model

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
