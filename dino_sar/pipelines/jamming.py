from __future__ import annotations

import hashlib
import os

import numpy as np

from mmdet.datasets.builder import PIPELINES

from .sar_aug import _get_img_fields, _infer_scale


def _stable_int_seed(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


@PIPELINES.register_module()
class DeterministicGaussianJamming:
    """Deterministic additive Gaussian noise based on filename hash (for reproducible robustness eval)."""

    def __init__(self, sigma: float = 0.0, seed_offset: int = 0):
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)
        self.seed_offset = int(seed_offset)

    def __call__(self, results: dict) -> dict:
        sigma = self.sigma
        seed_offset = self.seed_offset

        env_sigma = os.environ.get("DINO_SAR_JAM_SIGMA")
        if env_sigma is not None and env_sigma != "":
            sigma = float(env_sigma)
        env_seed = os.environ.get("DINO_SAR_JAM_SEED_OFFSET")
        if env_seed is not None and env_seed != "":
            seed_offset = int(env_seed)

        if sigma <= 0:
            return results

        fname = results.get("filename") or results.get("ori_filename") or ""
        seed = (_stable_int_seed(str(fname)) + seed_offset) & 0x7FFFFFFF
        rng = np.random.RandomState(seed)

        for key in _get_img_fields(results):
            img = results[key]
            scale = _infer_scale(img)
            img_f = img.astype(np.float32)
            img_n = np.clip(img_f / scale, 0.0, 1.0)

            noise = rng.normal(loc=0.0, scale=sigma, size=img_n.shape).astype(np.float32)
            img_n = np.clip(img_n + noise, 0.0, 1.0)
            out = img_n * scale
            results[key] = out.astype(np.uint8) if img.dtype == np.uint8 else out

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, seed_offset={self.seed_offset})"
