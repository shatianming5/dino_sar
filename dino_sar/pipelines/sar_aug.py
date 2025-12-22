from __future__ import annotations

import numpy as np

from mmdet.datasets.builder import PIPELINES


def _infer_scale(img: np.ndarray) -> float:
    if img.dtype == np.uint8:
        return 255.0
    vmax = float(np.nanmax(img)) if img.size else 1.0
    return 255.0 if vmax > 1.5 else 1.0


def _get_img_fields(results: dict) -> list[str]:
    fields = results.get("img_fields", ["img"])
    return list(fields) if isinstance(fields, (list, tuple)) else ["img"]


@PIPELINES.register_module()
class RandomSpeckleNoise:
    """Multiplicative speckle-like noise for SAR intensity images."""

    def __init__(self, p: float = 0.5, sigma_range: tuple[float, float] = (0.05, 0.25), mode: str = "lognormal"):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if len(sigma_range) != 2 or sigma_range[0] < 0 or sigma_range[1] < sigma_range[0]:
            raise ValueError("sigma_range must be (min, max) with 0 <= min <= max")
        if mode not in {"lognormal", "gaussian"}:
            raise ValueError("mode must be 'lognormal' or 'gaussian'")
        self.p = float(p)
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        self.mode = mode

    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.p:
            return results

        for key in _get_img_fields(results):
            img = results[key]
            scale = _infer_scale(img)
            img_f = img.astype(np.float32)
            img_n = np.clip(img_f / scale, 0.0, 1.0)

            sigma = float(np.random.uniform(self.sigma_range[0], self.sigma_range[1]))
            noise_shape = img_n.shape if img_n.ndim == 2 else img_n.shape[:2] + (1,)
            if self.mode == "lognormal":
                mult = np.exp(np.random.normal(loc=0.0, scale=sigma, size=noise_shape)).astype(np.float32)
            else:
                mult = (1.0 + np.random.normal(loc=0.0, scale=sigma, size=noise_shape)).astype(np.float32)

            img_n = np.clip(img_n * mult, 0.0, 1.0)
            out = img_n * scale
            results[key] = out.astype(np.uint8) if img.dtype == np.uint8 else out

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, sigma_range={self.sigma_range}, mode={self.mode!r})"


@PIPELINES.register_module()
class RandomGamma:
    """Random gamma correction for SAR intensity images."""

    def __init__(self, p: float = 0.5, gamma_range: tuple[float, float] = (0.7, 1.5)):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if len(gamma_range) != 2 or gamma_range[0] <= 0 or gamma_range[1] < gamma_range[0]:
            raise ValueError("gamma_range must be (min, max) with 0 < min <= max")
        self.p = float(p)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))

    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.p:
            return results

        gamma = float(np.random.uniform(self.gamma_range[0], self.gamma_range[1]))
        for key in _get_img_fields(results):
            img = results[key]
            scale = _infer_scale(img)
            img_f = img.astype(np.float32)
            img_n = np.clip(img_f / scale, 0.0, 1.0)
            img_n = np.power(img_n, gamma, dtype=np.float32)
            out = img_n * scale
            results[key] = out.astype(np.uint8) if img.dtype == np.uint8 else out
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, gamma_range={self.gamma_range})"


@PIPELINES.register_module()
class RandomLogTransform:
    """Log compression transform with random gain."""

    def __init__(self, p: float = 0.2, gain_range: tuple[float, float] = (1.0, 50.0), eps: float = 1e-6):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if len(gain_range) != 2 or gain_range[0] <= 0 or gain_range[1] < gain_range[0]:
            raise ValueError("gain_range must be (min, max) with 0 < min <= max")
        self.p = float(p)
        self.gain_range = (float(gain_range[0]), float(gain_range[1]))
        self.eps = float(eps)

    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.p:
            return results

        gain = float(np.random.uniform(self.gain_range[0], self.gain_range[1]))
        denom = float(np.log1p(gain) + self.eps)
        for key in _get_img_fields(results):
            img = results[key]
            scale = _infer_scale(img)
            img_f = img.astype(np.float32)
            img_n = np.clip(img_f / scale, 0.0, 1.0)
            img_n = np.log1p(gain * img_n) / denom
            out = img_n * scale
            results[key] = out.astype(np.uint8) if img.dtype == np.uint8 else out
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, gain_range={self.gain_range})"

