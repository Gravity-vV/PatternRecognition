from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from modules.base import BaseModule


def _as_numpy(X: Any) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)


@dataclass
class Normalizer(BaseModule):
    """归一化：Min-Max 或 Z-Score。"""

    method: Literal["minmax", "zscore"] = "zscore"
    eps: float = 1e-12
    axis: Optional[int] = None

    name: str = "Normalizer"

    _min: np.ndarray | None = None
    _max: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "Normalizer":
        Xn = _as_numpy(X).astype(np.float32, copy=False)

        if self.method == "minmax":
            self._min = Xn.min(axis=self.axis, keepdims=True)
            self._max = Xn.max(axis=self.axis, keepdims=True)
        elif self.method == "zscore":
            self._mean = Xn.mean(axis=self.axis, keepdims=True)
            self._std = Xn.std(axis=self.axis, keepdims=True)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)

        if self.method == "minmax":
            if self._min is None or self._max is None:
                raise RuntimeError("Normalizer(minmax) must be fit before transform")
            denom = np.maximum(self._max - self._min, self.eps)
            return (Xn - self._min) / denom

        if self.method == "zscore":
            if self._mean is None or self._std is None:
                raise RuntimeError("Normalizer(zscore) must be fit before transform")
            denom = np.maximum(self._std, self.eps)
            return (Xn - self._mean) / denom

        raise ValueError(f"Unknown method: {self.method}")


@dataclass
class GaussianFilter(BaseModule):
    """高斯滤波（降噪）。

    输入支持：
    - 单张图：H×W 或 H×W×C
    - 批量：N×H×W 或 N×H×W×C
    """

    sigma: float = 1.0
    truncate: float = 4.0

    name: str = "GaussianFilter"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        Xn = _as_numpy(X).astype(np.float32, copy=False)

        if Xn.ndim not in (2, 3, 4):
            raise ValueError(f"Unsupported input shape: {Xn.shape}")

        if Xn.ndim == 2:
            return gaussian_filter(Xn, sigma=self.sigma, truncate=self.truncate)

        if Xn.ndim == 3:
            # N×H×W  或 H×W×C
            if Xn.shape[-1] in (1, 3) and Xn.shape[0] not in (1, 3):
                sigmas = (self.sigma, self.sigma, 0.0)
                return gaussian_filter(Xn, sigma=sigmas, truncate=self.truncate)

            sigmas = (0.0, self.sigma, self.sigma)
            return gaussian_filter(Xn, sigma=sigmas, truncate=self.truncate)

        sigmas = (0.0, self.sigma, self.sigma, 0.0)
        return gaussian_filter(Xn, sigma=sigmas, truncate=self.truncate)


@dataclass
class ResizeNumpyImage(BaseModule):
    """Resize（numpy 图像）。

    输入支持：
    - 单张图：H×W 或 H×W×C
    - 批量：N×H×W 或 N×H×W×C
    """

    size: int = 128
    preserve_range: bool = True
    antialias: bool = True

    name: str = "ResizeNumpyImage"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        from skimage.transform import resize

        Xn = _as_numpy(X).astype(np.float32, copy=False)
        out_shape_hw = (self.size, self.size)

        def _resize_one(img: np.ndarray) -> np.ndarray:
            # 若已经是目标尺寸，直接返回（避免重复插值带来的耗时和数值误差）。
            if img.shape[0] == self.size and img.shape[1] == self.size:
                return img.astype(np.float32, copy=False)
            if img.ndim == 2:
                r = resize(
                    img,
                    out_shape_hw,
                    order=1,
                    mode="reflect",
                    anti_aliasing=self.antialias,
                    preserve_range=self.preserve_range,
                )
                return np.asarray(r, dtype=np.float32)
            if img.ndim == 3:
                if img.shape[0] == self.size and img.shape[1] == self.size:
                    return img.astype(np.float32, copy=False)
                r = resize(
                    img,
                    (*out_shape_hw, img.shape[-1]),
                    order=1,
                    mode="reflect",
                    anti_aliasing=self.antialias,
                    preserve_range=self.preserve_range,
                )
                return np.asarray(r, dtype=np.float32)
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if Xn.ndim == 2:
            return _resize_one(Xn)

        if Xn.ndim == 3:
            # N×H×W  或 H×W×C
            if Xn.shape[-1] in (1, 3) and Xn.shape[0] not in (1, 3):
                return _resize_one(Xn)
            # N×H×W
            if Xn.shape[1] == self.size and Xn.shape[2] == self.size:
                return Xn.astype(np.float32, copy=False)
            return np.stack([_resize_one(img) for img in Xn], axis=0)

        if Xn.ndim == 4:
            # N×H×W×C
            if Xn.shape[1] == self.size and Xn.shape[2] == self.size:
                return Xn.astype(np.float32, copy=False)
            return np.stack([_resize_one(img) for img in Xn], axis=0)

        raise ValueError(f"Unsupported input shape: {Xn.shape}")
