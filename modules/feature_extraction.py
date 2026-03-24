from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from modules.base import BaseModule


def _as_numpy(X: Any) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)


def _to_grayscale(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 1:
        return x[..., 0]
    if x.shape[-1] == 3:
        return (
            x[..., 0] * 0.33333334 + x[..., 1] * 0.33333334 + x[..., 2] * 0.33333334
        )
    raise ValueError(f"Unsupported channel count: {x.shape[-1]}")


def _prep_hog_image(img: np.ndarray, transform_sqrt: bool) -> np.ndarray:
    # transform_sqrt=True 时会对图像做 sqrt，负值会产生 NaN。
    if transform_sqrt:
        return np.maximum(img, 0.0)
    return img


def _to_unit_range(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.size == 0:
        return x
    if float(np.nanmax(x)) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


@dataclass
class ColorHistogramExtractor(BaseModule):
    """颜色直方图（传统特征）。

    - RGB 输入时可选择在 HSV 空间统计直方图。
    - 灰度输入则退化为亮度直方图。
    """

    bins: int = 16
    color_space: str = "hsv"  # 'rgb' | 'hsv'

    name: str = "ColorHistogramExtractor"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        from skimage.color import rgb2hsv

        Xn = _as_numpy(X)

        def _hist_1d(v: np.ndarray) -> np.ndarray:
            h, _ = np.histogram(v, bins=self.bins, range=(0.0, 1.0))
            h = h.astype(np.float32, copy=False)
            s = float(h.sum())
            return h / s if s > 0 else h

        def _feat_one(img: np.ndarray) -> np.ndarray:
            img = _to_unit_range(img)

            if img.ndim == 2:
                return _hist_1d(img.reshape(-1))

            if img.ndim == 3:
                if img.shape[-1] == 1:
                    return _hist_1d(img[..., 0].reshape(-1))
                if img.shape[-1] != 3:
                    raise ValueError(f"Unsupported channel count: {img.shape[-1]}")

                if self.color_space.lower() == "hsv":
                    img_cs = rgb2hsv(img)
                elif self.color_space.lower() == "rgb":
                    img_cs = img
                else:
                    raise ValueError(f"Unknown color_space: {self.color_space}")

                feats = [_hist_1d(img_cs[..., c].reshape(-1)) for c in range(3)]
                return np.concatenate(feats, axis=0)

            raise ValueError(f"Unsupported image shape: {img.shape}")

        if Xn.ndim == 2:
            return _feat_one(Xn)[None, :]

        if Xn.ndim == 3:
            # N×H×W 或 H×W×C
            if Xn.shape[-1] in (1, 3) and Xn.shape[0] not in (1, 3):
                return _feat_one(Xn)[None, :]
            return np.stack([_feat_one(img) for img in Xn], axis=0)

        if Xn.ndim == 4:
            return np.stack([_feat_one(img) for img in Xn], axis=0)

        raise ValueError(f"Unsupported input shape: {Xn.shape}")


@dataclass
class LBPExtractor(BaseModule):
    """LBP（Local Binary Pattern）纹理特征：输出 LBP 直方图。"""

    P: int = 8
    R: int = 1
    method: str = "uniform"

    name: str = "LBPExtractor"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        from skimage.feature import local_binary_pattern

        Xn = _as_numpy(X)
        n_bins = self.P + 2 if self.method == "uniform" else 2 ** self.P

        def _hist(v: np.ndarray) -> np.ndarray:
            h, _ = np.histogram(v, bins=n_bins, range=(0, n_bins))
            h = h.astype(np.float32, copy=False)
            s = float(h.sum())
            return h / s if s > 0 else h

        def _feat_one(img: np.ndarray) -> np.ndarray:
            img = _to_unit_range(img)
            if img.ndim == 3:
                img = _to_grayscale(img)
            if img.ndim != 2:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            # skimage 推荐对整数图像做 LBP，这里把 [0,1] 灰度映射到 uint8。
            img_u8 = np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8, copy=False)
            lbp = local_binary_pattern(img_u8, P=self.P, R=self.R, method=self.method)
            return _hist(lbp.reshape(-1))

        if Xn.ndim == 2:
            return _feat_one(Xn)[None, :]

        if Xn.ndim == 3:
            # N×H×W 或 H×W×C
            if Xn.shape[-1] in (1, 3) and Xn.shape[0] not in (1, 3):
                return _feat_one(Xn)[None, :]
            return np.stack([_feat_one(img) for img in Xn], axis=0)

        if Xn.ndim == 4:
            return np.stack([_feat_one(img) for img in Xn], axis=0)

        raise ValueError(f"Unsupported input shape: {Xn.shape}")


@dataclass
class PixelFlattenExtractor(BaseModule):
    """像素展平：将图像/数组展平成向量（N×D）。"""

    name: str = "PixelFlattenExtractor"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        if Xn.ndim < 2:
            raise ValueError(f"Unsupported input shape: {Xn.shape}")
        if Xn.ndim == 2:
            # 单张灰度图 H×W -> 1×D
            return Xn.reshape(1, -1)
        # 其余情况：N×... -> N×D
        return Xn.reshape(Xn.shape[0], -1)


@dataclass
class HOGExtractor(BaseModule):
    """HOG 特征提取（scikit-image）。"""

    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True

    name: str = "HOGExtractor"

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        from skimage.feature import hog

        Xn = _as_numpy(X).astype(np.float32, copy=False)

        if Xn.ndim == 2:
            img = _prep_hog_image(Xn, self.transform_sqrt)
            feats = hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                transform_sqrt=self.transform_sqrt,
                feature_vector=True,
            )
            return np.asarray(feats, dtype=np.float32)[None, :]

        if Xn.ndim == 3:
            # N×H×W 或 H×W×C
            if Xn.shape[-1] in (1, 3) and Xn.shape[0] not in (1, 3):
                img = _prep_hog_image(_to_grayscale(Xn), self.transform_sqrt)
                feats = hog(
                    img,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm,
                    transform_sqrt=self.transform_sqrt,
                    feature_vector=True,
                )
                return np.asarray(feats, dtype=np.float32)[None, :]

            feats_list: list[np.ndarray] = []
            for img in Xn:
                img = _prep_hog_image(img, self.transform_sqrt)
                feats = hog(
                    img,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm,
                    transform_sqrt=self.transform_sqrt,
                    feature_vector=True,
                )
                feats_list.append(np.asarray(feats, dtype=np.float32))
            return np.stack(feats_list, axis=0)

        if Xn.ndim == 4:
            # N×H×W×C
            Xg = _prep_hog_image(_to_grayscale(Xn), self.transform_sqrt)
            feats_list: list[np.ndarray] = []
            for img in Xg:
                feats = hog(
                    img,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm,
                    transform_sqrt=self.transform_sqrt,
                    feature_vector=True,
                )
                feats_list.append(np.asarray(feats, dtype=np.float32))
            return np.stack(feats_list, axis=0)

        raise ValueError(f"Unsupported input shape: {Xn.shape}")


@dataclass
class PCAExtractor(BaseModule):
    """PCA 降维（sklearn）。"""

    n_components: Optional[int] = None
    whiten: bool = False
    random_state: Optional[int] = 0

    name: str = "PCAExtractor"

    def __post_init__(self) -> None:
        from sklearn.decomposition import PCA

        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "PCAExtractor":
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        X2d = Xn.reshape(Xn.shape[0], -1) if Xn.ndim > 2 else Xn

        # 允许在不同特征维度下复用同一个 pca 模块：当 n_components 超过可用维度时自动截断。
        # 例如：LBP(10维) 或 color_hist(<=48维) 上，默认 n_components=128 会直接报错。
        if self.n_components is not None:
            max_allowed = int(min(X2d.shape[0], X2d.shape[1]))
            target = int(min(self.n_components, max_allowed))
            # sklearn 要求 n_components >= 1
            target = max(1, target)
            if getattr(self._pca, "n_components", None) != target:
                from sklearn.decomposition import PCA

                self._pca = PCA(
                    n_components=target,
                    whiten=self.whiten,
                    random_state=self.random_state,
                )
        self._pca.fit(X2d)
        return self

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        X2d = Xn.reshape(Xn.shape[0], -1) if Xn.ndim > 2 else Xn
        return self._pca.transform(X2d).astype(np.float32, copy=False)


@dataclass
class HandcraftedFusionExtractor(BaseModule):
    """融合手工特征：HOG + LBP + Color Histogram。

    目的：在不引入深度模型的前提下，尽量榨干传统特征的上限。
    输出为拼接后的向量特征（N×D）。
    """

    # HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    hog_transform_sqrt: bool = True

    # LBP
    lbp_P: int = 8
    lbp_R: int = 1
    lbp_method: str = "uniform"

    # Color hist
    color_bins: int = 16
    color_space: str = "hsv"

    name: str = "HandcraftedFusionExtractor"

    def __post_init__(self) -> None:
        self._hog = HOGExtractor(
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm=self.hog_block_norm,
            transform_sqrt=self.hog_transform_sqrt,
        )
        self._lbp = LBPExtractor(P=self.lbp_P, R=self.lbp_R, method=self.lbp_method)
        self._color = ColorHistogramExtractor(bins=self.color_bins, color_space=self.color_space)

    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        hog_f = self._hog.transform(X)
        lbp_f = self._lbp.transform(X)
        col_f = self._color.transform(X)

        if hog_f.shape[0] != lbp_f.shape[0] or hog_f.shape[0] != col_f.shape[0]:
            raise ValueError("Fusion feature batch size mismatch")
        return np.concatenate([hog_f, lbp_f, col_f], axis=1).astype(np.float32, copy=False)
