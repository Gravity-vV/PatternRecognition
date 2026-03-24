from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NumpyDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: Optional[list[str]] = None


def load_oxford_pets_numpy(
    *,
    root: str | Path = "data/oxford_pets",
    image_size: int = 64,
    seed: int = 0,
) -> NumpyDataset:
    """把 Pets 预处理成 numpy（RGB、固定尺寸），供传统流水线使用。

    说明：
    - 这里保留 RGB（而不是灰度），以便传统方法也能使用颜色类特征（如颜色直方图）。
    - 归一化（z-score/minmax）由 pipeline 的 preprocessing 模块负责。
    """

    from torchvision.datasets import OxfordIIITPet
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import Compose, Resize

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 固定 split：trainval/test
    train_ds = OxfordIIITPet(root=str(root), split="trainval", target_types="category", download=True)
    test_ds = OxfordIIITPet(root=str(root), split="test", target_types="category", download=True)

    class_names = None
    if hasattr(train_ds, "classes"):
        class_names = list(getattr(train_ds, "classes"))

    # 只做 resize（传统 pipeline 里的归一化由模块做）
    tf = Compose([Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True)])

    def _to_np(ds):
        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        for img, y in ds:
            img = tf(img)
            arr = np.asarray(img, dtype=np.float32)
            # 统一到 [0, 1]，便于传统特征（HOG/颜色直方图等）与归一化模块处理
            if arr.max() > 1.5:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            X_list.append(arr)
            y_list.append(int(y))
        X = np.stack(X_list, axis=0)
        y = np.asarray(y_list, dtype=np.int64)
        return X, y

    # seed 在这里不影响 split，但保留字段语义一致
    _ = seed

    X_train, y_train = _to_np(train_ds)
    X_test, y_test = _to_np(test_ds)

    return NumpyDataset(
        name="oxford_pets",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )
