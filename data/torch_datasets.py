from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TorchDatasetBundle:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: Optional[list[str]] = None


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_oxford_pets(
    *,
    root: str | Path = "data/oxford_pets",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 0,
):
    """Oxford-IIIT Pets（torchvision）。

    split 策略：使用 torchvision 预定义的 trainval/test，保证稳定可复现。
    seed 用于 DataLoader shuffle 与训练初始化。
    """

    from torchvision.datasets import OxfordIIITPet

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # transforms 由 runner 注入；这里先返回 dataset 句柄
    train_ds = OxfordIIITPet(root=str(root), split="trainval", target_types="category", download=True)
    test_ds = OxfordIIITPet(root=str(root), split="test", target_types="category", download=True)

    # 类别名：dataset 内部有 classes
    class_names = None
    if hasattr(train_ds, "classes"):
        class_names = list(getattr(train_ds, "classes"))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    num_classes = 37
    return TorchDatasetBundle(
        name="oxford_pets",
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        class_names=class_names,
    )
