from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from modules.base import BaseModule


def _as_3chw(x: Any) -> Any:
    # torchvision transforms 可以处理 PIL.Image 或 torch.Tensor。
    return x


@dataclass
class Resize(BaseModule):
    size: int = 224
    name: str = "Resize"

    def as_transform(self):
        from torchvision.transforms.v2 import InterpolationMode, Resize as _Resize

        return _Resize((self.size, self.size), interpolation=InterpolationMode.BILINEAR, antialias=True)

    def transform(self, X: Any, **kwargs: Any) -> Any:
        return self.as_transform()(_as_3chw(X))


@dataclass
class ImageNetNormalizer(BaseModule):
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    name: str = "ImageNetNormalizer"

    def as_transform(self):
        from torchvision.transforms.v2 import Normalize

        return Normalize(mean=self.mean, std=self.std)

    def transform(self, X: Any, **kwargs: Any) -> Any:
        return self.as_transform()(_as_3chw(X))


@dataclass
class TorchAugmentor(BaseModule):
    size: int = 224
    hflip_p: float = 0.5
    name: str = "TorchAugmentor"

    def as_transform(self):
        from torchvision.transforms.v2 import RandomHorizontalFlip, RandomResizedCrop

        return [
            RandomResizedCrop((self.size, self.size), scale=(0.75, 1.0)),
            RandomHorizontalFlip(p=self.hflip_p),
        ]

    def transform(self, X: Any, **kwargs: Any) -> Any:
        # 作为单张图像的 transform 使用时，按顺序应用。
        out = _as_3chw(X)
        for t in self.as_transform():
            out = t(out)
        return out


def build_vision_transforms(
    *,
    resize: Resize,
    normalizer: ImageNetNormalizer,
    augmentor: Optional[TorchAugmentor] = None,
    train: bool,
):
    from torchvision.transforms.v2 import Compose, ToImage, ToDtype

    ops: list[Any] = []
    if train and augmentor is not None:
        ops.extend(augmentor.as_transform())
    else:
        ops.append(resize.as_transform())

    # 统一为 float32 tensor in [0,1]
    ops.append(ToImage())
    ops.append(ToDtype(torch.float32, scale=True))
    ops.append(normalizer.as_transform())

    return Compose(ops)


def describe_preprocess_chain(mods: Iterable[BaseModule]) -> str:
    mods = list(mods)
    if not mods:
        return "none"
    return "+".join(m.__class__.__name__ for m in mods)
