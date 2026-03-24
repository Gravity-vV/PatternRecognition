from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from modules.base import BaseModule


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _configure_torch_cache() -> None:
    # 复用 modules.vision_models 的缓存策略（保持只下载一次）
    try:
        from modules.vision_models import _configure_torch_cache as _cfg  # type: ignore

        _cfg()
    except Exception:
        # 最差情况下不阻塞功能
        return


@dataclass
class ConvNeXtTinyEmbeddingExtractor(BaseModule):
    """ConvNeXt-Tiny 冻结 embedding 提取器。

    输入：torch DataLoader（images, targets），images 需已做 ToDtype/Normalize 等预处理。
    输出：numpy features（N×D）。
    """

    seed: int = 0
    device: Optional[str] = None
    pretrained: bool = True

    name: str = "ConvNeXtTinyEmbeddingExtractor"

    _device: torch.device | None = None
    _model: nn.Module | None = None

    def __post_init__(self) -> None:
        _set_seed(self.seed)
        self._device = torch.device(self.device) if self.device else _default_device()
        _configure_torch_cache()

        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if self.pretrained else None
        base = convnext_tiny(weights=weights)
        if not hasattr(base, "features"):
            raise RuntimeError("unexpected convnext_tiny structure")

        embed = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        for p in embed.parameters():
            p.requires_grad = False
        self._model = embed.to(self._device).eval()

    @torch.no_grad()
    def transform(self, X: Any, **kwargs: Any) -> np.ndarray:
        if self._model is None or self._device is None:
            raise RuntimeError("model not initialized")

        # 约定 X 是 DataLoader
        feats: list[np.ndarray] = []
        for images, _targets in X:
            images = images.to(self._device)
            f = self._model(images).to("cpu").numpy().astype(np.float32, copy=False)
            feats.append(f)
        if not feats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(feats, axis=0)
