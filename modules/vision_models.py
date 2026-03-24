from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

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


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configure_torch_cache() -> None:
    # 让 torchvision/torch hub 的权重缓存落到项目目录，方便复用与迁移
    # 若用户已经手动设置 TORCH_HOME，则尊重用户配置。
    if os.environ.get("TORCH_HOME"):
        return
    cache_dir = _PROJECT_ROOT / ".cache" / "torch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)


def _get_convnext_tiny(num_classes: int, *, pretrained: bool = True) -> tuple[nn.Module, Any]:
    _configure_torch_cache()

    from torchvision.models import convnext_tiny
    from torchvision.models import ConvNeXt_Tiny_Weights

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    # 替换分类头
    if not hasattr(model, "classifier"):
        raise RuntimeError("unexpected convnext_tiny structure")
    last = model.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("unexpected convnext_tiny classifier head")
    in_features = last.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model, weights


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


@dataclass
class TorchTransferClassifier(BaseModule):
    """迁移学习分类器（PyTorch + torchvision）。

    mode:
    - head: 冻结 backbone，仅训练分类头
    - partial: 解冻最后一个 stage + 分类头
    """

    num_classes: int
    backbone: Literal["convnext_tiny"] = "convnext_tiny"
    mode: Literal["head", "partial"] = "head"
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    seed: int = 0
    device: Optional[str] = None
    pretrained: bool = True

    name: str = "TorchTransferClassifier"

    _model: nn.Module | None = None
    _device: torch.device | None = None
    _weights_meta: Any = None

    def __post_init__(self) -> None:
        _set_seed(self.seed)
        self._device = torch.device(self.device) if self.device else _default_device()

        if self.backbone != "convnext_tiny":
            raise ValueError(f"unsupported backbone: {self.backbone}")
        model, weights = _get_convnext_tiny(self.num_classes, pretrained=self.pretrained)
        self._weights_meta = weights

        if self.mode == "head":
            _freeze_all(model)
            _unfreeze_module(model.classifier)
        elif self.mode == "partial":
            _freeze_all(model)
            # convnext：解冻最后一个 stage + classifier
            if hasattr(model, "features") and isinstance(model.features, nn.Sequential) and len(model.features) > 0:
                _unfreeze_module(model.features[-1])
            _unfreeze_module(model.classifier)
        else:
            raise ValueError(f"unsupported mode: {self.mode}")

        self._model = model.to(self._device)

    def fit(
        self,
        train_loader: Any,
        y: Any = None,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_loss: Optional[Callable[[int, int, float], None]] = None,
        log_every: int = 10,
        **kwargs: Any,
    ) -> "TorchTransferClassifier":
        if self._model is None or self._device is None:
            raise RuntimeError("model not initialized")

        self._model.train()
        opt = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        n_batches = 0
        try:
            n_batches = int(len(train_loader))
        except Exception:
            n_batches = 0

        total_steps = self.epochs * n_batches if n_batches > 0 else 0
        global_step = 0

        for epoch in range(self.epochs):
            if on_progress:
                on_progress(f"epoch {epoch + 1}/{self.epochs}", global_step / total_steps if total_steps else 0.0)

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self._device)
                targets = targets.to(self._device)

                opt.zero_grad(set_to_none=True)
                logits = self._model(images)
                loss = loss_fn(logits, targets)
                loss.backward()
                opt.step()

                global_step += 1
                if on_loss and (global_step == 1 or (log_every > 0 and global_step % log_every == 0)):
                    on_loss(epoch, global_step, float(loss.detach().to("cpu").item()))

                if on_progress and total_steps:
                    on_progress(
                        f"训练中：epoch {epoch + 1}/{self.epochs} | batch {batch_idx + 1}/{n_batches}",
                        min(1.0, global_step / total_steps),
                    )

        return self

    @torch.no_grad()
    def predict(self, test_loader: Any, **kwargs: Any) -> np.ndarray:
        if self._model is None or self._device is None:
            raise RuntimeError("model not initialized")
        self._model.eval()

        preds: list[np.ndarray] = []
        for images, _targets in test_loader:
            images = images.to(self._device)
            logits = self._model(images)
            y_pred = torch.argmax(logits, dim=1).to("cpu").numpy()
            preds.append(y_pred)
        return np.concatenate(preds, axis=0)

    def save(self, path: str | Path) -> str:
        if self._model is None:
            raise RuntimeError("model not initialized")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._model.state_dict(),
            "backbone": self.backbone,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "seed": self.seed,
        }
        torch.save(payload, path)
        return str(path)

    def label(self) -> str:
        d = {
            "backbone": self.backbone,
            "mode": self.mode,
            "pretrained": self.pretrained,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": str(self._device) if self._device else None,
        }
        return f"{self.__class__.__name__}({json.dumps(d, ensure_ascii=False, sort_keys=True)})"


@dataclass
class FrozenEmbeddingClassifier(BaseModule):
    """冻结预训练表征 + 线性分类器（sklearn）。"""

    num_classes: int
    backbone: Literal["convnext_tiny"] = "convnext_tiny"
    head: Literal["logreg"] = "logreg"
    seed: int = 0
    device: Optional[str] = None
    pretrained: bool = True

    name: str = "FrozenEmbeddingClassifier"

    _device: torch.device | None = None
    _embed_model: nn.Module | None = None
    _clf: Any = None

    def __post_init__(self) -> None:
        _set_seed(self.seed)
        self._device = torch.device(self.device) if self.device else _default_device()

        if self.backbone != "convnext_tiny":
            raise ValueError(f"unsupported backbone: {self.backbone}")

        model, _weights = _get_convnext_tiny(num_classes=self.num_classes, pretrained=self.pretrained)
        # 作为 embedding extractor：去掉分类头，只用 features + avgpool
        if not hasattr(model, "features"):
            raise RuntimeError("unexpected convnext_tiny structure")
        embed = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        _freeze_all(embed)
        self._embed_model = embed.to(self._device).eval()

        if self.head != "logreg":
            raise ValueError(f"unsupported head: {self.head}")

        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=self.seed,
        )

    @torch.no_grad()
    def _embed(self, loader: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._embed_model is None or self._device is None:
            raise RuntimeError("embed model not initialized")

        feats: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for images, targets in loader:
            images = images.to(self._device)
            f = self._embed_model(images).to("cpu").numpy().astype(np.float32, copy=False)
            feats.append(f)
            ys.append(np.asarray(targets, dtype=np.int64))
        return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)

    def fit(self, train_loader: Any, y: Any = None, **kwargs: Any) -> "FrozenEmbeddingClassifier":
        Xemb, yemb = self._embed(train_loader)
        self._clf.fit(Xemb, yemb)
        return self

    def predict(self, test_loader: Any, **kwargs: Any) -> np.ndarray:
        Xemb, _y = self._embed(test_loader)
        return self._clf.predict(Xemb)

    def save(self, path: str | Path) -> str:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backbone": self.backbone,
                "head": self.head,
                "seed": self.seed,
                "clf": self._clf,
            },
            path,
        )
        return str(path)

    def label(self) -> str:
        d = {
            "backbone": self.backbone,
            "head": self.head,
            "pretrained": self.pretrained,
            "seed": self.seed,
            "device": str(self._device) if self._device else None,
        }
        return f"{self.__class__.__name__}({json.dumps(d, ensure_ascii=False, sort_keys=True)})"
