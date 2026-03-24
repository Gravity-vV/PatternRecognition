"""方法注册系统：预处理、特征提取、分类器的统一注册与组合。

每个环节的方法都可以自由组合，只需保证输入输出格式兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from modules.base import BaseModule


@dataclass
class MethodInfo:
    """方法元信息"""
    name: str  # 唯一标识
    label: str  # 显示名称
    category: Literal["preprocessing", "feature", "classifier"]
    factory: Callable[..., BaseModule]  # 工厂函数
    # 数据类型约束：用于组合校验
    # - numpy_image: numpy 图像/数组（如 N×H×W 或 N×H×W×C）
    # - torch_image: torch DataLoader 产出的 image tensor
    # - vector: numpy 特征向量（N×D）
    input_kind: Literal["numpy_image", "torch_image", "vector"]
    output_kind: Literal["numpy_image", "torch_image", "vector", "label"]

    default_params: dict[str, Any] = field(default_factory=dict)


class MethodRegistry:
    """方法注册表"""

    def __init__(self) -> None:
        self._methods: dict[str, MethodInfo] = {}

    def register(
        self,
        name: str,
        label: str,
        category: Literal["preprocessing", "feature", "classifier"],
        factory: Callable[..., BaseModule],
        default_params: Optional[dict[str, Any]] = None,
        *,
        input_kind: Literal["numpy_image", "torch_image", "vector"],
        output_kind: Literal["numpy_image", "torch_image", "vector", "label"],
    ) -> None:
        self._methods[name] = MethodInfo(
            name=name,
            label=label,
            category=category,
            factory=factory,
            default_params=default_params or {},
            input_kind=input_kind,
            output_kind=output_kind,
        )

    def get(self, name: str) -> MethodInfo:
        if name not in self._methods:
            raise KeyError(f"方法 '{name}' 未注册")
        return self._methods[name]

    def list_by_category(self, category: Literal["preprocessing", "feature", "classifier"]) -> list[MethodInfo]:
        return [m for m in self._methods.values() if m.category == category]

    def create(self, name: str, **override_params: Any) -> BaseModule:
        info = self.get(name)
        params = {**info.default_params, **override_params}
        return info.factory(**params)

    def all_names(self) -> list[str]:
        return list(self._methods.keys())


# 全局注册表实例
REGISTRY = MethodRegistry()


def _register_all() -> None:
    """注册所有可用方法"""

    # ========== 预处理方法 ==========
    from modules.preprocessing import Normalizer, GaussianFilter, ResizeNumpyImage

    # Normalizer 同时可用于图像（逐元素）与向量特征（逐维/全局）
    REGISTRY.register(
        name="normalizer_zscore_image",
        label="Z-Score标准化（image）",
        category="preprocessing",
        factory=lambda **kw: Normalizer(method="zscore", **kw),
        default_params={"axis": None},
        input_kind="numpy_image",
        output_kind="numpy_image",
    )

    REGISTRY.register(
        name="normalizer_minmax_image",
        label="MinMax归一化（image）",
        category="preprocessing",
        factory=lambda **kw: Normalizer(method="minmax", **kw),
        default_params={"axis": None},
        input_kind="numpy_image",
        output_kind="numpy_image",
    )

    REGISTRY.register(
        name="normalizer_zscore_vec",
        label="Z-Score标准化（vector）",
        category="preprocessing",
        factory=lambda **kw: Normalizer(method="zscore", **kw),
        default_params={"axis": 0},
        input_kind="vector",
        output_kind="vector",
    )

    REGISTRY.register(
        name="normalizer_minmax_vec",
        label="MinMax归一化（vector）",
        category="preprocessing",
        factory=lambda **kw: Normalizer(method="minmax", **kw),
        default_params={"axis": 0},
        input_kind="vector",
        output_kind="vector",
    )

    REGISTRY.register(
        name="gaussian_filter",
        label="高斯滤波",
        category="preprocessing",
        factory=lambda **kw: GaussianFilter(**kw),
        default_params={"sigma": 0.8},
        input_kind="numpy_image",
        output_kind="numpy_image",
    )

    REGISTRY.register(
        name="resize_numpy_128",
        label="Resize(128)（numpy）",
        category="preprocessing",
        factory=lambda **kw: ResizeNumpyImage(size=128, **kw),
        default_params={},
        input_kind="numpy_image",
        output_kind="numpy_image",
    )

    REGISTRY.register(
        name="resize_numpy_224",
        label="Resize(224)（numpy）",
        category="preprocessing",
        factory=lambda **kw: ResizeNumpyImage(size=224, **kw),
        default_params={},
        input_kind="numpy_image",
        output_kind="numpy_image",
    )

    # ========== 特征提取方法 ==========
    from modules.feature_extraction import (
        ColorHistogramExtractor,
        HOGExtractor,
        HandcraftedFusionExtractor,
        LBPExtractor,
        PCAExtractor,
        PixelFlattenExtractor,
    )

    REGISTRY.register(
        name="pixel_flatten",
        label="像素展平",
        category="feature",
        factory=lambda **kw: PixelFlattenExtractor(**kw),
        default_params={},
        input_kind="numpy_image",
        output_kind="vector",
    )

    REGISTRY.register(
        name="pca",
        label="PCA降维",
        category="feature",
        factory=lambda **kw: PCAExtractor(**kw),
        default_params={"n_components": 128, "whiten": False},
        input_kind="vector",
        output_kind="vector",
    )

    REGISTRY.register(
        name="hog",
        label="HOG特征",
        category="feature",
        factory=lambda **kw: HOGExtractor(**kw),
        default_params={"pixels_per_cell": (8, 8), "cells_per_block": (2, 2), "orientations": 9},
        input_kind="numpy_image",
        output_kind="vector",
    )

    REGISTRY.register(
        name="lbp",
        label="LBP纹理特征",
        category="feature",
        factory=lambda **kw: LBPExtractor(**kw),
        default_params={"P": 8, "R": 1, "method": "uniform"},
        input_kind="numpy_image",
        output_kind="vector",
    )

    REGISTRY.register(
        name="color_hist",
        label="颜色直方图特征",
        category="feature",
        factory=lambda **kw: ColorHistogramExtractor(**kw),
        default_params={"bins": 16, "color_space": "hsv"},
        input_kind="numpy_image",
        output_kind="vector",
    )

    REGISTRY.register(
        name="handcrafted_fusion",
        label="融合手工特征（HOG+LBP+ColorHist）",
        category="feature",
        factory=lambda **kw: HandcraftedFusionExtractor(**kw),
        default_params={},
        input_kind="numpy_image",
        output_kind="vector",
    )

    # ========== 分类器方法 ==========
    from modules.classifiers import LogisticRegressionClassifier, RandomForestClassifier, SVMClassifier

    REGISTRY.register(
        name="random_forest",
        label="随机森林",
        category="classifier",
        factory=lambda **kw: RandomForestClassifier(**kw),
        default_params={"n_estimators": 200, "n_jobs": -1},
        input_kind="vector",
        output_kind="label",
    )

    REGISTRY.register(
        name="svm_linear",
        label="线性SVM",
        category="classifier",
        factory=lambda **kw: SVMClassifier(kernel="linear", **kw),
        default_params={"C": 1.0},
        input_kind="vector",
        output_kind="label",
    )

    REGISTRY.register(
        name="svm_rbf",
        label="RBF核SVM",
        category="classifier",
        factory=lambda **kw: SVMClassifier(kernel="rbf", **kw),
        default_params={"C": 1.0, "gamma": "scale"},
        input_kind="vector",
        output_kind="label",
    )

    REGISTRY.register(
        name="logreg",
        label="逻辑回归",
        category="classifier",
        factory=lambda **kw: LogisticRegressionClassifier(**kw),
        default_params={"max_iter": 2000},
        input_kind="vector",
        output_kind="label",
    )

    # ========== 深度特征提取/分类（同样按三段式可组合） ==========
    from modules.torch_feature_extraction import ConvNeXtTinyEmbeddingExtractor
    from modules.vision_models import TorchTransferClassifier
    from modules.vision_preprocessing import ImageNetNormalizer, Resize, TorchAugmentor

    # torch 图像预处理（作为可组合的 preprocessing）
    REGISTRY.register(
        name="resize_224",
        label="Resize(224)",
        category="preprocessing",
        factory=lambda **kw: Resize(size=224, **kw),
        default_params={},
        input_kind="torch_image",
        output_kind="torch_image",
    )

    REGISTRY.register(
        name="torch_augment_224",
        label="Augment(224)",
        category="preprocessing",
        factory=lambda **kw: TorchAugmentor(size=224, **kw),
        default_params={},
        input_kind="torch_image",
        output_kind="torch_image",
    )

    REGISTRY.register(
        name="imagenet_normalize",
        label="ImageNetNormalize",
        category="preprocessing",
        factory=lambda **kw: ImageNetNormalizer(**kw),
        default_params={},
        input_kind="torch_image",
        output_kind="torch_image",
    )

    REGISTRY.register(
        name="convnext_tiny_embedding",
        label="ConvNeXt-Tiny Embedding",
        category="feature",
        factory=lambda **kw: ConvNeXtTinyEmbeddingExtractor(**kw),
        default_params={"pretrained": True},
        input_kind="torch_image",
        output_kind="vector",
    )

    # 端到端训练：作为“分类器模型”保留（此时 feature 应为空）
    REGISTRY.register(
        name="convnext_tiny_transfer",
        label="ConvNeXt-Tiny Transfer",
        category="classifier",
        factory=lambda **kw: TorchTransferClassifier(**kw),
        default_params={"backbone": "convnext_tiny", "mode": "head", "epochs": 1, "lr": 1e-3, "pretrained": True},
        input_kind="torch_image",
        output_kind="label",
    )


# 启动时自动注册
_register_all()


def get_registry() -> MethodRegistry:
    """获取全局注册表"""
    return REGISTRY


def list_preprocessing() -> list[MethodInfo]:
    """列出所有预处理方法"""
    return REGISTRY.list_by_category("preprocessing")


def list_feature_extraction() -> list[MethodInfo]:
    """列出所有特征提取方法"""
    return REGISTRY.list_by_category("feature")


def list_classifiers() -> list[MethodInfo]:
    """列出所有分类器方法"""
    return REGISTRY.list_by_category("classifier")
