from __future__ import annotations

from abc import ABC
from typing import Any


class BaseModule(ABC):
    """所有模块的统一基类。

    约定：
    - 预处理/特征提取：实现 transform
    - 分类器：实现 predict

    fit 默认返回 self，方便链式调用。
    """

    name: str = "BaseModule"

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "BaseModule":
        return self

    def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        self.fit(X, y=y, **kwargs)
        return self.transform(X, **kwargs)

    def transform(self, X: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement transform()."
        )

    def predict(self, X: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict()."
        )
