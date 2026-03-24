from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from modules.base import BaseModule


def _as_numpy(X: Any) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)


@dataclass
class SVMClassifier(BaseModule):
    """SVM 分类器（sklearn 封装）。"""

    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    random_state: Optional[int] = 0

    name: str = "SVMClassifier"

    def __post_init__(self) -> None:
        from sklearn.svm import SVC

        self._model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, random_state=self.random_state)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "SVMClassifier":
        if y is None:
            raise ValueError("SVMClassifier.fit requires y")
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        self._model.fit(Xn, np.asarray(y))
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        return self._model.predict(Xn)


@dataclass
class RandomForestClassifier(BaseModule):
    """随机森林分类器（sklearn 封装）。"""

    n_estimators: int = 200
    max_depth: Optional[int] = None
    random_state: Optional[int] = 0
    n_jobs: Optional[int] = -1

    name: str = "RandomForestClassifier"

    def __post_init__(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as _RF

        self._model = _RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "RandomForestClassifier":
        if y is None:
            raise ValueError("RandomForestClassifier.fit requires y")
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        self._model.fit(Xn, np.asarray(y))
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        return self._model.predict(Xn)


@dataclass
class LogisticRegressionClassifier(BaseModule):
    """逻辑回归分类器（sklearn 封装）。"""

    max_iter: int = 2000
    solver: str = "lbfgs"
    random_state: Optional[int] = 0

    name: str = "LogisticRegressionClassifier"

    def __post_init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "LogisticRegressionClassifier":
        if y is None:
            raise ValueError("LogisticRegressionClassifier.fit requires y")
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        self._model.fit(Xn, np.asarray(y))
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        Xn = _as_numpy(X).astype(np.float32, copy=False)
        return self._model.predict(Xn)
