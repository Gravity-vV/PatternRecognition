from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return ClassificationMetrics(
        accuracy=acc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=cm,
    )
