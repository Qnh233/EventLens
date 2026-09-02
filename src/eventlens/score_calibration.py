from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ClasswiseScoreCalibrator:
    """校准 OVR 类别分数的跨类尺度；不使用事件名或主体真值。"""

    def __init__(self, method: str):
        if method not in {"identity", "zscore", "platt"}:
            raise ValueError(f"unsupported calibration method: {method}")
        self.method = method
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.models_: list[LogisticRegression | None] | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray | None = None) -> "ClasswiseScoreCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D matrix")

        if self.method == "identity":
            return self
        if self.method == "zscore":
            self.center_ = scores.mean(axis=0)
            scale = scores.std(axis=0)
            self.scale_ = np.where(scale > 1e-8, scale, 1.0)
            return self

        if labels is None:
            raise ValueError("platt calibration requires class-index labels")
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (scores.shape[0],):
            raise ValueError("labels length mismatch")
        self.models_ = []
        for class_index in range(scores.shape[1]):
            target = (labels == class_index).astype(np.int64)
            if target.min() == target.max():
                self.models_.append(None)
                continue
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            model.fit(scores[:, [class_index]], target)
            self.models_.append(model)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self.method == "identity":
            return scores.copy()
        if self.method == "zscore":
            if self.center_ is None or self.scale_ is None:
                raise RuntimeError("zscore calibrator is not fitted")
            return (scores - self.center_) / self.scale_
        if self.models_ is None:
            raise RuntimeError("platt calibrator is not fitted")

        calibrated = np.zeros_like(scores, dtype=np.float64)
        for class_index, model in enumerate(self.models_):
            if model is None:
                calibrated[:, class_index] = 0.0
            else:
                calibrated[:, class_index] = model.predict_proba(scores[:, [class_index]])[:, 1]
        return calibrated

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)
