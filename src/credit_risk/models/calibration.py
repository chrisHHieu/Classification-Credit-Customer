"""Probability calibration.

Two calibrators are supported and both are fitted on inner-fold out-of-fold
predictions of the development partition only:

* ``isotonic``: monotone step function, flexible but hungry for data;
* ``platt``: a two parameter sigmoid fitted on the logit of the base score, the
  modern form of Platt scaling (Lin, Lin and Weng, 2007).

``auto`` fits both and keeps the one with the lower Brier score on those same
inner predictions. The comparison table is returned so the choice is auditable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPSILON = 1e-6


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Logistic map from the base model logit to a calibrated probability."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    def fit(self, probability, target) -> PlattCalibrator:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model.fit(_logit(probability), np.asarray(target, dtype=int))
        return self

    def predict(self, probability) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        out = self.model.predict_proba(_logit(probability))[:, 1]
        return np.clip(out, 0.0, 1.0)


def fit_calibrator(method: str, probability, target):
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(np.asarray(probability, dtype=float), np.asarray(target, dtype=int))
        return calibrator
    if method == "platt":
        return PlattCalibrator().fit(probability, target)
    raise ValueError(f"unknown calibration method: {method}")


def brier(target, probability) -> float:
    return float(brier_score_loss(np.asarray(target, dtype=int), np.asarray(probability, float)))


def choose_calibrator(
    probability,
    target,
    methods: tuple[str, ...] = ("isotonic", "platt"),
) -> tuple[Any, str, list[dict]]:
    """Fit every method, keep the lowest inner-fold Brier, return the audit table."""
    table: list[dict] = []
    best_method, best_calibrator, best_brier = "", None, float("inf")
    for method in methods:
        calibrator = fit_calibrator(method, probability, target)
        calibrated = np.clip(calibrator.predict(probability), 0.0, 1.0)
        score = brier(target, calibrated)
        table.append({"method": method, "brier_inner": score})
        if score < best_brier:
            best_method, best_calibrator, best_brier = method, calibrator, score
    if best_calibrator is None:
        raise RuntimeError("no calibration method could be fitted")
    return best_calibrator, best_method, table


def resolve(method: str, probability, target) -> tuple[Any, str, list[dict]]:
    """Dispatch on the configured method, with ``auto`` comparing both."""
    if method == "auto":
        return choose_calibrator(probability, target)
    calibrator = fit_calibrator(method, probability, target)
    calibrated = np.clip(calibrator.predict(probability), 0.0, 1.0)
    return calibrator, method, [{"method": method, "brier_inner": brier(target, calibrated)}]
