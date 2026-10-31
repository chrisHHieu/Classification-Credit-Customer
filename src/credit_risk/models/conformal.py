"""Split-conformal label sets for the approve/reject decision.

The conformal score of a row is ``1 - p(true label)`` with the calibrated
probability, so the prediction set is ``{good, bad}`` minus the labels whose score
exceeds the conformal quantile. Mapping the set onto the decision vocabulary gives
three outcomes:

* ``approve``: only "good" survives, the applicant is confidently acceptable;
* ``reject``: only "bad" survives, the applicant is confidently a default;
* ``review``: both labels survive (or neither, which is counted separately),
  so the model abstains and a human decides.

Scores come from inner-fold out-of-fold predictions of the development partition;
the held-out 20 percent is only used to measure empirical coverage, never to fit
the quantile. ``alpha`` is the allowed miscoverage, so the target coverage is
``1 - alpha``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

APPROVE = "approve"
REVIEW = "review"
REJECT = "reject"


@dataclass(frozen=True)
class ConformalCalibrator:
    alpha: float
    quantile: float
    n_calibration: int
    method: str = "split_conformal_label_sets"

    def labels(self, p_bad: float) -> list[str]:
        """Labels whose conformal score is inside the quantile."""
        p = float(np.clip(p_bad, 0.0, 1.0))
        allowed: list[str] = []
        if p <= self.quantile:
            allowed.append("good")
        if (1.0 - p) <= self.quantile:
            allowed.append("bad")
        return allowed

    def decision(self, p_bad: float) -> str:
        allowed = self.labels(p_bad)
        if allowed == ["good"]:
            return APPROVE
        if allowed == ["bad"]:
            return REJECT
        return REVIEW

    def decisions(self, probability) -> np.ndarray:
        return np.array([self.decision(value) for value in np.asarray(probability, float)])

    def covers(self, p_bad: float, target: int) -> bool:
        label = "bad" if int(target) == 1 else "good"
        return label in self.labels(p_bad)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "quantile": self.quantile,
            "n_calibration": self.n_calibration,
            "target_coverage": 1.0 - self.alpha,
        }


def conformal_quantile(scores, alpha: float) -> float:
    """The finite-sample split-conformal quantile ``ceil((n+1)(1-alpha))/n``."""
    values = np.sort(np.asarray(scores, dtype=float))
    n = len(values)
    if n == 0:
        raise ValueError("conformal calibration needs at least one score")
    rank = min(int(math.ceil((n + 1) * (1.0 - alpha))), n)
    return float(values[rank - 1])


def fit_conformal(probability, target, alpha: float) -> ConformalCalibrator:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: p {p.shape}, y {y.shape}")
    scores = np.where(y == 1, 1.0 - p, p)
    return ConformalCalibrator(
        alpha=float(alpha),
        quantile=conformal_quantile(scores, alpha),
        n_calibration=int(len(scores)),
    )


def coverage(probability, target, calibrator: ConformalCalibrator) -> dict:
    """Empirical coverage and the composition of prediction sets."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    decisions = calibrator.decisions(p)
    covered = np.array([calibrator.covers(value, label) for value, label in zip(p, y, strict=True)])
    empty = np.array([len(calibrator.labels(value)) == 0 for value in p])
    return {
        "alpha": calibrator.alpha,
        "target_coverage": 1.0 - calibrator.alpha,
        "empirical_coverage": float(covered.mean()) if len(covered) else float("nan"),
        "rows": int(len(p)),
        "approve_rate": float((decisions == APPROVE).mean()) if len(p) else float("nan"),
        "review_rate": float((decisions == REVIEW).mean()) if len(p) else float("nan"),
        "reject_rate": float((decisions == REJECT).mean()) if len(p) else float("nan"),
        "empty_set_rate": float(empty.mean()) if len(p) else float("nan"),
        "quantile": calibrator.quantile,
    }
