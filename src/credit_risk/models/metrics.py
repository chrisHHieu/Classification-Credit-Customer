"""Metric definitions and the expected-cost threshold search.

Positive class is ``default`` (the raw target value 2, a bad credit risk). The
decision rule is "approve when the calibrated default probability is below the
threshold", because a low default probability is a good applicant.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from credit_risk.config import CostConfig


def _arrays(y_true, p_bad) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(p_bad, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape}, p {p.shape}")
    return y, np.clip(p, 0.0, 1.0)


def decision_mask(y_true, p_bad, threshold: float) -> np.ndarray:
    """True where the applicant is rejected (predicted bad)."""
    _, p = _arrays(y_true, p_bad)
    return p >= threshold


def expected_cost_per_1000(y_true, p_bad, threshold: float, cost: CostConfig) -> float:
    """Realised cost of the policy on 1000 applications.

    Approving a bad applicant costs ``cost_bad_approved``; rejecting a good one
    costs ``cost_good_rejected``.
    """
    y, p = _arrays(y_true, p_bad)
    if len(y) == 0:
        return float("nan")
    approve = p < threshold
    unit = np.where(approve, cost.cost_bad_approved * y, cost.cost_good_rejected * (1 - y))
    return float(unit.mean() * 1000.0)


def confusion_at(y_true, p_bad, threshold: float) -> dict[str, int]:
    y, _ = _arrays(y_true, p_bad)
    rejected = decision_mask(y_true, p_bad, threshold)
    tn, fp, fn, tp = confusion_matrix(y, rejected.astype(int), labels=[0, 1]).ravel()
    return {
        "true_good_approved": int(tn),
        "true_good_rejected": int(fp),
        "bad_approved": int(fn),
        "bad_rejected": int(tp),
    }


def ks_statistic(y_true, p_bad) -> float:
    y, p = _arrays(y_true, p_bad)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true, p_bad, bins: int = 10) -> float:
    """Weighted gap between predicted probability and observed default rate.

    Quantile bins keep each bucket populated, which matters with 200 held-out
    rows. Rows are ordered by predicted probability, so a model that is
    confidently wrong is penalised more than one that is unsure.
    """
    y, p = _arrays(y_true, p_bad)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    chunks = np.array_split(order, min(bins, len(order)))
    total = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        confidence = float(p[chunk].mean())
        frequency = float(y[chunk].mean())
        total += (len(chunk) / len(y)) * abs(frequency - confidence)
    return float(total)


def classification_metrics(y_true, p_bad, threshold: float, cost: CostConfig) -> dict:
    """All reported metrics for one set of calibrated probabilities."""
    y, p = _arrays(y_true, p_bad)
    rejected = p >= threshold
    confusion = confusion_at(y, p, threshold)
    return {
        "n": int(len(y)),
        "bad_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "accuracy_at_threshold": float(accuracy_score(y, rejected.astype(int))),
        "approval_rate": float((~rejected).mean()),
        "expected_cost_per_1000": expected_cost_per_1000(y, p, threshold, cost),
        "cost_0_5_per_1000": expected_cost_per_1000(y, p, 0.5, cost),
        "confusion": confusion,
    }


def metrics_with_row_thresholds(y_true, p_bad, thresholds, cost: CostConfig) -> dict:
    """Metrics when every row was scored under its own fold threshold.

    Used for the pooled cross-validation view: each row is decided by the
    threshold that was chosen on its training folds, so the pooled cost is not
    an artefact of one arbitrary shared threshold.
    """
    y, p = _arrays(y_true, p_bad)
    thr = np.asarray(thresholds, dtype=float).ravel()
    if thr.shape != p.shape:
        raise ValueError(f"shape mismatch: thresholds {thr.shape}, p {p.shape}")
    rejected = p >= thr
    unit = np.where(~rejected, cost.cost_bad_approved * y, cost.cost_good_rejected * (1 - y))
    tn, fp, fn, tp = confusion_matrix(y, rejected.astype(int), labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "bad_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(np.mean(thr)),
        "accuracy_at_threshold": float(accuracy_score(y, rejected.astype(int))),
        "approval_rate": float((~rejected).mean()),
        "expected_cost_per_1000": float(unit.mean() * 1000.0),
        "cost_0_5_per_1000": expected_cost_per_1000(y, p, 0.5, cost),
        "confusion": {
            "true_good_approved": int(tn),
            "true_good_rejected": int(fp),
            "bad_approved": int(fn),
            "bad_rejected": int(tp),
        },
    }


def always_approve_cost(y_true, cost: CostConfig) -> float:
    y = np.asarray(y_true, dtype=int).ravel()
    return float(y.mean() * cost.cost_bad_approved * 1000.0)


def always_reject_cost(y_true, cost: CostConfig) -> float:
    y = np.asarray(y_true, dtype=int).ravel()
    return float((1 - y).mean() * cost.cost_good_rejected * 1000.0)


def select_threshold(y_true, p_bad, cost: CostConfig) -> dict:
    """Pick the threshold with the lowest expected cost on the supplied rows.

    Ties inside ``cost.tolerance`` of the minimum are resolved toward the highest
    threshold, that is the policy that rejects the fewest applicants. The plateau
    is the contiguous run of thresholds around the choice that stays within
    tolerance, which is what a reviewer needs to see to know the choice is stable
    rather than a knife edge.
    """
    y, p = _arrays(y_true, p_bad)
    grid = np.unique(np.clip(p, cost.threshold_min, cost.threshold_max))
    if len(grid) == 0:
        grid = np.array([0.5])
    costs = np.array([expected_cost_per_1000(y, p, float(t), cost) for t in grid])
    best = float(costs.min())
    allowed = costs <= best * (1.0 + cost.tolerance)
    chosen_index = int(np.max(np.flatnonzero(allowed)))
    low_index = chosen_index
    while low_index > 0 and allowed[low_index - 1]:
        low_index -= 1
    high_index = chosen_index
    while high_index + 1 < len(grid) and allowed[high_index + 1]:
        high_index += 1
    return {
        "threshold": float(grid[chosen_index]),
        "cost_per_1000": float(costs[chosen_index]),
        "best_cost_per_1000": best,
        "plateau_low": float(grid[low_index]),
        "plateau_high": float(grid[high_index]),
        "grid_size": int(len(grid)),
    }


def cost_curve(y_true, p_bad, cost: CostConfig, points: int = 100) -> list[dict]:
    """Expected cost across a fixed threshold grid, for the report figure."""
    thresholds = np.linspace(cost.threshold_min, cost.threshold_max, points)
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": float(threshold),
                "expected_cost_per_1000": expected_cost_per_1000(y_true, p_bad, threshold, cost),
                "approval_rate": float((np.asarray(p_bad) < threshold).mean()),
            }
        )
    return rows
