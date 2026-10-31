from __future__ import annotations

import numpy as np

from credit_risk.models import conformal


def test_quantile_uses_the_finite_sample_rank() -> None:
    scores = np.arange(10, dtype=float)
    # ceil((10 + 1) * 0.9) = 10, so the 10th smallest score, not the 9th.
    assert conformal.conformal_quantile(scores, alpha=0.1) == 9.0
    # ceil((10 + 1) * 0.5) = 6, so the 6th smallest score.
    assert conformal.conformal_quantile(scores, alpha=0.5) == 5.0


def test_decisions_cover_three_outcomes() -> None:
    calibrator = conformal.ConformalCalibrator(alpha=0.1, quantile=0.15, n_calibration=100)
    assert calibrator.decision(0.05) == "approve"
    assert calibrator.decision(0.95) == "reject"
    assert calibrator.decision(0.5) == "review"
    assert set(calibrator.decisions(np.array([0.01, 0.5, 0.99]))) == {
        "approve",
        "review",
        "reject",
    }


def test_empty_set_is_reported_as_review() -> None:
    # q below 0.5 cannot admit either label for p in (q, 1-q); the code abstains.
    calibrator = conformal.ConformalCalibrator(alpha=0.4, quantile=0.3, n_calibration=100)
    assert calibrator.labels(0.5) == []
    assert calibrator.decision(0.5) == "review"
    block = conformal.coverage(np.array([0.5]), np.array([1]), calibrator)
    assert block["empty_set_rate"] == 1.0
    assert block["empirical_coverage"] == 0.0


def test_coverage_on_the_calibration_scores_reaches_the_target() -> None:
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 2000)
    p = np.clip(rng.beta(2 + 2 * y, 6 - 2 * y), 1e-3, 1 - 1e-3)
    calibrator = conformal.fit_conformal(p, y, alpha=0.2)
    block = conformal.coverage(p, y, calibrator)
    assert block["target_coverage"] == 0.8
    assert block["empirical_coverage"] >= 0.75
    assert block["rows"] == 2000
    assert block["quantile"] > 0.0
