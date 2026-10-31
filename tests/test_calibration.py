from __future__ import annotations

import numpy as np

from credit_risk.models import calibration


def _synthetic(seed: int = 3, rows: int = 400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, rows)
    raw = np.clip(rng.beta(2 + 3 * y, 5 - 2 * y), 1e-4, 1 - 1e-4)
    return y, raw


def test_platt_is_monotone() -> None:
    y, raw = _synthetic()
    calibrator = calibration.PlattCalibrator().fit(raw, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = calibrator.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_choose_calibrator_returns_lower_brier_method() -> None:
    y, raw = _synthetic()
    calibrator, method, table = calibration.choose_calibrator(raw, y)
    assert method in {"isotonic", "platt"}
    assert {row["method"] for row in table} == {"isotonic", "platt"}
    scores = {row["method"]: row["brier_inner"] for row in table}
    assert scores[method] == min(scores.values())
    calibrated = calibrator.predict(raw)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_auto_resolves_to_a_single_method() -> None:
    y, raw = _synthetic()
    _, method, table = calibration.resolve("auto", raw, y)
    assert method in {"isotonic", "platt"}
    assert len(table) == 2


def test_explicit_method_skips_comparison() -> None:
    y, raw = _synthetic()
    calibrator, method, table = calibration.resolve("platt", raw, y)
    assert method == "platt"
    assert len(table) == 1
    assert table[0]["method"] == "platt"
    assert calibrator.predict(raw).shape == raw.shape
