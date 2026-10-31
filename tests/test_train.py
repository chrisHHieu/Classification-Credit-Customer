from __future__ import annotations

import json

import numpy as np
import pytest

from credit_risk.data.schema import FEATURE_COLUMNS, feature_name_parts
from credit_risk.features.build import feature_frame
from credit_risk.models import registry
from credit_risk.models.bundle import CalibratedModel
from credit_risk.models.evaluate import confusion_matrix_array, run_evaluation
from credit_risk.models.metrics import expected_cost_per_1000, select_threshold
from credit_risk.models.train import fit_calibrated, make_pipeline
from credit_risk.monitor.drift import run_drift


def test_expected_cost_matches_hand_calculation(config) -> None:
    target = np.array([1, 0])
    probability = np.array([0.9, 0.1])
    assert expected_cost_per_1000(target, probability, 0.5, config.cost) == 0.0
    assert expected_cost_per_1000(target, probability, 0.95, config.cost) == 2500.0
    assert expected_cost_per_1000(target, probability, 0.05, config.cost) == 500.0


def test_select_threshold_prefers_low_cost(config) -> None:
    target = np.array([1] * 30 + [0] * 70)
    probability = np.where(target == 1, 0.8, 0.05)
    selection = select_threshold(target, probability, config.cost)
    assert selection["cost_per_1000"] == 0.0
    assert 0.05 < selection["threshold"] <= 0.8
    assert selection["plateau_low"] <= selection["threshold"] <= selection["plateau_high"]


def test_select_threshold_is_bounded(config) -> None:
    target = np.array([1, 0, 1, 0] * 10)
    probability = np.full(40, 0.5)
    selection = select_threshold(target, probability, config.cost)
    assert config.cost.threshold_min <= selection["threshold"] <= config.cost.threshold_max


def test_fit_calibrated_is_reproducible(processed_frame, config) -> None:
    development = processed_frame[processed_frame["is_heldout"] == 0]
    holdout = processed_frame[processed_frame["is_heldout"] == 1]
    first = fit_calibrated(make_pipeline("lightgbm", config), development, config, "lightgbm")
    second = fit_calibrated(make_pipeline("lightgbm", config), development, config, "lightgbm")
    assert isinstance(first, CalibratedModel)
    assert first.threshold == second.threshold
    np.testing.assert_allclose(
        first.predict_proba_bad(holdout), second.predict_proba_bad(holdout), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("model_type", ["logistic", "lightgbm"])
def test_reason_codes_are_readable(processed_frame, config, model_type: str) -> None:
    development = processed_frame[processed_frame["is_heldout"] == 0]
    model = fit_calibrated(make_pipeline(model_type, config), development, config, model_type)
    row = feature_frame(processed_frame.iloc[[0]])
    codes = model.reason_codes(row, top_k=3)
    assert len(codes) == 3
    for code in codes:
        assert code["label"]
        assert feature_name_parts(code["feature"])[0] in FEATURE_COLUMNS
        assert code["direction"] in {"raises_risk", "lowers_risk"}
        assert abs(code["contribution_log_odds"]) >= 0


def test_training_registers_both_models(project) -> None:
    model_type, version = registry.read_latest(project.cfg)
    assert model_type == "logistic"
    for candidate in ("logistic", "lightgbm"):
        metadata = registry.load_metadata(project.cfg, candidate, version)
        for key in (
            "version",
            "model_type",
            "seed",
            "config_hash",
            "data_sha256",
            "git_commit",
            "trained_at",
            "metrics",
            "threshold",
        ):
            assert key in metadata
        assert metadata["seed"] == 42
        assert 0.0 < metadata["threshold"] < 1.0
        assert (registry.artifact_dir(project.cfg, candidate, version) / "model.joblib").exists()


def test_artifact_reloads_and_scores(project) -> None:
    bundle, metadata = registry.load_bundle(project.cfg)
    holdout = project.processed[project.processed["is_heldout"] == 1]
    probability = bundle.predict_proba_bad(holdout)
    assert len(probability) == len(holdout)
    assert np.all((probability >= 0) & (probability <= 1))
    assert bundle.threshold == metadata["threshold"]
    decisions = bundle.decide(holdout)
    assert set(decisions) <= {"approve", "reject"}


def test_evaluate_writes_metrics_and_figures(project) -> None:
    payload = run_evaluation(project.cfg)
    on_disk = json.loads(project.cfg.metrics_file.read_text(encoding="utf-8"))
    assert on_disk["holdout"]["logistic"]["roc_auc"] == payload["holdout"]["logistic"]["roc_auc"]
    assert on_disk["consistency"]["roc_auc_matches_metadata"] is True
    assert len(on_disk["figures"]) == 5
    for relative in on_disk["figures"]:
        path = project.cfg.root / relative
        assert path.exists() and path.stat().st_size > 1000
    for key in ("roc_auc", "pr_auc", "ks", "brier", "accuracy_at_threshold"):
        assert key in on_disk["holdout"]["logistic"]
    assert len(on_disk["cost_table"]) >= 6


def test_drift_writes_report(project) -> None:
    payload = run_drift(project.cfg)
    assert project.cfg.drift_file.exists()
    assert set(payload["slices"]) == {"all", "holdout"}
    for block in payload["slices"].values():
        assert block["overall"] in {"ok", "warn", "alert"}
        assert len(block["features"]) == 21


@pytest.mark.parametrize("model_type", ["logistic", "lightgbm"])
def test_holdout_cost_beats_trivial_policies(project, model_type: str) -> None:
    summary = project.summary["models"][model_type]
    holdout = summary["holdout"]
    always_approve = project.summary["policies"]["always_approve_per_1000"]
    assert holdout["expected_cost_per_1000"] < always_approve


def test_confusion_matrix_array_orientation() -> None:
    metrics = {
        "confusion": {
            "true_good_approved": 77,
            "true_good_rejected": 63,
            "bad_approved": 8,
            "bad_rejected": 52,
        }
    }
    matrix = confusion_matrix_array(metrics)
    assert matrix.tolist() == [[77, 63], [8, 52]]
    assert matrix.sum() == 200
