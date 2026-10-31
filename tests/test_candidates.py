from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from credit_risk.data.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from credit_risk.features.build import feature_frame, split_frames
from credit_risk.models import candidates


def _skip_if_missing(name: str) -> None:
    if not candidates.is_available(name):
        pytest.skip(f"{name} is not installed")


@pytest.mark.parametrize("name", candidates.CANDIDATES)
def test_every_candidate_fits_and_scores(name, processed_frame: pd.DataFrame, config) -> None:
    _skip_if_missing(name)
    parts = split_frames(processed_frame)
    spec = candidates.spec(name, config)
    spec.estimator.fit(
        feature_frame(parts["development"]), parts["development"]["default"].to_numpy()
    )
    probability = spec.estimator.predict_proba(feature_frame(parts["holdout"]))[:, 1]
    assert probability.shape == (len(parts["holdout"]),)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert len(set(probability.round(6))) > 1


def test_encoding_flags_match_the_family(config) -> None:
    assert candidates.spec("logistic", config).encoding == "one_hot"
    assert candidates.spec("lightgbm", config).encoding == "one_hot"
    if candidates.is_available("xgboost"):
        assert candidates.spec("xgboost", config).encoding == "native"
    if candidates.is_available("catboost"):
        assert candidates.spec("catboost", config).encoding == "native"


@pytest.mark.parametrize("name", ["xgboost", "catboost"])
def test_native_models_receive_category_dtype(name, processed_frame, config) -> None:
    _skip_if_missing(name)
    parts = split_frames(processed_frame)
    spec = candidates.spec(name, config)
    spec.estimator.fit(
        feature_frame(parts["development"]), parts["development"]["default"].to_numpy()
    )
    matrix = spec.estimator.named_steps["preprocess"].transform(feature_frame(parts["holdout"]))
    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.columns) == list(FEATURE_COLUMNS)
    for column in CATEGORICAL_COLUMNS:
        assert str(matrix[column].dtype) == "category"


def test_missing_library_raises_a_clear_error(config, monkeypatch) -> None:
    original = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "xgboost":
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(candidates.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(candidates.CandidateUnavailable, match="xgboost"):
        candidates.spec("xgboost", config)


def test_available_candidates_reports_skips(config, monkeypatch) -> None:
    monkeypatch.setattr(candidates, "is_available", lambda name: name != "catboost")
    usable, skipped = candidates.available_candidates(["logistic", "catboost"])
    assert usable == ["logistic"]
    assert "catboost" in skipped


def test_unknown_candidate_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="unknown model type"):
        candidates.spec("tabpfn", config)
