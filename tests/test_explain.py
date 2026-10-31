from __future__ import annotations

import pytest

from credit_risk.data.schema import FEATURE_COLUMNS, feature_name_parts
from credit_risk.features.build import feature_frame
from credit_risk.models import explain, registry


@pytest.fixture
def holdout_rows(project):
    frame = project.processed
    return feature_frame(frame[frame["is_heldout"] == 1].head(5))


def test_explain_row_returns_ranked_attribute_reasons(project, holdout_rows) -> None:
    bundle, _ = registry.load_bundle(project.cfg, "logistic")
    reasons = explain.explain_row(bundle, holdout_rows.head(1), top_k=5)
    assert len(reasons) == 5
    for row in reasons:
        assert row["feature"] in FEATURE_COLUMNS
        assert row["direction"] in {"raises_risk", "lowers_risk"}
        assert isinstance(row["contribution_log_odds"], float)
    magnitudes = [abs(row["contribution_log_odds"]) for row in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)


@pytest.mark.parametrize("model_type", ["logistic", "lightgbm", "xgboost", "catboost"])
def test_shap_values_align_with_encoded_features(model_type, project, holdout_rows) -> None:
    if not registry.list_versions(project.cfg, model_type):
        pytest.skip(f"{model_type} was not trained in this fixture")
    bundle, _ = registry.load_bundle(project.cfg, model_type)
    values = explain.contributions(bundle, holdout_rows)
    assert values.shape == (len(holdout_rows), len(bundle.feature_names))


def test_global_importance_groups_one_hot_columns(project, holdout_rows) -> None:
    bundle, _ = registry.load_bundle(project.cfg, "lightgbm")
    table = explain.global_importance(bundle, holdout_rows, top_k=5)
    assert len(table) == 5
    assert [row["importance"] for row in table] == sorted(
        [row["importance"] for row in table], reverse=True
    )
    assert all(row["feature"] in FEATURE_COLUMNS for row in table)


def test_feature_names_for_native_model_have_no_one_hot_codes(project, holdout_rows) -> None:
    if not registry.list_versions(project.cfg, "catboost"):
        pytest.skip("catboost was not trained in this fixture")
    bundle, _ = registry.load_bundle(project.cfg, "catboost")
    for name in bundle.feature_names:
        source, code = feature_name_parts(name)
        assert source in FEATURE_COLUMNS
        assert code is None
