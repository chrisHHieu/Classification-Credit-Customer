from __future__ import annotations

import pandas as pd

from credit_risk.data.schema import FEATURE_COLUMNS, LABEL_COLUMN, TARGET
from credit_risk.features.build import (
    add_derived_features,
    build_dataset,
    feature_frame,
    load_processed,
    split_frames,
    write_processed,
)


def test_processed_columns_and_order(
    raw_frame: pd.DataFrame, processed_frame: pd.DataFrame
) -> None:
    for column in FEATURE_COLUMNS:
        assert column in processed_frame.columns
    assert processed_frame.columns[:3].tolist() == ["row_id", "duration_months", "credit_amount"]
    assert processed_frame["row_id"].tolist() == list(range(len(raw_frame)))


def test_target_is_encoded_without_leaking_into_features(raw_frame: pd.DataFrame) -> None:
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert TARGET not in FEATURE_COLUMNS
    assert "row_id" not in FEATURE_COLUMNS
    assert "fold" not in FEATURE_COLUMNS


def test_build_is_deterministic(raw_frame: pd.DataFrame, config) -> None:
    first = build_dataset(raw_frame, config)
    second = build_dataset(raw_frame, config)
    pd.testing.assert_frame_equal(first, second)


def test_derived_feature_is_a_plain_ratio(raw_frame: pd.DataFrame, config) -> None:
    frame = build_dataset(raw_frame, config)
    row = frame.iloc[0]
    expected = round(row["credit_amount"] / row["duration_months"], 4)
    assert abs(row["credit_amount_per_month"] - expected) < 1e-9


def test_split_is_stratified_and_frozen(raw_frame: pd.DataFrame, config) -> None:
    frame = build_dataset(raw_frame, config)
    parts = split_frames(frame)
    expected_holdout = round(len(frame) * config.split.test_size)
    assert len(parts["holdout"]) == expected_holdout
    assert len(parts["development"]) == len(frame) - expected_holdout

    development = parts["development"]
    assert sorted(development["fold"].unique().tolist()) == list(range(config.split.n_splits))
    assert (parts["holdout"]["fold"] == -1).all()
    assert set(parts["holdout"]["row_id"]).isdisjoint(set(development["row_id"]))

    for _, fold in development.groupby("fold"):
        assert abs(fold[LABEL_COLUMN].mean() - frame[LABEL_COLUMN].mean()) < 0.2


def test_write_and_load_roundtrip(raw_frame: pd.DataFrame, config, tmp_path) -> None:
    frame = build_dataset(raw_frame, config)
    path = write_processed(frame, config)
    assert path.exists()
    loaded = load_processed(config)
    assert len(loaded) == len(frame)
    for column in FEATURE_COLUMNS:
        assert loaded[column].tolist() == frame[column].tolist()


def test_feature_frame_is_canonical(raw_frame: pd.DataFrame, config) -> None:
    frame = build_dataset(raw_frame, config)
    features = feature_frame(frame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.isna().sum().sum() == 0


def test_derived_features_does_not_mutate_input(raw_frame: pd.DataFrame) -> None:
    before = raw_frame.copy()
    add_derived_features(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)
