"""Shared fixtures. Every fixture is built in code, so the suite needs no network."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Config, load_config
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
    TARGET,
    VALID_CODES,
)
from credit_risk.features.build import build_dataset, write_processed
from credit_risk.models.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROWS = 120


def make_raw_frame(rows: int = FIXTURE_ROWS, seed: int = 7) -> pd.DataFrame:
    """Synthetic applicants with the real schema and a learnable signal."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for column in NUMERIC_COLUMNS:
        low, high = NUMERIC_RANGES[column]
        if column == "credit_amount":
            high = 15000
        data[column] = rng.integers(int(low), int(high) + 1, rows)
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.choice(list(VALID_CODES[column]), rows)

    frame = pd.DataFrame(data)[list(COLUMNS[:-1])]
    signal = (
        (frame["installment_rate"] >= 3).astype(int)
        + frame["checking_status"].isin(["A11", "A12"]).astype(int)
        + (frame["duration_months"] >= 30).astype(int)
        + (frame["savings"].isin(["A61", "A62"])).astype(int)
        + rng.integers(0, 2, rows)
    )
    frame[TARGET] = np.where(signal >= 3, 2, 1).astype(int)
    return frame[list(COLUMNS)]


def write_raw_like(frame: pd.DataFrame, path: Path) -> Path:
    """Write a frame in the space separated Statlog layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, row in frame.iterrows():
        values = []
        for column in COLUMNS:
            value = row[column]
            values.append(str(int(value)) if column in NUMERIC_COLUMNS else str(value))
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_test_config(root: Path, rows: int = FIXTURE_ROWS) -> Config:
    """Project config pointed at a temp root, with small models for speed."""
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", root=root)
    lightgbm = {
        **cfg.train.lightgbm,
        "n_estimators": 60,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 5,
    }
    data = replace(cfg.data, expected_rows=rows, expected_bad_count=rows // 2)
    return replace(
        cfg,
        data=data,
        split=replace(cfg.split, n_splits=3),
        calibration=replace(cfg.calibration, inner_splits=3),
        train=replace(cfg.train, lightgbm=lightgbm, permutation_repeats=3),
    )


def config_matching(frame: pd.DataFrame, cfg: Config) -> Config:
    """Config whose expected row count and bad count match the supplied frame."""
    bad = int((frame[TARGET].astype(int) == 2).sum())
    return replace(
        cfg,
        data=replace(cfg.data, expected_rows=len(frame), expected_bad_count=bad),
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def config(tmp_path: Path, raw_frame: pd.DataFrame) -> Config:
    return build_test_config(tmp_path, rows=len(raw_frame))


@pytest.fixture
def processed_frame(raw_frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    return build_dataset(raw_frame, config)


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One trained project shared by the end-to-end and API tests."""
    root = tmp_path_factory.mktemp("credit_project")
    cfg = build_test_config(root)
    raw = make_raw_frame()
    processed = build_dataset(raw, cfg)
    write_processed(processed, cfg)
    write_raw_like(raw, cfg.raw_file)
    summary = run_training(cfg)
    return SimpleNamespace(cfg=cfg, root=root, raw=raw, processed=processed, summary=summary)
