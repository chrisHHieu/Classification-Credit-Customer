"""Deterministic feature build and split assignment.

Two rules drive this module:

* nothing is computed from the target, so there is no target leakage;
* the split is materialised once, in the processed file, so training, evaluation
  and drift all read the same rows. The held-out 20 percent is fixed before any
  model is fitted and is never touched by cross-validation folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_risk.config import Config
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
    TARGET,
)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the one ratio feature used by the models.

    The ratio is a plain transformation of two raw columns, which keeps the reason
    codes readable: a contribution attached to it still maps to a real applicant
    attribute.
    """
    out = frame.copy()
    duration = out["duration_months"].astype(float).replace(0.0, np.nan)
    out["credit_amount_per_month"] = (out["credit_amount"].astype(float) / duration).round(4)
    return out


def _cast_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in NUMERIC_COLUMNS + DERIVED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].astype(str)
    out[LABEL_COLUMN] = out[LABEL_COLUMN].astype("int64")
    return out


def assign_splits(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach ``fold`` (-1 for holdout) and ``is_heldout`` using the config seed."""
    out = frame.reset_index(drop=True).copy()
    out["row_id"] = np.arange(len(out), dtype="int64")
    label = out[LABEL_COLUMN].to_numpy()

    train_idx, holdout_idx = train_test_split(
        np.arange(len(out)),
        test_size=cfg.split.test_size,
        stratify=label,
        random_state=cfg.seed,
        shuffle=True,
    )
    folds = np.full(len(out), -1, dtype="int64")
    skf = StratifiedKFold(n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.seed)
    for fold, (_, valid_idx) in enumerate(skf.split(train_idx, label[train_idx])):
        folds[train_idx[valid_idx]] = fold
    out["fold"] = folds
    out["is_heldout"] = (out["fold"] < 0).astype("int64")
    return out


def build_dataset(raw_frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Raw Statlog frame to processed modelling frame."""
    frame = raw_frame.reset_index(drop=True).copy()
    frame[LABEL_COLUMN] = (frame[TARGET].astype(int) == 2).astype("int64")
    frame = add_derived_features(frame)
    frame = _cast_columns(frame)
    frame = assign_splits(frame, cfg)

    ordered = (
        ["row_id", *FEATURE_COLUMNS, LABEL_COLUMN, TARGET, "fold", "is_heldout"]
    )
    extra = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extra]


def write_processed(frame: pd.DataFrame, cfg: Config) -> Path:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.processed_file, index=False)
    return cfg.processed_file


def load_processed(cfg: Config) -> pd.DataFrame:
    if not cfg.processed_file.exists():
        raise FileNotFoundError(
            f"{cfg.processed_file} not found, run 'credit_risk features' first"
        )
    frame = pd.read_csv(cfg.processed_file)
    missing = [
        column
        for column in [*FEATURE_COLUMNS, LABEL_COLUMN, "fold"]
        if column not in frame
    ]
    if missing:
        raise ValueError(f"processed file is missing columns {missing}")
    return frame


def build_features(raw_frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Alias kept for the CLI vocabulary: features implies the frozen split too."""
    return build_dataset(raw_frame, cfg)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The model input matrix in canonical column order."""
    return frame[list(FEATURE_COLUMNS)].copy()


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the development and held-out partitions."""
    return {
        "development": frame[frame["is_heldout"] == 0].copy(),
        "holdout": frame[frame["is_heldout"] == 1].copy(),
    }
