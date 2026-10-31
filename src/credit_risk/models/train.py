"""Training entry point.

Protocol, stated once here and repeated in the README:

1. The processed file already carries a frozen stratified 80/20 split.
2. Headline numbers come from stratified 5-fold cross-validation on the 80 percent
   development partition. Inside every outer fold, the calibrator is fitted on
   inner folds only, and the threshold is chosen on those same inner predictions,
   so neither the validation fold nor the holdout ever select anything.
3. The deployable artifact is refitted on the full development partition, with the
   same inner-fold calibration and threshold rule, and is then scored once on the
   held-out 20 percent. That holdout number is the honest estimate; the CV numbers
   are the stable headline because all 1000 rows participate.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import Config
from credit_risk.data.fetch import sha256_file
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
)
from credit_risk.features.build import feature_frame, load_processed
from credit_risk.models import registry
from credit_risk.models.bundle import CalibratedModel
from credit_risk.models.metrics import (
    always_approve_cost,
    always_reject_cost,
    classification_metrics,
    metrics_with_row_thresholds,
    select_threshold,
)
from credit_risk.monitor.drift import build_reference

NUMERIC_FEATURES: list[str] = list(NUMERIC_COLUMNS + DERIVED_COLUMNS)
MEAN_SD_KEYS = (
    "roc_auc",
    "pr_auc",
    "ks",
    "brier",
    "accuracy_at_threshold",
    "expected_cost_per_1000",
    "cost_0_5_per_1000",
    "approval_rate",
)


def make_pipeline(model_type: str, cfg: Config) -> Pipeline:
    """One-hot encoded columns, scaled only for the logistic baseline."""
    if model_type == "logistic":
        numeric = Pipeline([("scale", StandardScaler())])
        model = LogisticRegression(random_state=cfg.seed, **cfg.train.logistic)
    elif model_type == "lightgbm":
        numeric = "passthrough"
        model = lightgbm.LGBMClassifier(random_state=cfg.seed, **cfg.train.lightgbm)
    else:
        raise ValueError(f"unknown model type: {model_type}")

    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float64
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_calibrated(
    estimator: Pipeline,
    frame: pd.DataFrame,
    cfg: Config,
    model_type: str,
) -> CalibratedModel:
    """Fit on ``frame``, calibrating and thresholding on inner-fold predictions only."""
    features = feature_frame(frame)
    target = frame[LABEL_COLUMN].to_numpy(dtype=int)

    inner = StratifiedKFold(
        n_splits=cfg.calibration.inner_splits, shuffle=True, random_state=cfg.seed
    )
    oof = np.full(len(target), np.nan, dtype=float)
    for train_idx, valid_idx in inner.split(features, target):
        inner_model = clone(estimator)
        inner_model.fit(features.iloc[train_idx], target[train_idx])
        oof[valid_idx] = inner_model.predict_proba(features.iloc[valid_idx])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("inner calibration loop left rows unscored")

    fitted = clone(estimator)
    fitted.fit(features, target)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof, target)
    inner_probability = np.clip(calibrator.predict(oof), 0.0, 1.0)
    selection = select_threshold(target, inner_probability, cfg.cost)
    return CalibratedModel(
        model_type=model_type,
        estimator=fitted,
        calibrator=calibrator,
        threshold=selection["threshold"],
        feature_columns=list(FEATURE_COLUMNS),
        threshold_detail=selection,
        calibration_method=cfg.calibration.method,
    )


def _prediction_frame(
    frame: pd.DataFrame, probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "y_default": frame[LABEL_COLUMN].to_numpy(),
            "p_bad": probability,
            "threshold": threshold,
            "decision": np.where(probability < threshold, "approve", "reject"),
        }
    )


def cross_validate(
    estimator: Pipeline,
    frame: pd.DataFrame,
    cfg: Config,
    model_type: str,
) -> tuple[dict, pd.DataFrame]:
    """Stratified k-fold on the development partition with per-fold calibration."""
    development = frame[frame["is_heldout"] == 0].reset_index(drop=True)
    folds = sorted(int(value) for value in development["fold"].unique())
    fold_metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for fold in folds:
        train_part = development[development["fold"] != fold]
        valid_part = development[development["fold"] == fold]
        model = fit_calibrated(estimator, train_part, cfg, model_type)
        probability = model.predict_proba_bad(valid_part)
        metrics = classification_metrics(
            valid_part[LABEL_COLUMN].to_numpy(), probability, model.threshold, cfg.cost
        )
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        prediction = _prediction_frame(valid_part, probability, model.threshold)
        prediction["fold"] = fold
        predictions.append(prediction)

    pooled_frame = pd.concat(predictions, ignore_index=True)
    pooled = metrics_with_row_thresholds(
        pooled_frame["y_default"], pooled_frame["p_bad"], pooled_frame["threshold"], cfg.cost
    )
    summary = {
        "n_splits": len(folds),
        "folds": fold_metrics,
        "mean": {key: float(np.mean([m[key] for m in fold_metrics])) for key in MEAN_SD_KEYS},
        "sd": {
            key: float(np.std([m[key] for m in fold_metrics], ddof=1))
            if len(fold_metrics) > 1
            else 0.0
            for key in MEAN_SD_KEYS
        },
        "pooled": pooled,
        "thresholds": [float(m["threshold"]) for m in fold_metrics],
    }
    return summary, pooled_frame


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def _raw_sha256(cfg: Config) -> str:
    if cfg.manifest_file.exists():
        manifest = json.loads(cfg.manifest_file.read_text(encoding="utf-8"))
        return str(manifest.get("sha256", "unknown"))
    if cfg.raw_file.exists():
        return sha256_file(cfg.raw_file)
    return "unknown"


def _metadata(
    cfg: Config,
    model_type: str,
    version: str,
    model: CalibratedModel,
    holdout_metrics: dict,
    cv_summary: dict,
    development_rows: int,
    holdout_rows: int,
) -> dict:
    return {
        "version": version,
        "model_type": model_type,
        "calibration": model.calibration_method,
        "seed": cfg.seed,
        "config_hash": cfg.config_hash,
        "data_sha256": _raw_sha256(cfg),
        "processed_sha256": sha256_file(cfg.processed_file),
        "git_commit": registry.git_commit(cfg.root),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": cfg.train.label,
        "features": list(FEATURE_COLUMNS),
        "encoded_features": len(model.feature_names),
        "threshold": model.threshold,
        "threshold_detail": model.threshold_detail,
        "cost_matrix": {
            "cost_bad_approved": cfg.cost.cost_bad_approved,
            "cost_good_rejected": cfg.cost.cost_good_rejected,
            "decision_rule": "approve when calibrated P(default) < threshold",
        },
        "split": {
            "protocol": "frozen stratified 80/20, 5-fold CV on the development partition",
            "development_rows": development_rows,
            "holdout_rows": holdout_rows,
            "seed": cfg.seed,
        },
        "metrics": {"holdout": holdout_metrics, "cv": cv_summary},
        "package_versions": _package_versions(),
    }


def run_training(cfg: Config, version: str | None = None) -> dict:
    """Fit baseline and main model, register both, and return the run summary."""
    frame = load_processed(cfg)
    development = frame[frame["is_heldout"] == 0].reset_index(drop=True)
    holdout = frame[frame["is_heldout"] == 1].reset_index(drop=True)
    if development.empty or holdout.empty:
        raise ValueError("processed frame has an empty development or holdout partition")

    run_version = version or registry.new_version()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    reference = build_reference(development, cfg.drift.bins)
    summary: dict = {
        "version": run_version,
        "seed": cfg.seed,
        "config_hash": cfg.config_hash,
        "deployed_model": cfg.train.deployed_model,
        "rows": {
            "total": int(len(frame)),
            "development": int(len(development)),
            "holdout": int(len(holdout)),
            "bad_rate": float(frame[LABEL_COLUMN].mean()),
        },
        "models": {},
    }

    holdout_target = holdout[LABEL_COLUMN].to_numpy()
    summary["policies"] = {
        "always_approve_per_1000": always_approve_cost(holdout_target, cfg.cost),
        "always_reject_per_1000": always_reject_cost(holdout_target, cfg.cost),
        "default_threshold": 0.5,
    }

    for model_type in ("logistic", "lightgbm"):
        estimator = make_pipeline(model_type, cfg)
        cv_summary, cv_predictions = cross_validate(estimator, frame, cfg, model_type)

        model = fit_calibrated(estimator, development, cfg, model_type)
        probability = model.predict_proba_bad(holdout)
        holdout_metrics = classification_metrics(
            holdout_target, probability, model.threshold, cfg.cost
        )
        holdout_predictions = _prediction_frame(holdout, probability, model.threshold)

        metadata = _metadata(
            cfg,
            model_type,
            run_version,
            model,
            holdout_metrics,
            cv_summary,
            len(development),
            len(holdout),
        )
        directory = registry.save_artifact(
            cfg, model_type, run_version, model, metadata, reference
        )

        cv_path = cfg.reports_dir / f"predictions_cv_{model_type}.csv"
        holdout_path = cfg.reports_dir / f"predictions_holdout_{model_type}.csv"
        cv_predictions.to_csv(cv_path, index=False)
        holdout_predictions.to_csv(holdout_path, index=False)

        summary["models"][model_type] = {
            "artifact": str(directory),
            "threshold": model.threshold,
            "threshold_detail": model.threshold_detail,
            "holdout": holdout_metrics,
            "cv": {
                "mean": cv_summary["mean"],
                "sd": cv_summary["sd"],
                "pooled": cv_summary["pooled"],
            },
            "cv_thresholds": cv_summary["thresholds"],
            "predictions": {"cv": str(cv_path), "holdout": str(holdout_path)},
        }

    registry.write_latest(cfg, cfg.train.deployed_model, run_version)
    summary["latest"] = f"{cfg.train.deployed_model}/{run_version}"
    cfg.train_summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"version {summary['version']}  seed {summary['seed']}",
        f"rows: {summary['rows']['total']} total, {summary['rows']['development']} development,"
        f" {summary['rows']['holdout']} holdout, bad rate {summary['rows']['bad_rate']:.3f}",
        "",
        f"{'model':10s} {'roc_auc':>8s} {'pr_auc':>8s} {'ks':>7s} {'brier':>7s}"
        f" {'acc':>6s} {'cost/1k':>9s} {'cost@.5':>9s}",
    ]
    for model_type, entry in summary["models"].items():
        holdout = entry["holdout"]
        lines.append(
            f"{model_type:10s} {holdout['roc_auc']:8.4f} {holdout['pr_auc']:8.4f}"
            f" {holdout['ks']:7.4f} {holdout['brier']:7.4f}"
            f" {holdout['accuracy_at_threshold']:6.3f}"
            f" {holdout['expected_cost_per_1000']:9.1f} {holdout['cost_0_5_per_1000']:9.1f}"
        )
    lines.append("")
    lines.append(f"holdout, always approve: {summary['policies']['always_approve_per_1000']:.1f}"
                 f" per 1000; always reject: {summary['policies']['always_reject_per_1000']:.1f}")
    lines.append(f"latest -> {summary['latest']}")
    return "\n".join(lines)


def artifact_path(cfg: Config) -> Path:
    model_type, version = registry.read_latest(cfg)
    return registry.artifact_dir(cfg, model_type, version)
