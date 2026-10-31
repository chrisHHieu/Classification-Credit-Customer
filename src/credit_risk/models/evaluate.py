"""Evaluation report: metrics, cost comparison, figures.

Everything here is computed from the prediction files written by ``train`` and
from the registered artifact, never from a fresh fit. That keeps the reported
numbers identical to the ones the model was selected with, and lets the report be
regenerated cheaply.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from credit_risk.config import Config  # noqa: E402
from credit_risk.data.fetch import sha256_file  # noqa: E402
from credit_risk.data.schema import FEATURE_LABELS, LABEL_COLUMN  # noqa: E402
from credit_risk.features.build import feature_frame, load_processed  # noqa: E402
from credit_risk.models import registry  # noqa: E402
from credit_risk.models.bundle import CalibratedModel  # noqa: E402
from credit_risk.models.metrics import (  # noqa: E402
    always_approve_cost,
    always_reject_cost,
    classification_metrics,
    cost_curve,
    expected_cost_per_1000,
    metrics_with_row_thresholds,
)

MODEL_ORDER = ("logistic", "lightgbm")
COLORS = {"logistic": "#4c72b0", "lightgbm": "#c44e52"}


class EvaluationError(RuntimeError):
    """Raised when the inputs for the report are missing."""


def predictions_path(cfg: Config, kind: str, model_type: str) -> Path:
    return cfg.reports_dir / f"predictions_{kind}_{model_type}.csv"


def load_predictions(cfg: Config, kind: str, model_type: str) -> pd.DataFrame:
    path = predictions_path(cfg, kind, model_type)
    if not path.exists():
        raise EvaluationError(f"{path} not found, run 'credit_risk train' first")
    return pd.read_csv(path)


def cv_summary(predictions: pd.DataFrame, cfg: Config) -> dict:
    """Recompute the fold table from saved cross-validation predictions."""
    fold_metrics = []
    for fold, part in predictions.groupby("fold", sort=True):
        threshold = float(part["threshold"].iloc[0])
        metrics = classification_metrics(part["y_default"], part["p_bad"], threshold, cfg.cost)
        metrics["fold"] = int(fold)
        fold_metrics.append(metrics)
    keys = (
        "roc_auc",
        "pr_auc",
        "ks",
        "brier",
        "accuracy_at_threshold",
        "expected_cost_per_1000",
        "cost_0_5_per_1000",
        "approval_rate",
    )
    pooled = metrics_with_row_thresholds(
        predictions["y_default"], predictions["p_bad"], predictions["threshold"], cfg.cost
    )
    return {
        "n_splits": int(len(fold_metrics)),
        "folds": fold_metrics,
        "mean": {key: float(np.mean([row[key] for row in fold_metrics])) for key in keys},
        "sd": {
            key: float(np.std([row[key] for row in fold_metrics], ddof=1))
            for key in keys
        },
        "pooled": pooled,
        "thresholds": [float(row["threshold"]) for row in fold_metrics],
    }


def permutation_table(
    bundle: CalibratedModel,
    frame: pd.DataFrame,
    cfg: Config,
    top_k: int = 10,
) -> list[dict]:
    """Permutation importance for ROC-AUC on the held-out rows."""
    features = feature_frame(frame)
    target = frame[LABEL_COLUMN].to_numpy()
    result = permutation_importance(
        bundle.estimator,
        features,
        target,
        scoring="roc_auc",
        n_repeats=cfg.train.permutation_repeats,
        random_state=cfg.seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": column,
            "label": FEATURE_LABELS.get(column, column),
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for column, mean, std in zip(
            features.columns, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)[:top_k]


def _reliability(y: np.ndarray, probability: np.ndarray, bins: int = 8):
    fraction_positive, mean_predicted = calibration_curve(
        y, probability, n_bins=bins, strategy="quantile"
    )
    return mean_predicted, fraction_positive


def plot_roc_pr(holdout: dict[str, pd.DataFrame], path: Path) -> Path:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model_type, frame in holdout.items():
        y = frame["y_default"].to_numpy()
        p = frame["p_bad"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        axes[0].plot(
            fpr,
            tpr,
            color=COLORS.get(model_type),
            label=f"{model_type} (AUC {auc:.3f})",
        )
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(
            recall,
            precision,
            color=COLORS.get(model_type),
            label=f"{model_type} (AP {average_precision_score(y, p):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC, held-out 20 percent")
    axes[1].axhline(
        float(np.mean(next(iter(holdout.values()))["y_default"])),
        linestyle=":",
        color="grey",
        linewidth=1,
        label="bad rate",
    )
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall, held-out 20 percent")
    for axis in axes:
        axis.legend(fontsize=8, frameon=False)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(
    bundle: CalibratedModel, frame: pd.DataFrame, path: Path, bins: int = 8
) -> Path:
    y = frame[LABEL_COLUMN].to_numpy()
    raw = bundle.raw_score(frame)
    calibrated = bundle.predict_proba_bad(frame)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, probability, title in (
        (axes[0], raw, "before calibration"),
        (axes[1], calibrated, "after isotonic calibration"),
    ):
        mean_predicted, fraction_positive = _reliability(y, probability, bins)
        axis.plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1)
        axis.plot(mean_predicted, fraction_positive, marker="o", color=COLORS["lightgbm"])
        axis.set_xlabel("mean predicted P(default)")
        axis.set_ylabel("observed share of defaults")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("Calibration on the held-out 20 percent", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_cost(
        holdout: dict[str, pd.DataFrame],
        cfg: Config,
        thresholds: dict[str, float],
        path: Path,
    ) -> Path:
        fig, axis = plt.subplots(figsize=(8, 4.4))
        for model_type, frame in holdout.items():
            curve = cost_curve(frame["y_default"], frame["p_bad"], cfg.cost, points=120)
            axis.plot(
                [row["threshold"] for row in curve],
                [row["expected_cost_per_1000"] for row in curve],
                color=COLORS.get(model_type),
                label=model_type,
            )
            axis.axvline(
            thresholds[model_type],
            color=COLORS.get(model_type),
            linestyle="--",
            linewidth=1,
        )

        y_reference = next(iter(holdout.values()))["y_default"].to_numpy()
        axis.axhline(
            always_approve_cost(y_reference, cfg.cost),
            color="grey",
            linestyle=":",
            linewidth=1,
            label="always approve",
        )
        axis.axhline(
            always_reject_cost(y_reference, cfg.cost),
            color="black",
            linestyle=":",
            linewidth=1,
            label="always reject",
        )

        low, high = axis.get_ylim()
        axis.set_ylim(low, high + 0.08 * (high - low))
        for index, (model_type, chosen) in enumerate(thresholds.items()):
            axis.annotate(
                f"{model_type} threshold {chosen:.3f}",
                xy=(chosen, high - (0.12 + 0.10 * index) * (high - low)),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=COLORS.get(model_type),
            )

        axis.set_xlabel("threshold on calibrated P(default)")
        axis.set_ylabel("expected cost per 1000 applications")
        axis.set_title("Cost curve, held-out 20 percent (5:1 cost ratio)")
        axis.legend(fontsize=8, frameon=False, loc="lower left")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def confusion_matrix_array(metrics: dict) -> np.ndarray:
    """2x2 counts with actual outcomes as rows and decisions as columns."""
    counts = metrics["confusion"]
    return np.array(
        [
            [counts["true_good_approved"], counts["true_good_rejected"]],
            [counts["bad_approved"], counts["bad_rejected"]],
        ]
    )


def plot_confusion(metrics: dict, model_type: str, path: Path) -> Path:
    matrix = confusion_matrix_array(metrics)
    fig, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    labels = [["good, approved", "good, rejected"], ["bad, approved", "bad, rejected"]]
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{labels[row][column]}\n{value}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > matrix.max() * 0.6 else "black",
            )
    axis.set_xticks([0, 1], ["decision: approve", "decision: reject"])
    axis.set_yticks([0, 1], ["actual: good", "actual: bad"])
    axis.set_title(
        f"{model_type}, threshold {metrics['threshold']:.3f}\n"
        f"cost {metrics['expected_cost_per_1000']:.1f} per 1000",
        fontsize=9,
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_importance(table: list[dict], model_type: str, path: Path) -> Path:
    rows = list(reversed(table))
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.barh(
        [row["label"] for row in rows],
        [row["importance_mean"] for row in rows],
        xerr=[row["importance_std"] for row in rows],
        color=COLORS.get(model_type),
        alpha=0.85,
    )
    axis.set_xlabel("drop in ROC-AUC when the feature is permuted")
    axis.set_title(f"Permutation importance, {model_type}, held-out rows", fontsize=9)
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _cost_row(policy: str, y: np.ndarray, p: np.ndarray, threshold: float, cfg: Config) -> dict:
    approval = float((p < threshold).mean())
    return {
        "policy": policy,
        "threshold": float(threshold),
        "expected_cost_per_1000": expected_cost_per_1000(y, p, threshold, cfg.cost),
        "approval_rate": approval,
    }


def run_evaluation(cfg: Config) -> dict:
    """Build ``reports/metrics.json`` plus the figures."""
    model_type, version = registry.read_latest(cfg)
    metadata = registry.load_metadata(cfg, model_type, version)
    bundle, _ = registry.load_bundle(cfg, model_type, version)
    frame = load_processed(cfg)
    holdout_frame = frame[frame["is_heldout"] == 1].reset_index(drop=True)
    development_frame = frame[frame["is_heldout"] == 0].reset_index(drop=True)
    if holdout_frame.empty:
        raise EvaluationError("no held-out rows in the processed file")

    holdout_predictions: dict[str, pd.DataFrame] = {}
    cv_predictions: dict[str, pd.DataFrame] = {}
    holdout_metrics: dict[str, dict] = {}
    cv_results: dict[str, dict] = {}
    thresholds: dict[str, float] = {}

    for candidate in MODEL_ORDER:
        holdout_predictions[candidate] = load_predictions(cfg, "holdout", candidate)
        cv_predictions[candidate] = load_predictions(cfg, "cv", candidate)
        threshold = float(holdout_predictions[candidate]["threshold"].iloc[0])
        thresholds[candidate] = threshold
        holdout_metrics[candidate] = classification_metrics(
            holdout_predictions[candidate]["y_default"],
            holdout_predictions[candidate]["p_bad"],
            threshold,
            cfg.cost,
        )
        cv_results[candidate] = cv_summary(cv_predictions[candidate], cfg)

    if len(holdout_predictions[model_type]) != len(holdout_frame):
        raise EvaluationError(
            "holdout predictions do not match the processed holdout rows, rerun train"
        )

    y_holdout = holdout_predictions[model_type]["y_default"].to_numpy()
    p_holdout = {
        name: holdout_predictions[name]["p_bad"].to_numpy() for name in holdout_predictions
    }

    cost_rows = []
    for candidate in MODEL_ORDER:
        frame_predictions = holdout_predictions[candidate]
        cost_rows.append(
            _cost_row(
                f"{candidate} at chosen threshold",
                frame_predictions["y_default"].to_numpy(),
                frame_predictions["p_bad"].to_numpy(),
                thresholds[candidate],
                cfg,
            )
        )
        cost_rows.append(
            _cost_row(
                f"{candidate} at default 0.5",
                frame_predictions["y_default"].to_numpy(),
                frame_predictions["p_bad"].to_numpy(),
                0.5,
                cfg,
            )
        )
    cost_rows.append(
        {
            "policy": "always approve",
            "threshold": None,
            "expected_cost_per_1000": always_approve_cost(y_holdout, cfg.cost),
            "approval_rate": 1.0,
        }
    )
    cost_rows.append(
        {
            "policy": "always reject",
            "threshold": None,
            "expected_cost_per_1000": always_reject_cost(y_holdout, cfg.cost),
            "approval_rate": 0.0,
        }
    )
    grid = np.linspace(cfg.cost.threshold_min, cfg.cost.threshold_max, 200)
    oracle_costs = np.array(
        [expected_cost_per_1000(y_holdout, p_holdout[model_type], t, cfg.cost) for t in grid]
    )
    oracle_index = int(np.argmin(oracle_costs))
    cost_rows.append(
        {
            "policy": "oracle threshold (uses holdout labels, not deployable)",
            "threshold": float(grid[oracle_index]),
            "expected_cost_per_1000": float(oracle_costs[oracle_index]),
            "approval_rate": float((p_holdout[model_type] < grid[oracle_index]).mean()),
        }
    )

    importance = permutation_table(bundle, holdout_frame, cfg)

    raw_probability = bundle.raw_score(holdout_frame)
    calibrated_probability = bundle.predict_proba_bad(holdout_frame)
    from sklearn.metrics import brier_score_loss

    calibration_block = {
        "model_type": model_type,
        "brier_raw": float(brier_score_loss(y_holdout, raw_probability)),
        "brier_calibrated": float(brier_score_loss(y_holdout, calibrated_probability)),
        "method": metadata.get("calibration", "isotonic"),
        "fitted_on": "inner-fold out-of-fold predictions of the development partition",
        "rows": int(len(holdout_frame)),
    }

    figures_dir = cfg.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        plot_roc_pr(holdout_predictions, figures_dir / "roc_pr_holdout.png"),
        plot_calibration(bundle, holdout_frame, figures_dir / "calibration_holdout.png"),
        plot_cost(holdout_predictions, cfg, thresholds, figures_dir / "cost_curve_holdout.png"),
        plot_confusion(
            holdout_metrics[model_type], model_type, figures_dir / "confusion_holdout.png"
        ),
        plot_importance(importance, model_type, figures_dir / "permutation_importance.png"),
    ]

    reported = metadata.get("metrics", {}).get("holdout", {})
    consistency = {
        "roc_auc_matches_metadata": bool(
            np.isclose(
                reported.get("roc_auc", np.nan), holdout_metrics[model_type]["roc_auc"], atol=1e-9
            )
        ),
        "expected_cost_matches_metadata": bool(
            np.isclose(
                reported.get("expected_cost_per_1000", np.nan),
                holdout_metrics[model_type]["expected_cost_per_1000"],
                atol=1e-6,
            )
        ),
        "holdout_rows": int(len(holdout_frame)),
        "development_rows": int(len(development_frame)),
    }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "type": model_type,
            "version": version,
            "threshold": thresholds[model_type],
            "calibration": metadata.get("calibration"),
            "trained_at": metadata.get("trained_at"),
            "git_commit": metadata.get("git_commit"),
            "config_hash": metadata.get("config_hash"),
        },
        "data": {
            "rows_total": int(len(frame)),
            "rows_holdout": int(len(holdout_frame)),
            "raw_sha256": metadata.get("data_sha256"),
            "processed_sha256": sha256_file(cfg.processed_file),
        },
        "protocol": {
            "headline": "stratified 5-fold CV on the 80 percent development partition",
            "holdout": "single stratified 20 percent split, frozen before training",
            "calibration": "isotonic, fitted on inner 3-fold out-of-fold predictions",
            "threshold": "minimum expected cost on inner-fold predictions only, 5:1 ratio",
            "cost_matrix": {
                "approve_bad": cfg.cost.cost_bad_approved,
                "reject_good": cfg.cost.cost_good_rejected,
            },
        },
        "holdout": holdout_metrics,
        "cv": cv_results,
        "cost_table": cost_rows,
        "calibration": calibration_block,
        "permutation_importance": importance,
        "consistency": consistency,
        "figures": [path.relative_to(cfg.root).as_posix() for path in figure_paths],
    }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def format_evaluation(payload: dict) -> str:
    lines = [
        f"model {payload['model']['type']}/{payload['model']['version']}"
        f"  threshold {payload['model']['threshold']:.4f}",
        "",
        f"{'model':10s} {'split':10s} {'roc_auc':>8s} {'pr_auc':>8s} {'ks':>7s}"
        f" {'brier':>7s} {'acc':>6s} {'cost/1k':>9s}",
    ]
    for name, entry in payload["holdout"].items():
        lines.append(
            f"{name:10s} {'holdout':10s} {entry['roc_auc']:8.4f} {entry['pr_auc']:8.4f}"
            f" {entry['ks']:7.4f} {entry['brier']:7.4f} {entry['accuracy_at_threshold']:6.3f}"
            f" {entry['expected_cost_per_1000']:9.1f}"
        )
    for name, entry in payload["cv"].items():
        mean = entry["mean"]
        lines.append(
            f"{name:10s} {'cv mean':10s} {mean['roc_auc']:8.4f} {mean['pr_auc']:8.4f}"
            f" {mean['ks']:7.4f} {mean['brier']:7.4f} {mean['accuracy_at_threshold']:6.3f}"
            f" {mean['expected_cost_per_1000']:9.1f}"
        )
    lines.append("")
    for row in payload["cost_table"]:
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.4f}"
        lines.append(
            f"{row['policy']:52s} threshold {threshold:>6s}"
            f"  cost/1k {row['expected_cost_per_1000']:8.1f}"
            f"  approval {row['approval_rate']:.3f}"
        )
    lines.append("")
    lines.append(f"figures: {len(payload['figures'])} written under {payload['figures'][0]}")
    return "\n".join(lines)
