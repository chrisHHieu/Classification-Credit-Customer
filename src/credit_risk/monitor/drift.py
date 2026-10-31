"""Population stability index per feature against the training reference.

The reference is written by ``train`` from the development partition only. Drift
then compares any incoming frame with it. Bins are quantiles of the reference
distribution, so every bin holds roughly the same number of training rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import Config
from credit_risk.data.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from credit_risk.models import registry

# Small floor so an empty bin does not produce an infinite index.
EPSILON = 1e-4


def _bin_edges(values: np.ndarray, bins: int) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        edges = np.array([float(values.min()) - 1.0, float(values.max()) + 1.0])
    return [float(edge) for edge in edges]


def _numeric_proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    return counts / total if total else np.zeros(len(counts), dtype=float)


def build_reference(frame: pd.DataFrame, bins: int) -> dict:
    """Training distributions for every model feature."""
    features: dict[str, dict] = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column].astype(float).dropna().to_numpy()
        edges = _bin_edges(values, bins)
        features[column] = {
            "type": "numeric",
            "edges": edges,
            "proportions": _numeric_proportions(values, edges).round(6).tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    for column in CATEGORICAL_COLUMNS:
        shares = frame[column].astype(str).value_counts(normalize=True)
        features[column] = {
            "type": "categorical",
            "proportions": {str(key): round(float(value), 6) for key, value in shares.items()},
        }
    # Derived columns can be described with the same machinery as their raw source.
    for column in FEATURE_COLUMNS:
        if column in features:
            continue
        values = frame[column].astype(float).dropna().to_numpy()
        edges = _bin_edges(values, bins)
        features[column] = {
            "type": "numeric",
            "edges": edges,
            "proportions": _numeric_proportions(values, edges).round(6).tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "bins": bins,
        "features": features,
    }


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index, floored to avoid taking log(0)."""
    expected = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _status(value: float, cfg: Config) -> str:
    if value >= cfg.drift.alert:
        return "alert"
    if value >= cfg.drift.warn:
        return "warn"
    return "ok"


def feature_psi(frame: pd.DataFrame, reference: dict, cfg: Config) -> list[dict]:
    results: list[dict] = []
    for column in FEATURE_COLUMNS:
        spec = reference["features"].get(column)
        if spec is None:
            continue
        actual_series = frame[column]
        if spec["type"] == "numeric":
            values = actual_series.astype(float).fillna(0.0).to_numpy()
            actual = _numeric_proportions(values, spec["edges"])
            expected = np.asarray(spec["proportions"], dtype=float)
        else:
            counts = actual_series.astype(str).value_counts(normalize=True)
            expected_map = spec["proportions"]
            keep = [key for key, value in expected_map.items() if value >= 0.01]
            expected = np.array([expected_map[key] for key in keep] + [
                sum(value for key, value in expected_map.items() if key not in keep)
            ])
            actual = np.array([float(counts.get(key, 0.0)) for key in keep] + [
                float(sum(value for key, value in counts.items() if key not in keep))
            ])
        value = psi(expected, actual)
        results.append(
            {
                "feature": column,
                "type": spec["type"],
                "psi": round(value, 6),
                "status": _status(value, cfg),
            }
        )
    return sorted(results, key=lambda row: row["psi"], reverse=True)


def run_drift(
    cfg: Config,
    input_path: str | Path | None = None,
    split: str = "both",
) -> dict:
    """Compare a frame with the training reference and write ``reports/drift.json``."""
    model_type, version = registry.read_latest(cfg)
    reference = registry.load_reference(cfg, model_type, version)

    if input_path is not None:
        frame = pd.read_csv(input_path)
    else:
        from credit_risk.features.build import load_processed

        frame = load_processed(cfg)

    slices: dict[str, pd.DataFrame] = {}
    if split in {"all", "both"}:
        slices["all"] = frame
    if split in {"holdout", "both"} and "is_heldout" in frame.columns:
        slices["holdout"] = frame[frame["is_heldout"] == 1]
    if not slices:
        slices[split] = frame

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": {
            "model_type": model_type,
            "version": version,
            "rows": reference["rows"],
            "bins": reference["bins"],
        },
        "input": str(input_path) if input_path is not None else str(cfg.processed_file),
        "thresholds": {"warn": cfg.drift.warn, "alert": cfg.drift.alert},
        "slices": {},
    }

    for name, part in slices.items():
        features = feature_psi(part, reference, cfg)
        statuses = [row["status"] for row in features]
        overall = "alert" if "alert" in statuses else ("warn" if "warn" in statuses else "ok")
        payload["slices"][name] = {
            "rows": int(len(part)),
            "overall": overall,
            "max_psi": float(features[0]["psi"]) if features else 0.0,
            "alert_count": int(sum(1 for status in statuses if status == "alert")),
            "warn_count": int(sum(1 for status in statuses if status == "warn")),
            "features": features,
        }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.drift_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def format_drift(payload: dict) -> str:
    lines = [
        f"reference {payload['reference']['model_type']}/{payload['reference']['version']}"
        f" ({payload['reference']['rows']} rows)",
        f"input {payload['input']}",
    ]
    for name, block in payload["slices"].items():
        lines.append(
            f"slice {name}: {block['rows']} rows, overall {block['overall']},"
            f" max PSI {block['max_psi']:.4f}, alert {block['alert_count']},"
            f" warn {block['warn_count']}"
        )
        for row in block["features"][:5]:
            lines.append(f"  {row['feature']:24s} {row['psi']:8.4f} {row['status']}")
    return "\n".join(lines)
