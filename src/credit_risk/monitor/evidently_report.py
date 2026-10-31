"""Drift report with Evidently.

This is the second opinion next to the PSI monitor in ``drift.py``: Evidently runs
a per-column statistical test (Kolmogorov-Smirnov for numeric, chi-square for
categorical) and a target drift test on the label, and writes a self-contained
HTML report. The reference is the development partition, the current data is the
held-out partition unless ``--input`` names another file. Comparing the two
methods on the same slice is the point: PSI is a magnitude, Evidently is a test.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_risk.config import Config
from credit_risk.data.schema import FEATURE_COLUMNS, LABEL_COLUMN


def available() -> bool:
    return importlib.util.find_spec("evidently") is not None


def _require():
    if not available():
        raise RuntimeError(
            "evidently is not installed; install it with "
            "python -m pip install -e '.[modern]'"
        )
    from evidently import Report
    from evidently.presets import DataDriftPreset

    return Report, DataDriftPreset


def _parse(metrics: list[dict]) -> tuple[dict, list[dict]]:
    feature_block: dict = {}
    columns: list[dict] = []
    for metric in metrics:
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")
        config = metric.get("config", {}) or {}
        if name.startswith("DriftedColumnsCount"):
            scope = config.get("columns") or []
            block = {
                "drifted_columns": int(float(value.get("count", 0))) if value else 0,
                "drift_share": float(value.get("share", 0.0)) if value else 0.0,
                "total_columns": len(scope),
            }
            if scope == [LABEL_COLUMN]:
                feature_block["target"] = block
            else:
                feature_block["features"] = block
        elif name.startswith("ValueDrift"):
            column = config.get("column")
            threshold = float(config.get("threshold", 0.05))
            p_value = float(value)
            columns.append(
                {
                    "column": column,
                    "p_value": round(p_value, 6),
                    "test": config.get("method"),
                    "threshold": threshold,
                    "drifted": bool(p_value < threshold),
                }
            )
    return feature_block, columns


def run_evidently(cfg: Config, input_path: str | Path | None = None) -> dict:
    """Compare the development partition with the holdout (or a supplied file)."""
    Report, DataDriftPreset = _require()
    from credit_risk.features.build import load_processed

    processed = load_processed(cfg)
    reference = processed[processed["is_heldout"] == 0]
    if input_path is not None:
        current = pd.read_csv(input_path)
    else:
        current = processed[processed["is_heldout"] == 1]
    columns = [*FEATURE_COLUMNS, LABEL_COLUMN]
    missing = [column for column in columns if column not in current.columns]
    if missing:
        raise ValueError(f"input frame is missing columns {missing}")

    report = Report(
        [
            DataDriftPreset(columns=list(FEATURE_COLUMNS)),
            DataDriftPreset(columns=[LABEL_COLUMN]),
        ]
    )
    snapshot = report.run(
        reference_data=reference[columns], current_data=current[columns]
    )
    blocks, column_rows = _parse(snapshot.dict().get("metrics", []))

    features = blocks.get("features", {})
    feature_rows = [row for row in column_rows if row["column"] != LABEL_COLUMN]
    drifted = [row for row in feature_rows if row["drifted"]]
    target_rows = [row for row in column_rows if row["column"] == LABEL_COLUMN]

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(cfg.evidently_html))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "library": "evidently",
        "reference": {"rows": int(len(reference)), "split": "development"},
        "current": {
            "rows": int(len(current)),
            "source": str(input_path) if input_path else "holdout split",
        },
        "feature_drift": {
            "drifted_columns": int(features.get("drifted_columns", len(drifted))),
            "total_columns": int(features.get("total_columns", len(feature_rows))),
            "drift_share": float(features.get("drift_share", 0.0)),
            "drifted": drifted,
            "columns": sorted(feature_rows, key=lambda row: row["p_value"]),
        },
        "target_drift": {
            "column": LABEL_COLUMN,
            "drifted": bool(target_rows[0]["drifted"]) if target_rows else False,
            "p_value": target_rows[0]["p_value"] if target_rows else None,
            "test": target_rows[0]["test"] if target_rows else None,
        },
        "thresholds": {"p_value": 0.05},
        "html": str(cfg.evidently_html.relative_to(cfg.root)),
    }
    cfg.evidently_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def format_evidently(payload: dict) -> str:
    feature = payload["feature_drift"]
    target = payload["target_drift"]
    lines = [
        f"evidently: reference {payload['reference']['rows']} rows,"
        f" current {payload['current']['rows']} rows",
        f"features: {feature['drifted_columns']} of {feature['total_columns']} drifted"
        f" (share {feature['drift_share']:.3f}, p < 0.05)",
        f"target {target['column']}: drifted {target['drifted']}"
        f" (p {target['p_value']})",
    ]
    for row in feature["drifted"]:
        lines.append(f"  {row['column']:26s} p={row['p_value']:.4f} ({row['test']})")
    lines.append(f"html {payload['html']}")
    return "\n".join(lines)
