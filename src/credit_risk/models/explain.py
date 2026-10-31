"""Per-applicant and global explanations.

Tree candidates go through ``shap.TreeExplainer``, which returns exact additive
contributions for the model the API actually serves. When ``shap`` is not
installed, the exact native implementations already used by the bundle are the
fallback (LightGBM ``pred_contrib``, XGBoost ``pred_contribs``, CatBoost
``ShapValues``), so predictions never lose their reason codes.

For the logistic baseline shap is not the natural tool: the fallback is the
coefficient multiplied by the encoded value per row, which is exact and additive
for a linear model, and permutation importance for the global ranking.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from credit_risk.data.schema import describe_feature, feature_name_parts


def shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


def _tree_values(bundle, frame: pd.DataFrame) -> np.ndarray:
    matrix = bundle.model_input(frame)
    named = bundle.estimator.named_steps["model"]
    model = getattr(named, "model_", named)
    if shap_available():
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:  # some versions return (rows, features, classes)
            array = array[:, :, -1]
        return array
    return bundle.contributions(frame)


def contributions(bundle, frame: pd.DataFrame) -> np.ndarray:
    """Contribution matrix aligned with ``bundle.feature_names``."""
    if bundle.family == "tree":
        values = _tree_values(bundle, frame)
        if values.shape[1] != len(bundle.feature_names):
            raise ValueError(
                f"shap returned {values.shape[1]} columns, expected {len(bundle.feature_names)}"
            )
        return values
    return bundle.contributions(frame)


def _grouped(bundle, values: np.ndarray, row: pd.Series | None) -> dict[str, dict]:
    names = bundle.feature_names
    observed = row.to_dict() if row is not None else {}
    grouped: dict[str, dict] = {}
    for index, name in enumerate(names):
        source, code = feature_name_parts(name)
        entry = grouped.setdefault(source, {"contribution": 0.0, "code": None})
        entry["contribution"] += float(values[index])
        if code is not None and code == str(observed.get(source)):
            entry["code"] = code
    return grouped


def explain_row(bundle, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Top ``top_k`` attributes for one applicant, largest absolute contribution first."""
    values = contributions(bundle, row)[0]
    grouped = _grouped(bundle, values, row.iloc[0])
    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]["contribution"]), reverse=True)
    codes = []
    for source, entry in ranked[:top_k]:
        contribution = entry["contribution"]
        display = f"{source}_{entry['code']}" if entry["code"] else source
        codes.append(
            {
                "feature": source,
                "label": describe_feature(display),
                "contribution_log_odds": round(contribution, 4),
                "direction": "raises_risk" if contribution > 0 else "lowers_risk",
            }
        )
    return codes


def global_importance(bundle, frame: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Mean absolute contribution per attribute over a frame."""
    values = contributions(bundle, frame)
    matrix = np.abs(np.asarray(values, dtype=float))
    grouped: dict[str, float] = {}
    for index, name in enumerate(bundle.feature_names):
        source, _ = feature_name_parts(name)
        grouped[source] = grouped.get(source, 0.0) + float(matrix[:, index].mean())
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:top_k]
    method = "mean absolute tree SHAP" if bundle.family == "tree" else "mean absolute coefficient"
    return [
        {
            "feature": source,
            "label": describe_feature(source),
            "importance": round(value, 4),
            "method": method,
        }
        for source, value in ranked
    ]


def permutation_importance_table(bundle, frame: pd.DataFrame, seed: int, repeats: int = 5):
    """Permutation importance for the linear fallback, computed on raw features."""
    from sklearn.inspection import permutation_importance

    from credit_risk.data.schema import FEATURE_COLUMNS, LABEL_COLUMN

    result = permutation_importance(
        bundle.estimator,
        frame[list(FEATURE_COLUMNS)],
        frame[LABEL_COLUMN].to_numpy(),
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": column,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for column, mean, std in zip(
            FEATURE_COLUMNS, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)
