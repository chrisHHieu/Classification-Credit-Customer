"""The object stored in the registry: estimator, calibrator and decision threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data.schema import describe_feature, feature_name_parts


class CalibratedModel:
    """A fitted pipeline plus an isotonic calibrator and a cost-based threshold.

    ``predict_proba_bad`` returns calibrated P(default). ``contributions`` returns
    per-feature values on the model's log-odds scale. For LightGBM these are exact
    tree SHAP values from ``pred_contrib=True``; for the logistic baseline they are
    ``coefficient * encoded value``. They explain the base score, and the calibrator
    is a monotone transform of it, so the sign and order of the contributions are
    unaffected by calibration.
    """

    def __init__(
        self,
        model_type: str,
        estimator,
        calibrator,
        threshold: float,
        feature_columns: list[str],
        threshold_detail: dict | None = None,
        calibration_method: str = "isotonic",
    ) -> None:
        self.model_type = model_type
        self.estimator = estimator
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.feature_columns = list(feature_columns)
        self.threshold_detail = threshold_detail or {}
        self.calibration_method = calibration_method

    # Scoring -------------------------------------------------------------------
    def raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Base model P(default), before calibration."""
        proba = self.estimator.predict_proba(frame[self.feature_columns])
        return np.asarray(proba)[:, 1].astype(float)

    def predict_proba_bad(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(frame)
        if self.calibrator is None:
            return raw
        return np.clip(np.asarray(self.calibrator.predict(raw), dtype=float), 0.0, 1.0)

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        """``approve`` where P(default) is below the threshold."""
        p = self.predict_proba_bad(frame)
        return np.where(p < self.threshold, "approve", "reject")

    # Explanation ----------------------------------------------------------------
    @property
    def transformer(self):
        return self.estimator.named_steps["preprocess"]

    @property
    def feature_names(self) -> list[str]:
        return [str(name) for name in self.transformer.get_feature_names_out()]

    def encoded(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.transformer.transform(frame[self.feature_columns])
        return np.asarray(matrix, dtype=float)

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """(rows, encoded features) contribution matrix on the log-odds scale."""
        matrix = self.encoded(frame)
        if self.model_type == "lightgbm":
            booster = self.estimator.named_steps["model"].booster_
            values = np.asarray(booster.predict(matrix, pred_contrib=True), dtype=float)
            return values[:, :-1]
        coef = np.asarray(self.estimator.named_steps["model"].coef_, dtype=float).ravel()
        return matrix * coef

    def reason_codes(self, row: pd.DataFrame, top_k: int = 3) -> list[dict]:
        """Top attribute-level contributions, with readable feature names.

        One-hot columns of the same attribute are summed, so a single attribute
        cannot fill the list twice.
        """
        values = self.contributions(row)[0]
        names = self.feature_names
        observed = row.iloc[0].to_dict()

        grouped: dict[str, dict] = {}
        for index, name in enumerate(names):
            source, code = feature_name_parts(name)
            entry = grouped.setdefault(
                source, {"contribution": 0.0, "code": None, "columns": 0}
            )
            entry["contribution"] += float(values[index])
            entry["columns"] += 1
            if code is not None and code == str(observed.get(source)):
                entry["code"] = code

        ranked = sorted(
            grouped.items(), key=lambda item: abs(item[1]["contribution"]), reverse=True
        )
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
