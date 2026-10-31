"""The object stored in the registry: estimator, calibrator and decision threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data.schema import describe_feature, feature_name_parts
from credit_risk.models.candidates import CATEGORY_INDICES, NATIVE_CANDIDATES
from credit_risk.models.conformal import ConformalCalibrator


class CalibratedModel:
    """A fitted pipeline plus a calibrator, a cost threshold and a conformal band.

    ``predict_proba_bad`` returns calibrated P(default). ``contributions`` returns
    per-feature values on the model's log-odds scale: exact tree SHAP for the tree
    candidates (LightGBM ``pred_contrib``, XGBoost ``pred_contribs``, CatBoost
    ``ShapValues``) and ``coefficient * encoded value`` for the logistic baseline.
    They explain the base score, and the calibrator is a monotone transform of it,
    so the sign and order of the contributions are unaffected by calibration.
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
        family: str | None = None,
        encoding: str | None = None,
        conformal: ConformalCalibrator | None = None,
        calibration_table: list[dict] | None = None,
    ) -> None:
        self.model_type = model_type
        self.estimator = estimator
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.feature_columns = list(feature_columns)
        self.threshold_detail = threshold_detail or {}
        self.calibration_method = calibration_method
        self.family = family or ("linear" if model_type == "logistic" else "tree")
        self.encoding = encoding or ("native" if model_type in NATIVE_CANDIDATES else "one_hot")
        self.conformal = conformal
        self.calibration_table = list(calibration_table or [])

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

    def conformal_decision(self, frame: pd.DataFrame) -> np.ndarray:
        """Approve / review / reject from the conformal prediction sets."""
        if self.conformal is None:
            return self.decide(frame)
        return self.conformal.decisions(self.predict_proba_bad(frame))

    # Explanation ----------------------------------------------------------------
    @property
    def transformer(self):
        return self.estimator.named_steps["preprocess"]

    @property
    def feature_names(self) -> list[str]:
        return [str(name) for name in self.transformer.get_feature_names_out()]

    def model_input(self, frame: pd.DataFrame):
        """Whatever the estimator consumes: an encoded matrix or a category frame."""
        return self.transformer.transform(frame[self.feature_columns])

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """(rows, encoded features) contribution matrix on the log-odds scale."""
        matrix = self.model_input(frame)
        if self.model_type == "logistic":
            encoded = np.asarray(matrix, dtype=float)
            coef = np.asarray(self.estimator.named_steps["model"].coef_, dtype=float).ravel()
            return encoded * coef

        if self.model_type == "lightgbm":
            booster = self.estimator.named_steps["model"].booster_
            values = np.asarray(booster.predict(matrix, pred_contrib=True), dtype=float)
            return values[:, :-1]

        if self.model_type == "xgboost":
            import xgboost

            model = self.estimator.named_steps["model"]
            dmatrix = xgboost.DMatrix(matrix, enable_categorical=True)
            values = np.asarray(
                model.get_booster().predict(dmatrix, pred_contribs=True), dtype=float
            )
            return values[:, :-1]

        if self.model_type == "catboost":
            from catboost import Pool

            wrapper = self.estimator.named_steps["model"]
            model = getattr(wrapper, "model_", wrapper)
            values = np.asarray(
                model.get_feature_importance(
                    Pool(matrix, cat_features=CATEGORY_INDICES), type="ShapValues"
                ),
                dtype=float,
            )
            return values[:, :-1]

        raise ValueError(f"no contribution method for model type {self.model_type}")

    def reason_codes(self, row: pd.DataFrame, top_k: int = 3) -> list[dict]:
        """Top attribute-level contributions, with readable feature names.

        One-hot columns of the same attribute are summed, so a single attribute
        cannot fill the list twice. Native categorical models already contribute
        one value per raw attribute.
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
