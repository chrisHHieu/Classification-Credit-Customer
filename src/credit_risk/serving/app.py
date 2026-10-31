"""FastAPI service for credit default decisions.

The service is stateless apart from the loaded artifact. ``/health`` reports
whether an artifact could be loaded; ``/predict`` returns 503 instead of guessing
when the registry is empty. Reason codes come from the registered model itself:
LightGBM's ``pred_contrib=True`` (exact tree SHAP, no extra dependency) or the
logistic coefficients, mapped back to readable applicant attributes.
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from credit_risk import __version__
from credit_risk.config import Config, ConfigError, load_config
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    VALID_CODES,
)
from credit_risk.features.build import add_derived_features
from credit_risk.models import registry

MAX_BATCH = 500


class ModelUnavailable(RuntimeError):
    """No usable artifact in the registry."""


class Applicant(BaseModel):
    """The 20 raw Statlog attributes for one applicant."""

    model_config = ConfigDict(extra="forbid")

    checking_status: str = Field(description="A11, A12, A13 or A14")
    duration_months: int = Field(ge=4, le=72)
    credit_history: str = Field(description="A30 to A34")
    purpose: str = Field(description="A40 to A49 or A410")
    credit_amount: float = Field(gt=0, le=100000)
    savings: str = Field(description="A61 to A65")
    employment_since: str = Field(description="A71 to A75")
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str = Field(description="A91 to A95")
    other_debtors: str = Field(description="A101 to A103")
    residence_since: int = Field(ge=1, le=4)
    property: str = Field(description="A121 to A124")
    age_years: int = Field(ge=18, le=100)
    other_installment_plans: str = Field(description="A141 to A143")
    housing: str = Field(description="A151 to A153")
    existing_credits: int = Field(ge=1, le=4)
    job: str = Field(description="A171 to A174")
    people_liable: int = Field(ge=1, le=2)
    telephone: str = Field(description="A191 or A192")
    foreign_worker: str = Field(description="A201 or A202")

    @field_validator(*CATEGORICAL_COLUMNS)
    @classmethod
    def known_code(cls, value: str, info) -> str:
        allowed = VALID_CODES[info.field_name]
        if value not in allowed:
            raise ValueError(f"unknown code for {info.field_name}, expected one of {allowed}")
        return value


class ReasonCode(BaseModel):
    feature: str
    label: str
    contribution_log_odds: float
    direction: Literal["raises_risk", "lowers_risk"]


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "reject"]
    threshold: float
    expected_cost_approve: float
    expected_cost_reject: float
    reason_codes: list[ReasonCode]
    model_type: str
    model_version: str
    calibration: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: list[Applicant] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class ModelStore:
    """Lazily loads ``models/latest`` once and keeps it for the process lifetime."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._bundle = None
        self._metadata: dict | None = None
        self._error: str | None = None

    def load(self) -> None:
        with self._lock:
            if self._bundle is not None:
                return
            try:
                self._bundle, self._metadata = registry.load_bundle(self.cfg)
                self._error = None
            except (registry.RegistryError, OSError, ValueError) as exc:
                self._error = str(exc)
                raise ModelUnavailable(self._error) from exc

    @property
    def bundle(self):
        if self._bundle is None:
            self.load()
        return self._bundle

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self.load()
        return self._metadata or {}

    @property
    def error(self) -> str | None:
        return self._error


def _resolve_config() -> Config:
    env_path = os.environ.get("CREDIT_RISK_CONFIG")
    if env_path:
        return load_config(env_path)
    try:
        return load_config()
    except ConfigError:
        root = Path(__file__).resolve().parents[3]
        return load_config(root / "configs" / "base.yaml", root=root)


def applicant_frame(applicant: Applicant) -> pd.DataFrame:
    row = applicant.model_dump()
    frame = pd.DataFrame([row])
    frame = add_derived_features(frame)
    for column in NUMERIC_COLUMNS + DERIVED_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str)
    return frame[list(FEATURE_COLUMNS)]


def _predict_one(bundle, metadata: dict, applicant: Applicant) -> Prediction:
    frame = applicant_frame(applicant)
    probability = float(bundle.predict_proba_bad(frame)[0])
    threshold = float(bundle.threshold)
    cost = metadata.get("cost_matrix", {})
    cost_approve = float(cost.get("cost_bad_approved", 5)) * probability
    cost_reject = float(cost.get("cost_good_rejected", 1)) * (1.0 - probability)
    reason_codes = bundle.reason_codes(frame, top_k=3)
    return Prediction(
        probability_of_default=round(probability, 6),
        decision="approve" if probability < threshold else "reject",
        threshold=round(threshold, 6),
        expected_cost_approve=round(cost_approve, 6),
        expected_cost_reject=round(cost_reject, 6),
        reason_codes=[ReasonCode(**code) for code in reason_codes],
        model_type=bundle.model_type,
        model_version=str(metadata.get("version", "unknown")),
        calibration=str(metadata.get("calibration", "isotonic")),
    )


def create_app(cfg: Config | None = None) -> FastAPI:
    config = cfg or _resolve_config()
    store = ModelStore(config)

    app = FastAPI(
        title="credit-risk",
        version=__version__,
        description="Default risk scoring on the Statlog German Credit schema.",
    )
    app.state.config = config
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        try:
            metadata = store.metadata
            return {
                "status": "ok",
                "model_loaded": True,
                "model_type": metadata.get("model_type"),
                "model_version": metadata.get("version"),
                "threshold": metadata.get("threshold"),
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        except ModelUnavailable as exc:
            return {
                "status": "degraded",
                "model_loaded": False,
                "detail": str(exc),
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }

    @app.get("/model/info")
    def model_info() -> dict:
        try:
            metadata = store.metadata
            bundle = store.bundle
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        holdout = metadata.get("metrics", {}).get("holdout", {})
        cv_mean = metadata.get("metrics", {}).get("cv", {}).get("mean", {})
        return {
            "model_type": metadata.get("model_type"),
            "version": metadata.get("version"),
            "threshold": metadata.get("threshold"),
            "calibration": metadata.get("calibration"),
            "trained_at": metadata.get("trained_at"),
            "git_commit": metadata.get("git_commit"),
            "seed": metadata.get("seed"),
            "config_hash": metadata.get("config_hash"),
            "features": metadata.get("features"),
            "cost_matrix": metadata.get("cost_matrix"),
            "metrics": {
                "holdout": {
                    key: holdout.get(key)
                    for key in ("roc_auc", "pr_auc", "ks", "brier", "accuracy_at_threshold")
                },
                "cv_mean": {
                    key: cv_mean.get(key)
                    for key in ("roc_auc", "pr_auc", "ks", "brier", "accuracy_at_threshold")
                },
            },
            "reason_codes": {
                "method": "exact tree SHAP, LightGBM pred_contrib=True"
                if bundle.model_type == "lightgbm"
                else "logistic coefficients times encoded values",
                "top_k": 3,
                "scale": "log-odds contribution to the base model score",
                "caveat": "contributions, not causal explanations",
            },
        }

    @app.post("/predict", response_model=Prediction)
    def predict(applicant: Applicant) -> Prediction:
        try:
            bundle = store.bundle
            metadata = store.metadata
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _predict_one(bundle, metadata, applicant)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest) -> BatchResponse:
        try:
            bundle = store.bundle
            metadata = store.metadata
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        predictions = [_predict_one(bundle, metadata, row) for row in request.applicants]
        return BatchResponse(
            predictions=predictions, model_version=str(metadata.get("version", "unknown"))
        )

    return app


app = create_app()
