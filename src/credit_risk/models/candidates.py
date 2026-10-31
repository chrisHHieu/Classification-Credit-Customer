"""Model candidate factory.

Every candidate is a scikit-learn compatible pipeline so the rest of the project
(training, calibration, cross-validation, permutation importance, the registry)
does not need to know which library produced the scores. Two encodings exist:

* ``one_hot``: numeric columns scaled or passed through, categorical columns
  one-hot encoded, which is what the logistic baseline and LightGBM use;
* ``native``: the frame reaches the model with pandas ``category`` dtype, which is
  what XGBoost (``enable_categorical``) and CatBoost (``cat_features``) use.

Optional libraries are imported lazily. A missing package raises
``CandidateUnavailable`` with the install hint, and ``train`` records the model as
skipped instead of crashing the whole command.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import Config
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
)

NUMERIC_FEATURES: list[str] = list(NUMERIC_COLUMNS + DERIVED_COLUMNS)
CANDIDATES: tuple[str, ...] = ("logistic", "lightgbm", "xgboost", "catboost")
TREE_CANDIDATES: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")
NATIVE_CANDIDATES: tuple[str, ...] = ("xgboost", "catboost")

# CatBoost monotone constraints, in feature order. Duration and installment rate
# are constrained to raise risk because the direction is documented credit
# practice (longer exposure and a higher payment burden mean more risk). Age and
# credit amount are deliberately left free: the direction is not monotone in this
# data, so pinning them would encode an assumption the labels do not support.
MONOTONE_INCREASING: tuple[str, ...] = ("duration_months", "installment_rate")

CATEGORY_INDICES: list[int] = [FEATURE_COLUMNS.index(name) for name in CATEGORICAL_COLUMNS]
MISSING_TOKEN = "__missing__"


class CandidateUnavailable(RuntimeError):
    """Raised when the optional dependency for a candidate is not installed."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str  # linear | tree
    encoding: str  # one_hot | native
    requires: str
    estimator: Any = field(repr=False, compare=False)


def is_available(name: str) -> bool:
    """True when the library behind ``name`` can be imported."""
    module = {"logistic": "sklearn", "lightgbm": "lightgbm", "xgboost": "xgboost",
              "catboost": "catboost"}.get(name)
    return module is not None and importlib.util.find_spec(module) is not None


def _require(name: str, module: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise CandidateUnavailable(
            f"model '{name}' needs the optional package '{module}'; install it with "
            "python -m pip install -e '.[modern]'"
        )


class FrameCaster(BaseEstimator, TransformerMixin):
    """Pass the feature frame through, with categoricals as pandas ``category``.

    XGBoost and CatBoost read the category dtype directly, so no one-hot expansion
    happens. Categories are learned in ``fit`` from the training rows only; an
    unseen category becomes missing at prediction time instead of silently
    creating a new column.
    """

    def __init__(self, columns: list[str] | None = None, categorical: list[str] | None = None):
        self.columns = columns
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y=None) -> FrameCaster:
        columns = list(self.columns) if self.columns is not None else list(FEATURE_COLUMNS)
        categorical = (
            list(self.categorical) if self.categorical is not None else list(CATEGORICAL_COLUMNS)
        )
        self.columns_ = columns
        self.categorical_ = categorical
        self.categories_ = {
            column: sorted(X[column].astype(str).unique().tolist()) for column in categorical
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns_].copy()
        for column in self.columns_:
            if column in self.categorical_:
                values = out[column].astype(object)
                # ``pd.Categorical`` turns a value it does not know into NaN, and
                # CatBoost refuses NaN in a categorical column. Map every unseen
                # or missing value to the explicit token instead: the model can
                # then split on "unknown" as its own level.
                known = values.isin(self.categories_[column])
                values = values.where(known, MISSING_TOKEN).astype(str)
                out[column] = pd.Categorical(
                    values,
                    categories=[*self.categories_[column], MISSING_TOKEN],
                )
            else:
                out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def _params(cfg: Config, name: str, override: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(getattr(cfg.train, name))
    if override:
        base.update(override)
    return base


class CatBoostAdapter(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around CatBoost.

    CatBoost rewrites ``cat_features`` and ``monotone_constraints`` in its own
    constructor, which makes ``sklearn.clone`` refuse it. The adapter keeps the
    parameters it was given and builds a fresh CatBoost model on every fit, so the
    pipeline clones cleanly and every fold starts from an unfitted booster.
    """

    def __init__(
        self,
        params: dict | None = None,
        cat_features: list[int] | None = None,
        monotone_constraints: list[int] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.params = {} if params is None else params
        self.cat_features = cat_features
        self.monotone_constraints = monotone_constraints
        self.random_seed = random_seed

    def _build(self):
        from catboost import CatBoostClassifier

        options = dict(self.params)
        options.setdefault("allow_writing_files", False)
        return CatBoostClassifier(
            random_seed=self.random_seed,
            cat_features=self.cat_features,
            monotone_constraints=self.monotone_constraints,
            **options,
        )

    def fit(self, X, y, **kwargs):
        self.model_ = self._build()
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.model_.predict(X).astype(int).ravel()

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)


def make_estimator(name: str, cfg: Config, override: dict[str, Any] | None = None):
    """Build the raw estimator for ``name``, without preprocessing."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown model type: {name}")
    params = _params(cfg, name, override)

    if name == "logistic":
        return LogisticRegression(random_state=cfg.seed, **params)

    if name == "lightgbm":
        _require(name, "lightgbm")
        import lightgbm

        return lightgbm.LGBMClassifier(random_state=cfg.seed, **params)

    if name == "xgboost":
        _require(name, "xgboost")
        import xgboost

        params.setdefault("tree_method", "hist")
        params.setdefault("enable_categorical", True)
        return xgboost.XGBClassifier(random_state=cfg.seed, **params)

    _require(name, "catboost")
    monotone = [1 if column in MONOTONE_INCREASING else 0 for column in FEATURE_COLUMNS]
    return CatBoostAdapter(
        params=params,
        cat_features=list(CATEGORY_INDICES),
        monotone_constraints=monotone,
        random_seed=cfg.seed,
    )


def make_pipeline(name: str, cfg: Config, override: dict[str, Any] | None = None) -> Pipeline:
    """Preprocessing plus estimator, ready for ``fit`` on a feature frame."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown model type: {name}")
    model = make_estimator(name, cfg, override)

    if name in NATIVE_CANDIDATES:
        preprocess: Any = FrameCaster()
    else:
        numeric: Any = (
            Pipeline([("scale", StandardScaler())]) if name == "logistic" else "passthrough"
        )
        preprocess = ColumnTransformer(
            [
                ("num", numeric, NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                    list(CATEGORICAL_COLUMNS),
                ),
            ],
            remainder="drop",
        )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def spec(name: str, cfg: Config, override: dict[str, Any] | None = None) -> CandidateSpec:
    """Candidate with its family and encoding metadata."""
    if name in TREE_CANDIDATES:
        family = "tree"
    elif name == "logistic":
        family = "linear"
    else:
        raise ValueError(f"unknown model type: {name}")
    encoding = "native" if name in NATIVE_CANDIDATES else "one_hot"
    requires = {
        "logistic": "scikit-learn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
    }[name]
    return CandidateSpec(
        name=name,
        family=family,
        encoding=encoding,
        requires=requires,
        estimator=make_pipeline(name, cfg, override),
    )


def available_candidates(names: list[str] | tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """Split names into (usable, {unusable: reason})."""
    usable: list[str] = []
    skipped: dict[str, str] = {}
    for name in names:
        if name not in CANDIDATES:
            skipped[name] = "unknown candidate"
        elif not is_available(name):
            module = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}[name]
            skipped[name] = f"optional package '{module}' is not installed"
        else:
            usable.append(name)
    return usable, skipped
