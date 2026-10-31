"""MLflow tracking with soft imports and an explicit fallback.

Run ``train --track`` to log one MLflow run per candidate. The tracking URI comes
from ``MLFLOW_TRACKING_URI`` and defaults to ``sqlite:///mlflow.db``: MLflow 3 put
the filesystem store in maintenance mode and refuses to open it, so a SQLite file
is the smallest store that is both writable and able to host a model registry.

The file store cannot host a model registry, so registration is attempted and, if
it is refused, the run still logs the estimator and records the reason in
``reports/train_summary.json`` instead of failing. A ``file:`` URI is still
accepted: the code opts that process back into the deprecated backend so an older
experiment keeps working, and the refusal is reported instead of raised.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from credit_risk.config import Config

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "credit-risk"


def available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _prepare_store(uri: str) -> None:
    """Opt a ``file:`` store back in so the deprecated backend still opens.

    MLflow 3 raises on the filesystem backend unless this variable is set. The
    store cannot host a registry, which the caller reports as a fallback, but the
    run itself is still worth logging.
    """
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _registered_version(info: object) -> str | None:
    """Read the version whether MLflow returns a number or a ModelVersion."""
    raw = getattr(info, "registered_model_version", None)
    raw = getattr(raw, "version", raw)
    return str(raw) if raw is not None else None


def registered_name(model_type: str) -> str:
    return f"credit_risk_{model_type}"


def _log_model(mlflow, estimator, model_type: str, record: dict) -> None:
    """Log the fitted estimator, keeping CatBoost off the sklearn flavor.

    MLflow's sklearn flavor serializes through skops/cloudpickle. The CatBoost
    adapter wraps a native booster, which that path does not round-trip cleanly,
    so the pipeline is written with joblib and logged as a file instead.
    """
    if type(estimator).__name__ == "CatBoostAdapter":
        import tempfile

        import joblib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catboost_pipeline.joblib"
            joblib.dump(estimator, path)
            mlflow.log_artifact(str(path))
        record["serialization_format"] = "joblib"
        record["fallback"] = (
            "catboost is logged as a joblib artifact: the sklearn flavor serializes "
            "the native booster through skops/cloudpickle, which does not round-trip"
        )
        return

    try:
        info = mlflow.sklearn.log_model(
            estimator,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_name(model_type),
        )
        record["serialization_format"] = "cloudpickle"
        record["registered_model"] = registered_name(model_type)
        record["model_uri"] = info.model_uri
        record["version"] = _registered_version(info)
    except Exception as exc:
        record["fallback"] = (
            f"model registration refused by the tracking store "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]}); "
            "logged the run and artifacts without a registry entry"
        )
        try:
            info = mlflow.sklearn.log_model(
                estimator,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            record["model_uri"] = info.model_uri
        except Exception as inner:  # pragma: no cover - both paths broken
            record["fallback"] += f"; logging the model also failed ({type(inner).__name__})"


def log_training_run(
    cfg: Config,
    *,
    model_type: str,
    params: dict,
    metrics: dict,
    estimator,
    artifacts: list[Path] | None = None,
    tags: dict | None = None,
) -> dict:
    """Log one candidate. Returns a record describing what actually happened."""
    record: dict = {
        "model_type": model_type,
        "tracking_uri": tracking_uri(),
        "experiment": EXPERIMENT,
        "run_id": None,
        "registered_model": None,
        "version": None,
        "serialization_format": None,
        "fallback": None,
    }
    if not available():
        record["fallback"] = "mlflow is not installed"
        return record

    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - broken install
        record["fallback"] = f"mlflow import failed: {type(exc).__name__}: {exc}"
        return record

    try:
        _prepare_store(tracking_uri())
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=model_type) as run:
            record["run_id"] = run.info.run_id
            mlflow.log_params({key: value for key, value in params.items() if value is not None})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            if tags:
                mlflow.set_tags(tags)
            for path in artifacts or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            _log_model(mlflow, estimator, model_type, record)
    except Exception as exc:
        record["fallback"] = f"tracking failed: {type(exc).__name__}: {exc}"
    return record


def set_champion(model_type: str, version: str | None) -> dict:
    """Point the ``champion`` alias at the cost-winning model version."""
    record = {
        "model": registered_name(model_type),
        "version": version,
        "alias": None,
        "error": None,
    }
    if not available() or version is None:
        record["error"] = "no registered version to alias"
        return record
    try:
        from mlflow import MlflowClient

        _prepare_store(tracking_uri())
        client = MlflowClient(tracking_uri=tracking_uri())
        client.set_registered_model_alias(registered_name(model_type), "champion", str(version))
        record["alias"] = "champion"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return record


def write_records(path: Path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
