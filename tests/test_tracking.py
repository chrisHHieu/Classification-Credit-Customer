from __future__ import annotations

import pytest

from credit_risk import tracking
from credit_risk.models import registry

pytest.importorskip("mlflow")


@pytest.mark.modern
def test_mlflow_run_is_logged_and_registered(project, tmp_path, monkeypatch) -> None:
    database = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    bundle, metadata = registry.load_bundle(project.cfg, "logistic")
    record = tracking.log_training_run(
        project.cfg,
        model_type="logistic",
        params={"model": "logistic", "seed": 7},
        metrics={"holdout_roc_auc": 0.7, "holdout_brier": 0.18},
        estimator=bundle.estimator,
        artifacts=[],
        tags={"pipeline": "credit-risk"},
    )
    assert record["run_id"], record["fallback"]
    assert record["fallback"] is None
    assert record["registered_model"] == "credit_risk_logistic"
    assert record["version"] is not None


@pytest.mark.modern
def test_champion_alias_points_at_the_version(project, tmp_path, monkeypatch) -> None:
    database = tmp_path / "mlflow_alias.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    bundle, _ = registry.load_bundle(project.cfg, "logistic")
    record = tracking.log_training_run(
        project.cfg,
        model_type="logistic",
        params={"model": "logistic"},
        metrics={"holdout_roc_auc": 0.7},
        estimator=bundle.estimator,
    )
    alias = tracking.set_champion("logistic", record["version"])
    assert alias["error"] is None
    assert alias["alias"] == "champion"

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking.tracking_uri())
    resolved = client.get_model_version_by_alias("credit_risk_logistic", "champion")
    assert str(resolved.version) == record["version"]


@pytest.mark.modern
def test_file_store_still_logs_a_run(project, tmp_path, monkeypatch) -> None:
    """MLflow 3 puts the filesystem store in maintenance mode; the code opts in.

    Registration on that backend is no longer guaranteed, so the contract worth
    testing is the graceful one: the run is logged and nothing raises.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{(tmp_path / 'mlruns').as_posix()}")
    bundle, _ = registry.load_bundle(project.cfg, "logistic")
    record = tracking.log_training_run(
        project.cfg,
        model_type="logistic",
        params={"model": "logistic"},
        metrics={"holdout_roc_auc": 0.7},
        estimator=bundle.estimator,
    )
    assert record["run_id"], "the run itself must still be logged"
    assert record["fallback"] is None or record["registered_model"]


@pytest.mark.modern
def test_unreachable_store_is_recorded_not_raised(project, tmp_path, monkeypatch) -> None:
    """A tracking outage must never cost a finished training run."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///Z:/no/such/directory/mlflow.db")
    bundle, _ = registry.load_bundle(project.cfg, "logistic")
    record = tracking.log_training_run(
        project.cfg,
        model_type="logistic",
        params={"model": "logistic"},
        metrics={"holdout_roc_auc": 0.7},
        estimator=bundle.estimator,
    )
    assert record["run_id"] is None
    assert record["fallback"], "the failure must be reported in the record"
    assert record["registered_model"] is None
