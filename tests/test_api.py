from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import build_test_config, make_raw_frame

from credit_risk.data.schema import COLUMNS
from credit_risk.serving.app import create_app

INTEGER_FIELDS = {
    "duration_months",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
}


def applicant_payload(frame, index: int = 0) -> dict:
    row = frame.iloc[index]
    payload: dict[str, object] = {}
    for column in COLUMNS[:-1]:
        if column in INTEGER_FIELDS:
            payload[column] = int(row[column])
        elif column == "credit_amount":
            payload[column] = float(row[column])
        else:
            payload[column] = str(row[column])
    return payload


@pytest.fixture(scope="module")
def client(project) -> TestClient:
    with TestClient(create_app(project.cfg)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "logistic"


def test_model_info_exposes_threshold_and_metrics(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 < body["threshold"] < 1.0
    assert body["calibration"] == "isotonic"
    assert len(body["features"]) == 21
    assert 0.0 <= body["metrics"]["holdout"]["roc_auc"] <= 1.0
    assert body["reason_codes"]["top_k"] == 3
    assert "causal" in body["reason_codes"]["caveat"]


def test_predict_returns_decision_and_reason_codes(client: TestClient, project) -> None:
    response = client.post("/predict", json=applicant_payload(project.processed, 1))
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in {"approve", "reject"}
    expected = "approve" if body["probability_of_default"] < body["threshold"] else "reject"
    assert body["decision"] == expected
    assert len(body["reason_codes"]) == 3
    for code in body["reason_codes"]:
        assert {"feature", "label", "contribution_log_odds", "direction"} <= set(code)
    assert body["expected_cost_approve"] == pytest.approx(
        body["probability_of_default"] * 5, abs=1e-5
    )
    assert body["model_version"] == project.summary["version"]


def test_batch_predict(client: TestClient, project) -> None:
    applicants = [applicant_payload(project.processed, index) for index in range(3)]
    response = client.post("/predict/batch", json={"applicants": applicants})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3
    assert body["model_version"] == project.summary["version"]


def test_predict_returns_conformal_set_and_top_reasons(client: TestClient, project) -> None:
    response = client.post("/predict", json=applicant_payload(project.processed, 2))
    assert response.status_code == 200
    body = response.json()
    assert body["conformal_set"] in {"approve", "review", "reject"}
    assert body["top_reasons"] == body["reason_codes"]
    assert len(body["top_reasons"]) == 3


def test_model_info_alias_and_conformal_block(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["conformal"]["alpha"] == 0.1
    assert 0.0 <= body["conformal"]["quantile"] <= 1.0
    assert "ece" in body["metrics"]["holdout"]


def test_explain_endpoint_returns_five_reasons(client: TestClient, project) -> None:
    response = client.post("/explain", json=applicant_payload(project.processed, 3))
    assert response.status_code == 200
    body = response.json()
    assert body["top_k"] == 5
    assert len(body["reasons"]) == 5
    assert body["method"]
    magnitudes = [abs(reason["contribution_log_odds"]) for reason in body["reasons"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_unknown_category_code_is_rejected(client: TestClient, project) -> None:
    payload = applicant_payload(project.processed, 0)
    payload["purpose"] = "A999"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unknown_field_is_rejected(client: TestClient, project) -> None:
    payload = applicant_payload(project.processed, 0)
    payload["secret_income"] = 100
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_field_is_rejected(client: TestClient, project) -> None:
    payload = applicant_payload(project.processed, 0)
    payload.pop("credit_amount")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_registry_reports_degraded_health(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with TestClient(create_app(cfg)) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert test_client.get("/model/info").status_code == 503
        assert test_client.get("/model-info").status_code == 503
        payload = applicant_payload(make_raw_frame(rows=4), 0)
        assert test_client.post("/predict", json=payload).status_code == 503
        assert test_client.post("/explain", json=payload).status_code == 503

