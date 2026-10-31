from __future__ import annotations

import json

import pytest

from credit_risk.data.schema import LABEL_COLUMN
from credit_risk.monitor import evidently_report

pytest.importorskip("evidently")


@pytest.mark.modern
def test_evidently_report_writes_html_and_json(project) -> None:
    payload = evidently_report.run_evidently(project.cfg)
    assert payload["feature_drift"]["total_columns"] == 21
    assert 0 <= payload["feature_drift"]["drift_share"] <= 1
    assert payload["target_drift"]["column"] == LABEL_COLUMN
    assert "p_value" in payload["target_drift"]

    html = project.root / payload["html"]
    assert html.exists() and html.stat().st_size > 0
    stored = json.loads(project.cfg.evidently_json.read_text(encoding="utf-8"))
    assert stored["feature_drift"]["total_columns"] == 21


@pytest.mark.modern
def test_evidently_accepts_an_external_csv(project, tmp_path) -> None:
    current = project.processed[project.processed["is_heldout"] == 1].head(20)
    path = tmp_path / "current.csv"
    current.to_csv(path, index=False)
    payload = evidently_report.run_evidently(project.cfg, input_path=path)
    assert payload["current"]["rows"] == 20
    assert payload["current"]["source"].endswith("current.csv")
