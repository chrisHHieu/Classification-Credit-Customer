from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.data import contracts

pandera = pytest.importorskip("pandera")


def test_raw_contract_accepts_the_published_schema(raw_frame: pd.DataFrame) -> None:
    result = contracts.check(raw_frame, "raw")
    assert result["available"]
    assert result["ok"], result["samples"]


def test_raw_contract_flags_unknown_category(raw_frame: pd.DataFrame) -> None:
    broken = raw_frame.copy()
    broken.loc[0, "purpose"] = "A999"
    result = contracts.check(broken, "raw")
    assert not result["ok"]
    assert result["error_count"] >= 1
    assert any("purpose" in sample for sample in result["samples"])


def test_processed_contract_accepts_built_frame(processed_frame: pd.DataFrame) -> None:
    result = contracts.check(processed_frame, "processed")
    assert result["ok"], result["samples"]


def test_processed_contract_flags_split_inconsistency(processed_frame: pd.DataFrame) -> None:
    broken = processed_frame.copy()
    holdout = broken["is_heldout"] == 1
    broken.loc[holdout, "fold"] = 0
    result = contracts.check(broken, "processed")
    assert not result["ok"]
    assert any("held-out rows must have fold -1" in sample for sample in result["samples"])


def test_processed_contract_rejects_unexpected_columns(processed_frame: pd.DataFrame) -> None:
    leaky = processed_frame.copy()
    leaky["leaked_target"] = leaky["default"]
    result = contracts.check(leaky, "processed")
    assert not result["ok"]


def test_contract_is_skipped_without_pandera(monkeypatch) -> None:
    monkeypatch.setattr(contracts, "available", lambda: False)
    monkeypatch.setattr(contracts.importlib.util, "find_spec", lambda name: None)
    result = contracts.check(pd.DataFrame({"row_id": [0]}), "raw")
    assert result["skipped"]
    assert result["ok"]
