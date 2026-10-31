from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from tests.conftest import config_matching, write_raw_like

from credit_risk.data import validate as validate_module
from credit_risk.data.fetch import read_raw, sha256_file
from credit_risk.data.schema import TARGET
from credit_risk.data.validate import Schema, validate_file, validate_frame


def test_valid_frame_passes(raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    report = validate_frame(raw_frame, cfg)
    assert report.ok, report.summary()
    names = {check.name for check in report.checks}
    assert {
        "row_count",
        "columns",
        "missing_values",
        "target_domain",
        "class_balance",
        "categorical_domain",
        "numeric_range",
        "integral_columns",
    } <= names


def test_row_count_mismatch_fails(raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    report = validate_frame(raw_frame.iloc[:-5], cfg)
    assert not report.ok
    assert any(check.name == "row_count" for check in report.failures)


def test_unknown_category_fails(raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    broken = raw_frame.copy()
    broken.loc[0, "purpose"] = "A999"
    report = validate_frame(broken, cfg)
    assert not report.ok
    assert "categorical_domain" in {check.name for check in report.failures}


def test_class_balance_fails(raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    shifted = replace(
        cfg,
        data=replace(cfg.data, expected_bad_count=cfg.data.expected_bad_count + 5),
    )
    report = validate_frame(raw_frame, shifted)
    assert not report.ok
    assert "class_balance" in {check.name for check in report.failures}


def test_missing_values_fail(raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    broken = raw_frame.copy()
    broken.loc[3, "credit_amount"] = None
    report = validate_frame(broken, cfg)
    assert "missing_values" in {check.name for check in report.failures}


def test_schema_is_monkey_patchable(raw_frame: pd.DataFrame, config, monkeypatch) -> None:
    cfg = config_matching(raw_frame, config)
    assert validate_frame(raw_frame, cfg).ok

    narrow = Schema(
        columns=validate_module.SCHEMA.columns,
        categorical=validate_module.SCHEMA.categorical,
        numeric={**validate_module.SCHEMA.numeric, "duration_months": (4, 6)},
        target_values=validate_module.SCHEMA.target_values,
    )
    monkeypatch.setattr(validate_module, "SCHEMA", narrow)
    report = validate_frame(raw_frame, cfg)
    assert not report.ok
    detail = {check.name: check.detail for check in report.failures}
    assert "duration_months" in detail["numeric_range"]


def test_validate_file_checks_manifest(tmp_path, raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    write_raw_like(raw_frame, cfg.raw_file)
    cfg.manifest_file.write_text(
        json.dumps({"sha256": sha256_file(cfg.raw_file)}), encoding="utf-8"
    )

    report = validate_file(cfg)
    assert report.ok, report.summary()
    assert {"sha256", "parse"} <= {check.name for check in report.checks}

    with cfg.raw_file.open("a", encoding="utf-8") as handle:
        handle.write("A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 "
                     "A192 A201 1\n")
    tampered = validate_file(cfg)
    assert not tampered.ok
    assert {"sha256", "row_count"} & {check.name for check in tampered.failures}


def test_read_raw_gives_readable_columns(tmp_path, raw_frame: pd.DataFrame, config) -> None:
    cfg = config_matching(raw_frame, config)
    write_raw_like(raw_frame, cfg.raw_file)
    frame = read_raw(cfg.raw_file)
    assert list(frame.columns)[:3] == ["checking_status", "duration_months", "credit_history"]
    assert frame["duration_months"].dtype == "float64"
    assert int(frame[TARGET].iloc[0]) == int(raw_frame[TARGET].iloc[0])
    assert len(frame) == len(raw_frame)


def test_missing_file_is_reported(tmp_path, config) -> None:
    report = validate_file(config)
    assert not report.ok
    assert report.failures[0].name == "file_present"


@pytest.mark.parametrize("column", ["duration_months", "age_years"])
def test_non_integer_values_fail(raw_frame: pd.DataFrame, config, column: str) -> None:
    cfg = config_matching(raw_frame, config)
    broken = raw_frame.copy()
    broken[column] = broken[column].astype(float)
    broken.loc[0, column] = 20.5
    report = validate_frame(broken, cfg)
    assert "integral_columns" in {check.name for check in report.failures}
