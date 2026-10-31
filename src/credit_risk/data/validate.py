"""Data contract for the raw Statlog file.

``validate`` is deliberately concrete: exact column order, exact row count, exact
class balance, category code domains, numeric ranges and an integral check that
the monetary fields still look like the published dataset. It returns a report
object and the CLI turns a failed report into a non-zero exit code.

The schema is a module level object so tests can monkey-patch it and check that
the contract actually follows the patch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from credit_risk.config import Config
from credit_risk.data import contracts
from credit_risk.data.fetch import FetchError, load_raw, read_raw, sha256_file
from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
    TARGET,
    TARGET_VALUES,
    VALID_CODES,
)


@dataclass(frozen=True)
class Schema:
    """Structure the validator enforces. Patchable for tests."""

    columns: tuple[str, ...]
    categorical: dict[str, tuple[str, ...]]
    numeric: dict[str, tuple[float, float]]
    target_values: tuple[int, ...]
    integral_columns: tuple[str, ...] = (
        "duration_months",
        "installment_rate",
        "residence_since",
        "age_years",
        "existing_credits",
        "people_liable",
    )


SCHEMA = Schema(
    columns=COLUMNS,
    categorical=dict(VALID_CODES),
    numeric=dict(NUMERIC_RANGES),
    target_values=TARGET_VALUES,
)


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, "pass" if ok else "fail", detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "warn", detail))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }

    def summary(self) -> str:
        lines = [
            f"[{check.status:4}] {check.name}: {check.detail}" for check in self.checks
        ]
        lines.append("RESULT: pass" if self.ok else "RESULT: fail")
        return "\n".join(lines)


def _check_columns(report: ValidationReport, frame: pd.DataFrame, schema: Schema) -> None:
    actual = list(frame.columns)
    expected = list(schema.columns)
    report.add(
        "columns",
        actual == expected,
        f"{len(actual)} columns, order matches schema: {actual == expected}",
    )


def _check_missing(report: ValidationReport, frame: pd.DataFrame) -> None:
    missing = int(frame.isna().sum().sum())
    per_column = frame.isna().sum()
    offenders = [column for column in frame.columns if per_column[column] > 0]
    report.add(
        "missing_values",
        missing == 0,
        f"{missing} missing cells" + (f" in {offenders}" if offenders else ""),
    )


def _check_target(
    report: ValidationReport, frame: pd.DataFrame, cfg: Config, schema: Schema
) -> None:
    if TARGET not in frame.columns:
        report.add("target_domain", False, "target column is missing")
        return
    values = frame[TARGET].dropna().astype(int)
    domain = sorted(values.unique().tolist())
    known = domain and all(value in schema.target_values for value in domain)
    report.add(
        "target_domain",
        bool(known),
        f"values {domain}, expected subset {schema.target_values}",
    )
    bad = int((values == 2).sum())
    good = int((values == 1).sum())
    report.add(
        "class_balance",
        bad == cfg.data.expected_bad_count,
        f"{good} good / {bad} bad, expected {cfg.data.expected_rows - cfg.data.expected_bad_count}"
        f" good / {cfg.data.expected_bad_count} bad",
    )


def _check_categories(
    report: ValidationReport, frame: pd.DataFrame, schema: Schema
) -> None:
    unknown: dict[str, list[str]] = {}
    for column, allowed in schema.categorical.items():
        if column not in frame.columns:
            unknown[column] = ["column missing"]
            continue
        seen = set(frame[column].dropna().astype(str).unique())
        extra = sorted(seen - set(allowed))
        if extra:
            unknown[column] = extra
    report.add(
        "categorical_domain",
        not unknown,
        "all codes known" if not unknown else f"unknown codes: {unknown}",
    )


def _check_numeric(report: ValidationReport, frame: pd.DataFrame, schema: Schema) -> None:
    problems: list[str] = []
    for column, (low, high) in schema.numeric.items():
        if column not in frame.columns:
            problems.append(f"{column}: missing column")
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.isna().any():
            problems.append(f"{column}: non numeric values")
            continue
        out_of_range = (series < low) | (series > high)
        if out_of_range.any():
            problems.append(
                f"{column}: {int(out_of_range.sum())} values outside [{low}, {high}]"
            )
    report.add(
        "numeric_range",
        not problems,
        "all numeric columns inside documented bounds" if not problems else "; ".join(problems),
    )


def _check_integral(report: ValidationReport, frame: pd.DataFrame, schema: Schema) -> None:
    problems: list[str] = []
    for column in schema.integral_columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) and (series % 1 != 0).any():
            problems.append(column)
    report.add(
        "integral_columns",
        not problems,
        "integer valued" if not problems else f"non integer values in {problems}",
    )


def _check_duplicates(report: ValidationReport, frame: pd.DataFrame, cfg: Config) -> None:
    duplicates = int(frame.duplicated().sum())
    expected_rows = cfg.data.expected_rows
    if duplicates == 0:
        report.add("duplicate_rows", True, "no exact duplicate rows")
        return
    report.warn(
        "duplicate_rows",
        f"{duplicates} exact duplicate rows ({duplicates / max(expected_rows, 1):.1%} of the file),"
        " kept because the published file has no record identifier to deduplicate on",
    )


def _check_contract(report: ValidationReport, frame: pd.DataFrame, schema: Schema) -> None:
    """Declarative dtype and domain contract, merged with the business checks."""
    result = contracts.check(frame, "raw", schema=schema)
    detail = contracts.format_check(result)
    if not result["available"]:
        report.warn("pandera_contract", detail)
    else:
        report.add("pandera_contract", bool(result["ok"]), detail)


def validate_frame(
    frame: pd.DataFrame,
    cfg: Config,
    schema: Schema | None = None,
) -> ValidationReport:
    """Run every structural check against an in-memory frame."""
    active = schema if schema is not None else SCHEMA
    report = ValidationReport()
    report.add(
        "row_count",
        len(frame) == cfg.data.expected_rows,
        f"{len(frame)} rows, expected {cfg.data.expected_rows}",
    )
    _check_columns(report, frame, active)
    _check_missing(report, frame)
    _check_target(report, frame, cfg, active)
    _check_categories(report, frame, active)
    _check_numeric(report, frame, active)
    _check_integral(report, frame, active)
    _check_duplicates(report, frame, cfg)
    _check_contract(report, frame, active)
    return report


def validate_file(cfg: Config, path: str | Path | None = None, schema: Schema | None = None):
    """Read the raw file, check its checksum against the manifest, then its shape."""
    raw_path = Path(path) if path is not None else cfg.raw_file
    report = ValidationReport()
    if not raw_path.exists():
        report.add("file_present", False, f"{raw_path} does not exist, run fetch first")
        return report
    report.add("file_present", True, f"{raw_path} ({raw_path.stat().st_size} bytes)")

    if cfg.manifest_file.exists() and raw_path == cfg.raw_file:

        manifest = json.loads(cfg.manifest_file.read_text(encoding="utf-8"))
        digest_match = manifest.get("sha256") == sha256_file(raw_path)
        report.add(
            "sha256",
            digest_match,
            f"file {sha256_file(raw_path)[:16]}... manifest {str(manifest.get('sha256'))[:16]}...",
        )
    else:
        report.warn("sha256", "no manifest for this file, checksum not pinned")

    try:
        frame = read_raw(raw_path)
    except (FetchError, ValueError, pd.errors.ParserError) as exc:
        report.add("parse", False, str(exc))
        return report
    report.add("parse", True, f"parsed {frame.shape[0]} rows x {frame.shape[1]} columns")

    structural = validate_frame(frame, cfg, schema)
    report.checks.extend(structural.checks)
    return report


def load_validated(cfg: Config) -> pd.DataFrame:
    """Convenience for downstream steps: read the file and refuse to continue if bad."""
    report = validate_file(cfg)
    if not report.ok:
        raise ValueError("raw data failed validation:\n" + report.summary())
    return load_raw(cfg)


__all__ = [
    "CATEGORICAL_COLUMNS",
    "NUMERIC_COLUMNS",
    "SCHEMA",
    "Schema",
    "ValidationReport",
    "load_validated",
    "validate_file",
    "validate_frame",
]
