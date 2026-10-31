"""Typed data contracts with pandera.

``validate.py`` keeps the business checks (row count, class balance, checksum);
pandera adds declarative dtypes, value domains and column sets that are enforced
row by row, with ``lazy=True`` so one bad row does not hide the rest. Both layers
run and their results are merged into the same report.

The module imports pandera lazily: on a core install the contract is reported as
skipped, and every other check still runs.
"""

from __future__ import annotations

import importlib.util

import pandas as pd

from credit_risk.data.schema import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
    TARGET,
    TARGET_VALUES,
    VALID_CODES,
)

PROCESSED_COLUMNS: list[str] = (
    ["row_id"] + list(FEATURE_COLUMNS) + [LABEL_COLUMN, TARGET, "fold", "is_heldout"]
)


def available() -> bool:
    return importlib.util.find_spec("pandera") is not None


def _pandera():
    import pandera.pandas as pa

    return pa


def raw_schema(schema=None):
    """Contract for the parsed raw payload; follows the patchable validator schema."""
    pa = _pandera()
    columns = list(schema.columns) if schema is not None else list(COLUMNS)
    categorical = dict(schema.categorical) if schema is not None else dict(VALID_CODES)
    numeric = dict(schema.numeric) if schema is not None else dict(NUMERIC_RANGES)
    target_values = schema.target_values if schema is not None else TARGET_VALUES

    fields = {}
    for column in columns:
        if column in categorical:
            fields[column] = pa.Column(
                str, pa.Check.isin(list(categorical[column])), nullable=False
            )
        elif column in numeric:
            low, high = numeric[column]
            fields[column] = pa.Column(
                float, pa.Check.in_range(float(low), float(high)), nullable=False
            )
        else:
            fields[column] = pa.Column(nullable=False)
    fields[TARGET] = pa.Column(int, pa.Check.isin(list(target_values)), nullable=False)
    return pa.DataFrameSchema(fields, strict=True, ordered=True, coerce=True)


def processed_schema():
    """Contract for ``data/processed``: dtypes, domains and split consistency."""
    pa = _pandera()
    fields = {"row_id": pa.Column(int, pa.Check.ge(0), nullable=False, unique=True)}
    for column in NUMERIC_COLUMNS:
        low, high = NUMERIC_RANGES[column]
        fields[column] = pa.Column(float, pa.Check.in_range(float(low), float(high)))
    fields["credit_amount_per_month"] = pa.Column(float, pa.Check.ge(0.0))
    for column in CATEGORICAL_COLUMNS:
        fields[column] = pa.Column(str, pa.Check.isin(list(VALID_CODES[column])))
    fields[LABEL_COLUMN] = pa.Column(int, pa.Check.isin([0, 1]))
    fields[TARGET] = pa.Column(int, pa.Check.isin(list(TARGET_VALUES)))
    fields["fold"] = pa.Column(int, pa.Check.ge(-1))
    fields["is_heldout"] = pa.Column(int, pa.Check.isin([0, 1]))

    split_consistent = pa.Check(
        lambda frame: bool(
            ((frame["is_heldout"] == 1) == (frame["fold"] == -1)).all()
            and not ((frame["is_heldout"] == 0) & (frame["fold"] < 0)).any()
        ),
        error="held-out rows must have fold -1 and development rows a fold >= 0",
    )
    return pa.DataFrameSchema(
        fields,
        checks=split_consistent,
        strict=True,
        ordered=False,
        coerce=True,
    )


def _summarise(exc) -> dict:
    cases = exc.failure_cases
    samples = []
    for _, row in cases.head(5).iterrows():
        samples.append(
            f"{row.get('column', '?')} {row.get('check', '?')}: {row.get('failure_case', '?')}"
        )
    return {
        "available": True,
        "ok": False,
        "error_count": int(len(cases)),
        "samples": samples,
        "skipped": None,
    }


def check(frame: pd.DataFrame, kind: str = "raw", schema=None) -> dict:
    """Validate ``frame`` against the named contract. Never raises for bad data."""
    if not available():
        return {
            "available": False,
            "ok": True,
            "error_count": 0,
            "samples": [],
            "skipped": "pandera is not installed; contract not enforced",
        }
    pa = _pandera()
    contract = raw_schema(schema) if kind == "raw" else processed_schema()
    try:
        contract.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return _summarise(exc)
    except pa.errors.SchemaError as exc:  # pragma: no cover - non lazy path
        return {
            "available": True,
            "ok": False,
            "error_count": 1,
            "samples": [str(exc).splitlines()[0]],
            "skipped": None,
        }
    return {"available": True, "ok": True, "error_count": 0, "samples": [], "skipped": None}


def format_check(result: dict) -> str:
    if result.get("skipped"):
        return str(result["skipped"])
    if result["ok"]:
        return f"pandera contract satisfied ({result['error_count']} violations)"
    return (
        f"pandera contract failed with {result['error_count']} violations; "
        + "; ".join(result["samples"])
    )
