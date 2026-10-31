"""Statlog German Credit schema, code labels and validation ranges.

The raw file from UCI has no header and uses the Statlog attribute codes (A11,
A34, ...). This module is the single place where those codes become readable
names, so the fetcher, validator, feature builder, model and API all agree.
"""

from __future__ import annotations

from typing import Final

# Column order of the raw space separated file (20 attributes plus target).
COLUMNS: Final[tuple[str, ...]] = (
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "target",
)

TARGET: Final[str] = "target"
LABEL_COLUMN: Final[str] = "default"

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "checking_status",
    "credit_history",
    "purpose",
    "savings",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
)

DERIVED_COLUMNS: Final[tuple[str, ...]] = ("credit_amount_per_month",)

FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + DERIVED_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_LABELS: Final[dict[str, str]] = {
    "checking_status": "checking account status",
    "duration_months": "loan duration in months",
    "credit_history": "credit history",
    "purpose": "loan purpose",
    "credit_amount": "credit amount",
    "savings": "savings balance",
    "employment_since": "time in current employment",
    "installment_rate": "installment rate as percent of income",
    "personal_status_sex": "personal status and sex",
    "other_debtors": "other debtors",
    "residence_since": "years at current residence",
    "property": "property owned",
    "age_years": "age in years",
    "other_installment_plans": "other installment plans",
    "housing": "housing",
    "existing_credits": "number of existing credits",
    "job": "job type",
    "people_liable": "people liable for maintenance",
    "telephone": "telephone registered",
    "foreign_worker": "foreign worker flag",
    "credit_amount_per_month": "credit amount per month of term",
}

CODE_LABELS: Final[dict[str, str]] = {
    "A11": "overdrawn account",
    "A12": "0 to 200 DM",
    "A13": "200 DM or more",
    "A14": "no checking account",
    "A30": "no credits or all paid back",
    "A31": "all credits at this bank paid back",
    "A32": "existing credits paid back so far",
    "A33": "past delay in payments",
    "A34": "critical account",
    "A40": "new car",
    "A41": "used car",
    "A42": "furniture or equipment",
    "A43": "radio or television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "other purpose",
    "A61": "savings below 100 DM",
    "A62": "savings 100 to 500 DM",
    "A63": "savings 500 to 1000 DM",
    "A64": "savings 1000 DM or more",
    "A65": "unknown or no savings",
    "A71": "unemployed",
    "A72": "employed under 1 year",
    "A73": "employed 1 to 4 years",
    "A74": "employed 4 to 7 years",
    "A75": "employed 7 years or more",
    "A91": "male, divorced or separated",
    "A92": "female, divorced, separated or married",
    "A93": "male, single",
    "A94": "male, married or widowed",
    "A95": "female, single",
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
    "A121": "real estate",
    "A122": "building society savings or life insurance",
    "A123": "car or other property",
    "A124": "unknown or no property",
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
    "A151": "rent",
    "A152": "own",
    "A153": "free of charge",
    "A171": "unemployed or unskilled non-resident",
    "A172": "unskilled resident",
    "A173": "skilled employee or official",
    "A174": "management, self-employed or highly qualified",
    "A191": "no telephone",
    "A192": "telephone registered",
    "A201": "foreign worker",
    "A202": "not a foreign worker",
}

VALID_CODES: Final[dict[str, tuple[str, ...]]] = {
    "checking_status": ("A11", "A12", "A13", "A14"),
    "credit_history": ("A30", "A31", "A32", "A33", "A34"),
    "purpose": (
        "A40",
        "A41",
        "A42",
        "A43",
        "A44",
        "A45",
        "A46",
        "A47",
        "A48",
        "A49",
        "A410",
    ),
    "savings": ("A61", "A62", "A63", "A64", "A65"),
    "employment_since": ("A71", "A72", "A73", "A74", "A75"),
    "personal_status_sex": ("A91", "A92", "A93", "A94", "A95"),
    "other_debtors": ("A101", "A102", "A103"),
    "property": ("A121", "A122", "A123", "A124"),
    "other_installment_plans": ("A141", "A142", "A143"),
    "housing": ("A151", "A152", "A153"),
    "job": ("A171", "A172", "A173", "A174"),
    "telephone": ("A191", "A192"),
    "foreign_worker": ("A201", "A202"),
}

# Inclusive bounds observed in the published dataset. Validation allows a little
# slack around them so a refreshed copy of the file does not fail on noise, but
# anything far outside the documented range is a contract violation.
NUMERIC_RANGES: Final[dict[str, tuple[float, float]]] = {
    "duration_months": (4, 72),
    "credit_amount": (250, 20000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "age_years": (18, 80),
    "existing_credits": (1, 4),
    "people_liable": (1, 2),
}

TARGET_VALUES: Final[tuple[int, int]] = (1, 2)  # 1 = good, 2 = bad


def feature_name_parts(raw_name: str) -> tuple[str, str | None]:
    """Split a transformer output name into (column, category code or None).

    ``cat__purpose_A43`` becomes ``("purpose", "A43")`` and
    ``num__duration_months`` becomes ``("duration_months", None)``.
    """
    name = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    for column in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
        prefix = f"{column}_"
        if name.startswith(prefix):
            return column, name[len(prefix) :]
    return name, None


def describe_feature(raw_name: str) -> str:
    """Human readable label for a transformer output, used in reason codes."""
    column, code = feature_name_parts(raw_name)
    label = FEATURE_LABELS.get(column, column.replace("_", " "))
    if code is None:
        return label
    meaning = CODE_LABELS.get(code)
    return f"{label} = {code} ({meaning})" if meaning else f"{label} = {code}"


def selected_columns(frame_columns: list[str]) -> list[str]:
    """Feature columns present in a frame, in the canonical order."""
    available = set(frame_columns)
    return [column for column in FEATURE_COLUMNS if column in available]
