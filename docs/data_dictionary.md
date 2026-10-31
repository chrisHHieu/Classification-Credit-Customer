# Data dictionary

Source: UCI Statlog German Credit, `german.data`, 1000 rows, space separated, no header. Column order below is the file order and the order the schema enforces.

## Target

| Column | Type | Values | Meaning |
|---|---|---|---|
| `target` | integer | 1, 2 | 1 = good credit risk, 2 = bad credit risk |

The modelling label is `default`, which is 1 when `target` is 2. The published file has 700 good and 300 bad rows.

## Attributes

| Column | Type | Range or codes | Meaning |
|---|---|---|---|
| `checking_status` | category | A11, A12, A13, A14 | status of the existing checking account |
| `duration_months` | integer | 4 to 72 | loan duration in months |
| `credit_history` | category | A30 to A34 | past credit behaviour |
| `purpose` | category | A40 to A49, A410 | purpose of the loan |
| `credit_amount` | numeric | 250 to 18424 | loan amount in DM |
| `savings` | category | A61 to A65 | savings balance |
| `employment_since` | category | A71 to A75 | time in current employment |
| `installment_rate` | integer | 1 to 4 | installment rate as percent of disposable income |
| `personal_status_sex` | category | A91 to A95 | personal status and sex |
| `other_debtors` | category | A101 to A103 | other debtors or guarantors |
| `residence_since` | integer | 1 to 4 | years at current residence |
| `property` | category | A121 to A124 | property owned |
| `age_years` | integer | 19 to 75 | age in years |
| `other_installment_plans` | category | A141 to A143 | other installment plans |
| `housing` | category | A151 to A153 | housing situation |
| `existing_credits` | integer | 1 to 4 | number of existing credits at this bank |
| `job` | category | A171 to A174 | job type |
| `people_liable` | integer | 1 to 2 | number of people liable for maintenance |
| `telephone` | category | A191, A192 | telephone registered in the customer's name |
| `foreign_worker` | category | A201, A202 | foreign worker flag |

The pipeline adds one derived feature, `credit_amount_per_month = credit_amount / duration_months`, and never uses the target to build a feature.

## Category codes

| Code | Meaning |
|---|---|
| A11 | checking account overdrawn |
| A12 | checking balance 0 to 200 DM |
| A13 | checking balance 200 DM or more |
| A14 | no checking account |
| A30 | no credits taken, or all paid back |
| A31 | all credits at this bank paid back |
| A32 | existing credits paid back so far |
| A33 | past delay in payments |
| A34 | critical account |
| A40 | new car |
| A41 | used car |
| A42 | furniture or equipment |
| A43 | radio or television |
| A44 | domestic appliances |
| A45 | repairs |
| A46 | education |
| A47 | vacation |
| A48 | retraining |
| A49 | business |
| A410 | other purpose |
| A61 | savings below 100 DM |
| A62 | savings 100 to 500 DM |
| A63 | savings 500 to 1000 DM |
| A64 | savings 1000 DM or more |
| A65 | unknown or no savings account |
| A71 | unemployed |
| A72 | employed under 1 year |
| A73 | employed 1 to 4 years |
| A74 | employed 4 to 7 years |
| A75 | employed 7 years or more |
| A91 | male, divorced or separated |
| A92 | female, divorced, separated or married |
| A93 | male, single |
| A94 | male, married or widowed |
| A95 | female, single |
| A101 | none |
| A102 | co-applicant |
| A103 | guarantor |
| A121 | real estate |
| A122 | building society savings or life insurance |
| A123 | car or other property |
| A124 | unknown or no property |
| A141 | bank |
| A142 | stores |
| A143 | none |
| A151 | rent |
| A152 | own |
| A153 | free of charge |
| A171 | unemployed or unskilled non-resident |
| A172 | unskilled resident |
| A173 | skilled employee or official |
| A174 | management, self-employed or highly qualified |
| A191 | no telephone |
| A192 | telephone registered |
| A201 | foreign worker |
| A202 | not a foreign worker |

## Cost matrix

| Outcome | Cost |
|---|---|
| Approve an applicant who turns out bad | 5 |
| Reject an applicant who would have been good | 1 |

The ratio comes from the German Credit documentation. The break-even default probability is `1 / (1 + 5) = 0.167`; the fitted thresholds land at 0.190 for the baseline and 0.200 for LightGBM because isotonic calibration makes the probability grid discrete.

## Processed file

`data/processed/credit_features.csv` has 1000 rows and 26 columns:

| Column | Meaning |
|---|---|
| `row_id` | 0 to 999, stable identifier |
| 21 feature columns | the 20 attributes plus `credit_amount_per_month`, as listed above |
| `default` | 1 when the applicant is bad |
| `target` | original 1 or 2, kept for audit |
| `fold` | 0 to 4 for development rows, -1 for held-out rows |
| `is_heldout` | 1 for the 200 held-out rows |

The split is produced with `train_test_split(stratify=default, test_size=0.2, random_state=42)` followed by `StratifiedKFold(5, shuffle=True, random_state=42)` on the development partition.
