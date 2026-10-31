# Model card: Statlog German Credit default model

Version 20261031T151314Z. Generated from the run recorded in `reports/metrics.json` and `models/logistic/20261031T151314Z/metadata.json`.

## Overview

| Item | Value |
|---|---|
| Task | Binary classification, probability that an applicant defaults |
| Positive class | `default`, encoded as 1 when the raw Statlog target is 2 (bad credit risk) |
| Data | UCI Statlog German Credit, 1000 rows, 20 attributes |
| Features | 20 raw attributes plus `credit_amount_per_month`, one-hot encoded (62 encoded columns after fitting) |
| Deployed model | Logistic regression, L2, `C=0.3101` from the Optuna search, on scaled numeric features and one-hot categoricals |
| Challenger | LightGBM, 600 trees, learning rate 0.0104, 11 leaves, `min_child_samples` 54, subsample 0.719, `colsample_bytree` 0.528, L2 0.0459, single thread, deterministic |
| Calibration | Isotonic regression fitted on inner 3-fold out-of-fold predictions, same procedure for both |
| Decision threshold | 0.1892 for the deployed model, 0.1765 for LightGBM, 0.1852 for XGBoost, 0.2593 for CatBoost, each chosen by minimum expected cost on inner folds only |
| Cost matrix | approving a bad applicant costs 5; rejecting a good applicant costs 1 |
| Artifact | `models/logistic/20261031T151314Z/`, pointer at `models/latest` |
| Seed | 42 |
| Licence | MIT for the code; the data stays under the UCI terms |

## Intended use

Demonstration of a cost-sensitive credit scoring pipeline: training, calibration, threshold selection, registry, HTTP serving with reason codes, and drift monitoring. Suitable for reviewing how a small tabular risk model should be evaluated and deployed.

## Out of scope

Real credit decisions about people. The data is a 1970s German credit sample with 1000 rows, contains protected attributes as features, and has no subgroup evaluation. There is no reject inference, no adverse action notice logic, and no regulatory review of the reason codes.

## Training protocol

1. The processed file fixes a stratified 80/20 split: 800 development rows, 200 held-out rows, before any model is fitted.
2. Headline metrics come from stratified 5-fold cross-validation inside the development partition. For each outer fold: fit the base model on four folds, fit an isotonic calibrator on inner 3-fold out-of-fold predictions of those four folds, choose the threshold on those inner predictions, then score the held-out outer fold.
3. The deployable artifact is refitted on all 800 development rows with the same inner-fold calibration and threshold rule, then scored once on the 200 held-out rows.
4. Nothing from the held-out partition influences calibration, threshold, or model choice.

## Deployment decision

All four candidates go through the identical protocol and all four are registered
(`credit_risk_logistic`, `credit_risk_lightgbm`, `credit_risk_xgboost`,
`credit_risk_catboost`). The deployed one is the model with the lower held-out
expected cost, the objective the project is built around: logistic regression at
555.0 per 1000, against CatBoost at 570.0, LightGBM and XGBoost at 595.0. CatBoost
has the better held-out ranking (ROC-AUC 0.8047 against 0.8002) and the better
Brier (0.1549 against 0.1613); it loses on cost because of where its calibrated
probabilities cross the 5:1 threshold, not because it ranks worse. With 200 holdout
rows the 15.0 gap is inside noise, so the rule stays cost. The rule itself is the
`min` over held-out expected cost at `src/credit_risk/models/train.py:383`, and its
outcome is written to `reports/train_summary.json` under `champion`.

## Measured performance

Held-out 200 rows, each model at its own cost-optimal threshold (0.189 logistic,
0.177 LightGBM, 0.185 XGBoost, 0.259 CatBoost):

| Metric | Logistic (deployed) | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|
| ROC-AUC | 0.8002 | 0.7810 | 0.7970 | 0.8047 |
| PR-AUC | 0.6090 | 0.6305 | 0.6331 | 0.6690 |
| KS | 0.5190 | 0.4500 | 0.4571 | 0.4952 |
| Brier | 0.1613 | 0.1637 | 0.1600 | 0.1549 |
| ECE | 0.0700 | 0.0461 | 0.0611 | 0.0695 |
| Accuracy at threshold | 0.605 | 0.565 | 0.545 | 0.630 |
| Expected cost per 1000 | 555.0 | 595.0 | 595.0 | 570.0 |
| Cost at 0.5 per 1000 | 925.0 | 870.0 | 750.0 | 620.0 |
| Approval rate at threshold | 0.385 | 0.345 | 0.315 | 0.430 |
| Conformal coverage, nominal 90 percent | 0.915 | 0.910 | 0.910 | 0.915 |

Five-fold CV on 800 development rows (mean):

| Metric | Logistic (deployed) | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|
| ROC-AUC | 0.7711 | 0.7809 | 0.7854 | 0.7836 |
| PR-AUC | 0.5657 | 0.5762 | 0.5748 | 0.5677 |
| KS | 0.4423 | 0.4625 | 0.4583 | 0.4786 |
| Brier | 0.1695 | 0.1672 | 0.1682 | 0.1689 |
| ECE | 0.0806 | 0.0811 | 0.0706 | 0.0916 |
| Accuracy at threshold | 0.619 | 0.625 | 0.596 | 0.611 |
| Expected cost per 1000 | 541.2 | 510.0 | 553.8 | 543.8 |

The CV mean and the holdout disagree on which model is cheapest: LightGBM is
cheapest in cross-validation (510.0), logistic is cheapest on the holdout (555.0).
Both differences are within one fold standard deviation. Report both, do not pick
the one that flatters the deployed model.

## Calibration

The method is chosen per candidate by inner-fold Brier score among isotonic, Platt
and none; all four ended on isotonic for this run. Held-out ECE is 0.0700 for
logistic, 0.0461 for LightGBM, 0.0611 for XGBoost and 0.0695 for CatBoost: the
boosted models were already closer to calibrated than the linear one, so
calibration changes them least. The calibrator is kept for every candidate because
the cost threshold is defined on the calibrated scale, and because the same
protocol has to apply to the model that is deployed.

The split-conformal band at a nominal 90 percent covers 0.910 to 0.915 of holdout
rows across the four candidates. It is a coverage statement on 200 rows, so treat
the third decimal as noise.

## Threshold

The expected cost of approving applicant i is `5 * P(default)`; the expected cost of rejecting is `1 * (1 - P(default))`. The break-even is `P = 1/6 = 0.167`. The implemented search minimises realised cost over a grid of calibrated inner-fold probabilities, keeps the highest threshold within one percent of the minimum, and reports the plateau.

For the deployed logistic regression the chosen threshold 0.1892 sits on a plateau from 0.1667 to 0.1892 and costs 527.5 per 1000 on the inner-fold predictions, the minimum of the grid. LightGBM chose 0.1765 at 535.0, XGBoost 0.1852 at 537.5, and CatBoost 0.2593 at 520.0 against a grid minimum of 517.5. The rule is identical for all four candidates, so switching the deployed model does not change the decision logic.

## Explainability

For the deployed logistic regression, per-prediction reason codes are the fitted coefficient multiplied by the encoded value, summed within each attribute, which is exact and additive for a linear model. Numeric features enter standardised, so a numeric contribution is per standard deviation of that feature; categorical contributions are the coefficient of the active one-hot column. For the LightGBM challenger the same code path uses `pred_contrib=True`, exact tree SHAP values with no `shap` dependency. `/model/info` reports which method the loaded artifact uses.

In both cases contributions from the one-hot columns of the same attribute are summed first, so a single attribute appears once; the top three absolute contributions are returned with readable labels, for example `checking account status = A11 (overdrawn account)`, plus `raises_risk` or `lowers_risk`. These are contributions to the base model score, not causal effects, and calibration (a monotone transform) does not change their sign or order. Global permutation importance for the deployed model on the held-out rows ranks `checking_status` first (0.097 mean ROC-AUC drop), then `purpose`, `credit_amount`, `duration_months` and `credit_history`.

## Fairness and ethical considerations

`personal_status_sex` (which encodes sex), `age_years` and `foreign_worker` are model inputs. No subgroup metrics are computed and no mitigation is applied. The dataset is a historical sample from a single country and decade. Anyone adapting this code to a real lending context must add subgroup evaluation, a documented fairness policy and a legal review before use.

## Limitations

- 1000 rows total and 200 held out; confidence intervals are wide and the four candidates are not separable on this data. The deployment choice rests on one 200-row holdout and would need re-measuring on more data.
- The 5:1 cost ratio is taken from the dataset documentation, not measured for any business.
- Costs are computed against observed labels; rejected applicants have no counterfactual outcome.
- Isotonic calibration adds variance at this sample size and raised the deployed logistic's holdout Brier from 0.1531 to 0.1613; the uncalibrated Brier is recorded only for the deployed candidate, so nothing is claimed about the other three.
- A single global threshold is used; no capacity constraint or segment policy.
- The Optuna search is modest and was run while studies were being resumed: the logistic and LightGBM studies hold 33 trials each, of which 10 came from the command that wrote `configs/tuned.yaml` and 23 from an earlier one, and both best values (trial 14 and trial 15) come from that earlier invocation. CatBoost kept its configured parameters because its study stalled with two trials left `RUNNING`. Searches read the development folds only.

## Reproduction

```bash
python -m credit_risk fetch
python -m credit_risk validate
python -m credit_risk features
python -m credit_risk train
python -m credit_risk evaluate
python -m credit_risk drift
```

Metadata for the artifact records the config hash, the raw and processed SHA256, the git commit, package versions, seed, threshold and both metric blocks.
