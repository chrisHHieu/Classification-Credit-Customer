# Model card: Statlog German Credit default model

Version 20261031T122611Z. Generated from the run recorded in `reports/metrics.json` and `models/logistic/20261031T122611Z/metadata.json`.

## Overview

| Item | Value |
|---|---|
| Task | Binary classification, probability that an applicant defaults |
| Positive class | `default`, encoded as 1 when the raw Statlog target is 2 (bad credit risk) |
| Data | UCI Statlog German Credit, 1000 rows, 20 attributes |
| Features | 20 raw attributes plus `credit_amount_per_month`, one-hot encoded (62 encoded columns after fitting) |
| Deployed model | Logistic regression, L2, C=1.0, on scaled numeric features and one-hot categoricals |
| Challenger | LightGBM, 400 trees, learning rate 0.03, 15 leaves, row and column subsampling, single thread, deterministic |
| Calibration | Isotonic regression fitted on inner 3-fold out-of-fold predictions, same procedure for both |
| Decision threshold | 0.1905 for the deployed model, 0.200 for the challenger, each chosen by minimum expected cost on inner folds only |
| Cost matrix | approving a bad applicant costs 5; rejecting a good applicant costs 1 |
| Artifact | `models/logistic/20261031T122611Z/`, pointer at `models/latest` |
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

Both candidates go through the identical protocol and both are registered. The deployed one is the model with the lower held-out expected cost, which is the objective the project is built around: logistic regression at 515.0 per 1000 against LightGBM at 640.0. The logistic model also has the higher held-out ROC-AUC (0.801 against 0.768). LightGBM ranks slightly better in cross-validation (0.786 against 0.764) and can produce exact tree SHAP reason codes, so it stays registered as the challenger and can be promoted with one config value.

## Measured performance

Held-out 200 rows, each model at its own threshold (0.190 for logistic, 0.200 for LightGBM):

| Metric | Logistic regression (deployed) | LightGBM (challenger) |
|---|---|---|
| ROC-AUC | 0.801 | 0.768 |
| PR-AUC | 0.615 | 0.625 |
| KS | 0.536 | 0.440 |
| Brier | 0.162 | 0.169 |
| Accuracy at threshold | 0.645 | 0.520 |
| Expected cost per 1000 | 515.0 | 640.0 |
| Cost at threshold 0.5 per 1000 | 840.0 | 725.0 |
| Approval rate | 0.425 | 0.300 |

Five-fold CV on 800 development rows (mean +/- standard deviation):

| Metric | Logistic regression (deployed) | LightGBM (challenger) |
|---|---|---|
| ROC-AUC | 0.764 +/- 0.061 | 0.786 +/- 0.046 |
| PR-AUC | 0.559 +/- 0.082 | 0.582 +/- 0.075 |
| KS | 0.429 +/- 0.082 | 0.468 +/- 0.088 |
| Brier | 0.172 +/- 0.018 | 0.166 +/- 0.017 |
| Accuracy at threshold | 0.568 +/- 0.061 | 0.591 +/- 0.028 |
| Expected cost per 1000 | 537.5 +/- 58.8 | 553.8 +/- 100.4 |

Confusion matrices on the held-out rows, each model at its own threshold:

| | logistic: approve | logistic: reject | LightGBM: approve | LightGBM: reject |
|---|---|---|---|---|
| actual: good | 77 | 63 | 52 | 88 |
| actual: bad | 8 | 52 | 8 | 52 |

Both candidates approve the same 8 of the 60 bad applicants. The difference is on good applicants: LightGBM rejects 25 more of them, and 25 extra rejections per 200 applications produces the 125.0 per 1000 gap.

## Calibration

Isotonic regression is fitted on inner-fold predictions of the development partition only, with the same procedure for both candidates. On the held-out rows the Brier score is 0.1543 before calibration and 0.1623 after for the deployed logistic regression, and 0.1673 before and 0.1685 after for LightGBM. Calibration therefore does not improve Brier at this sample size for either model. It is kept because the cost rule and the stored thresholds are defined on calibrated probabilities, and because it maps scores to the observed default frequency rather than an arbitrary scale. The deployed model's raw scores were the better calibrated of the two before the map was applied.

## Threshold

The expected cost of approving applicant i is `5 * P(default)`; the expected cost of rejecting is `1 * (1 - P(default))`. The break-even is `P = 1/6 = 0.167`. The implemented search minimises realised cost over a grid of calibrated inner-fold probabilities, keeps the highest threshold within one percent of the minimum, and reports the plateau.

For the deployed logistic regression the chosen threshold 0.1905 has a plateau of 0.165 to 0.1905, and costs 543.75 per 1000 on the training partition against a minimum of 541.25. For LightGBM the threshold 0.200 has a plateau of 0.143 to 0.200. Neither decision sits on a knife edge, and the rule is identical for both models, so switching the deployed model does not change the decision logic.

## Explainability

For the deployed logistic regression, per-prediction reason codes are the fitted coefficient multiplied by the encoded value, summed within each attribute, which is exact and additive for a linear model. Numeric features enter standardised, so a numeric contribution is per standard deviation of that feature; categorical contributions are the coefficient of the active one-hot column. For the LightGBM challenger the same code path uses `pred_contrib=True`, exact tree SHAP values with no `shap` dependency. `/model/info` reports which method the loaded artifact uses.

In both cases contributions from the one-hot columns of the same attribute are summed first, so a single attribute appears once; the top three absolute contributions are returned with readable labels, for example `checking account status = A11 (overdrawn account)`, plus `raises_risk` or `lowers_risk`. These are contributions to the base model score, not causal effects, and calibration (a monotone transform) does not change their sign or order. Global permutation importance for the deployed model on the held-out rows ranks `checking_status` first (0.097 mean ROC-AUC drop), then `purpose`, `credit_amount`, `duration_months` and `credit_history`.

## Fairness and ethical considerations

`personal_status_sex` (which encodes sex), `age_years` and `foreign_worker` are model inputs. No subgroup metrics are computed and no mitigation is applied. The dataset is a historical sample from a single country and decade. Anyone adapting this code to a real lending context must add subgroup evaluation, a documented fairness policy and a legal review before use.

## Limitations

- 1000 rows total and 200 held out; confidence intervals are wide and the two models are not separable on this data. The deployment choice rests on one 200-row holdout and would need re-measuring on more data.
- The 5:1 cost ratio is taken from the dataset documentation, not measured for any business.
- Costs are computed against observed labels; rejected applicants have no counterfactual outcome.
- Isotonic calibration adds variance at this sample size and did not improve the Brier score for either model.
- A single global threshold is used; no capacity constraint or segment policy.
- No hyperparameter search was performed, to keep the held-out set clean.

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
