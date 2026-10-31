"""Optuna search on the development folds only.

The objective is the mean expected cost per 1000 applications across the
stratified CV folds of the development partition, computed with the same
in-fold calibration and threshold rule as ``train``. The held-out 20 percent is
never touched: tuning reads only rows with ``is_heldout == 0``.

Each trial reports the running mean after every fold so the median pruner can stop
hopeless configurations early. Results are written to ``configs/tuned.yaml`` with a
header naming the date, the number of trials and the objective value per model.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from credit_risk.config import Config
from credit_risk.data.schema import LABEL_COLUMN
from credit_risk.features.build import load_processed
from credit_risk.models import candidates
from credit_risk.models.metrics import expected_cost_per_1000
from credit_risk.models.train import fit_calibrated

TIMEOUT_MESSAGE = "optuna is not installed"


def _naive(value: datetime) -> datetime:
    """Optuna hands back the timestamps sqlite stored, which carry no timezone."""
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


class OptunaUnavailable(RuntimeError):
    """Raised when optuna cannot be imported."""


def available() -> bool:
    return importlib.util.find_spec("optuna") is not None


def _require_optuna():
    if not available():
        raise OptunaUnavailable(
            "hyperparameter search needs optuna; install it with "
            "python -m pip install -e '.[modern]'"
        )
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def suggest(trial, model_type: str) -> dict:
    """Search space per candidate, deliberately modest for 800 rows."""
    if model_type == "logistic":
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth": trial.suggest_int("depth", 2, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
        }
    raise ValueError(f"no search space for model type {model_type}")


def _development_folds(frame: pd.DataFrame) -> list[int]:
    development = frame[frame["is_heldout"] == 0]
    return sorted(int(value) for value in development["fold"].unique())


def cv_cost(cfg: Config, frame: pd.DataFrame, model_type: str, params: dict, trial=None) -> dict:
    """Mean held-out-fold cost over the development partition."""
    development = frame[frame["is_heldout"] == 0].reset_index(drop=True)
    folds = _development_folds(frame)
    costs: list[float] = []
    aucs: list[float] = []
    auc = None
    for step, fold in enumerate(folds):
        train_part = development[development["fold"] != fold]
        valid_part = development[development["fold"] == fold]
        estimator = candidates.make_pipeline(model_type, cfg, override=params)
        model = fit_calibrated(estimator, train_part, cfg, model_type)
        probability = model.predict_proba_bad(valid_part)
        target = valid_part[LABEL_COLUMN].to_numpy(dtype=int)
        costs.append(expected_cost_per_1000(target, probability, model.threshold, cfg.cost))
        if len(np.unique(target)) > 1:
            from sklearn.metrics import roc_auc_score

            aucs.append(float(roc_auc_score(target, probability)))
        if trial is not None:
            trial.report(float(np.mean(costs)), step)
            if trial.should_prune():
                optuna = _require_optuna()
                raise optuna.TrialPruned()
    auc = float(np.mean(aucs)) if aucs else float("nan")
    return {
        "cost_per_1000": float(np.mean(costs)),
        "cost_sd": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
        "roc_auc": auc,
        "folds": len(costs),
    }


def _study(cfg: Config, model_type: str, trials: int, timeout: int):
    optuna = _require_optuna()
    storage = f"sqlite:///{cfg.optuna_storage.as_posix()}"
    cfg.optuna_storage.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=cfg.train.search.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    return optuna.create_study(
        study_name=f"credit_risk_{model_type}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


def tune(
    cfg: Config,
    models: list[str] | None = None,
    trials: int | None = None,
    timeout: int | None = None,
) -> dict:
    """Run one study per candidate; return best parameters and objective values."""
    optuna = _require_optuna()
    from credit_risk.models import registry

    frame = load_processed(cfg)
    requested = models or list(cfg.train.candidates)
    usable, skipped = candidates.available_candidates(requested)
    n_trials = trials or cfg.train.search.trials
    limit = timeout if timeout is not None else cfg.train.search.timeout_seconds
    results: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "objective": "mean expected cost per 1000 across development folds",
        "trials": n_trials,
        "models": {},
        "skipped": skipped,
    }

    for model_type in usable:
        study = _study(cfg, model_type, n_trials, limit)
        # Studies are resumed by name, so the trial list holds every earlier
        # invocation too. Record both counts: the run just made and the study
        # total the sampler actually saw.
        started = datetime.now(UTC)

        def objective(trial, model_type=model_type):
            params = suggest(trial, model_type)
            result = cv_cost(cfg, frame, model_type, params, trial=trial)
            trial.set_user_attr("roc_auc", result["roc_auc"])
            return result["cost_per_1000"]

        study.optimize(objective, n_trials=n_trials, timeout=limit or None, gc_after_trial=True)
        this_run = [
            trial
            for trial in study.trials
            if trial.datetime_start is not None
            and _naive(trial.datetime_start) >= _naive(started)
        ]
        pruned = sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED)
        results["models"][model_type] = {
            "study": study.study_name,
            "storage": str(cfg.optuna_storage),
            "trials": len(study.trials),
            "trials_this_run": len(this_run),
            "pruned": pruned,
            "pruned_this_run": sum(
                1 for trial in this_run if trial.state == optuna.trial.TrialState.PRUNED
            ),
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "best_trial_number": int(study.best_trial.number),
            "best_roc_auc": study.best_trial.user_attrs.get("roc_auc"),
        }
    results["git_commit"] = registry.git_commit(cfg.root)
    return results


def write_tuned_config(
    cfg: Config, results: dict, source: Path, destination: Path
) -> Path:
    """Copy the config and replace the tuned model blocks."""
    body = yaml.safe_load(source.read_text(encoding="utf-8"))
    body["train"]["candidates"] = list(cfg.train.candidates)
    if body["train"]["deployed_model"] not in body["train"]["candidates"]:
        body["train"]["deployed_model"] = body["train"]["candidates"][0]
    for model_type, entry in results["models"].items():
        params = dict(body["train"][model_type])
        params.update(entry["best_params"])
        body["train"][model_type] = params
    lines = [
        f"# Generated by `python -m credit_risk tune --trials {results['trials']}`"
        f" on {results['generated_at']}.",
        f"# Objective: {results['objective']}.",
        f"# Source config: {source.name}; commit {results.get('git_commit', 'unknown')}.",
    ]
    for model_type, entry in results["models"].items():
        lines.append(
            f"# {model_type}: best {entry['best_value']:.2f} per 1000, trial"
            f" {entry.get('best_trial_number', '?')} of {entry['trials']} in this study;"
            f" {entry.get('trials_this_run', entry['trials'])} trials run by this command"
            f" and {entry['pruned']} pruned in total."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(lines) + "\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def format_results(results: dict) -> str:
    lines = [f"trials {results['trials']}  objective {results['objective']}"]
    for model_type, entry in results["models"].items():
        lines.append(
            f"{model_type:10s} best {entry['best_value']:8.2f}"
            f"  trials {entry.get('trials_this_run', entry['trials']):3d} this run"
            f" / {entry['trials']:3d} in the study"
            f"  pruned {entry['pruned']:3d}"
            f"  best trial {entry.get('best_trial_number', 0):3d}"
            f"  params {entry['best_params']}"
        )
    for name, reason in results.get("skipped", {}).items():
        lines.append(f"skipped {name}: {reason}")
    return "\n".join(lines)
