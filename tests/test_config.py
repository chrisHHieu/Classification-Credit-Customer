from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from tests.conftest import REPO_ROOT, build_test_config

from credit_risk.config import ConfigError, load_config


def test_default_config_values(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    assert cfg.seed == 42
    assert cfg.cost.cost_bad_approved == 5
    assert cfg.cost.cost_good_rejected == 1
    assert cfg.calibration.method == "isotonic"
    # The test config shrinks the fold count to keep the suite fast.
    assert cfg.split.n_splits == 3
    assert cfg.train.deployed_model == "logistic"
    assert cfg.train.candidates == ["logistic", "lightgbm"]
    assert cfg.conformal.alpha == 0.1
    # The test config shrinks the search budget to two trials.
    assert cfg.train.search.trials == 2
    assert cfg.train.search.seed == 42
    assert cfg.optuna_storage == tmp_path / "reports" / "optuna.db"
    assert cfg.evidently_html == tmp_path / "reports" / "drift_evidently.html"


def test_full_config_lists_all_four_candidates(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "full.yaml", root=tmp_path)
    assert cfg.train.candidates == ["logistic", "lightgbm", "xgboost", "catboost"]
    assert cfg.calibration.method == "auto"
    assert cfg.train.deployed_model in cfg.train.candidates


def test_deployed_model_must_be_a_candidate(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with pytest.raises(ConfigError, match="deployed_model"):
        replace(cfg.train, deployed_model="catboost")


def test_unknown_candidate_is_rejected(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with pytest.raises(ConfigError, match="unknown models"):
        replace(cfg.train, candidates=["logistic", "tabpfn"])


def test_calibration_method_domain(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with pytest.raises(ConfigError, match="calibration.method"):
        replace(cfg.calibration, method="spline")
    assert replace(cfg.calibration, method="auto").method == "auto"


def test_conformal_alpha_must_be_a_probability(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with pytest.raises(ConfigError, match="conformal.alpha"):
        replace(cfg.conformal, alpha=0.0)
    with pytest.raises(ConfigError, match="conformal.alpha"):
        replace(cfg.conformal, alpha=1.5)


def test_paths_resolve_under_root(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    assert cfg.raw_dir == tmp_path / "data" / "raw"
    assert cfg.processed_file == tmp_path / "data" / "processed" / "credit_features.csv"
    assert cfg.metrics_file == tmp_path / "reports" / "metrics.json"
    assert cfg.models_dir == tmp_path / "models"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    source = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    absolute = tmp_path / "absolute_reports"
    source["paths"]["reports"] = str(absolute)
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(source), encoding="utf-8")
    cfg = load_config(custom, root=tmp_path / "elsewhere")
    assert cfg.reports_dir == absolute


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    source["nonsense"] = {"a": 1}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown sections"):
        load_config(path, root=tmp_path)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    source["cost"].pop("tolerance")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(path, root=tmp_path)


def test_bad_cost_matrix_is_rejected(tmp_path: Path) -> None:
    cfg = build_test_config(tmp_path)
    with pytest.raises(ConfigError):
        replace(cfg, cost=replace(cfg.cost, cost_bad_approved=0))


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    first = build_test_config(tmp_path)
    second = build_test_config(tmp_path)
    assert first.config_hash == second.config_hash
    changed = replace(first, seed=7)
    assert changed.config_hash != first.config_hash


def test_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml", root=tmp_path)
