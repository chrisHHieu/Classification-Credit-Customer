"""Typed configuration loaded from YAML.

The project keeps its dependency list deliberately small (see the allowed set in
``pyproject.toml``), so the config tree is built from frozen dataclasses instead of
pydantic models. The guarantees are the same as far as this project needs them:
unknown keys are rejected, every section has explicit fields and types, and the
resolved settings can be hashed into artifact metadata.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs") / "base.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing fields or has unknown keys."""


def _build(cls: type, raw: Any, where: str) -> Any:
    """Build a dataclass from a mapping, rejecting unknown and missing keys."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    missing = sorted(names - set(raw))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")
    return cls(**raw)


def _as_type(value: Any, expected: type, where: str) -> Any:
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got bool")
    if not isinstance(value, expected):
        raise ConfigError(f"{where}: expected {expected.__name__}, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Paths:
    raw: str
    processed: str
    reports: str
    figures: str
    models: str


@dataclass(frozen=True)
class DataConfig:
    source_url: str
    fallback_url: str
    description_url: str
    filename: str
    description_filename: str
    expected_rows: int
    expected_bad_count: int
    user_agent: str
    timeout_seconds: float
    retries: int
    backoff_seconds: float
    min_interval_seconds: float

    def __post_init__(self) -> None:
        _as_type(self.expected_rows, int, "data.expected_rows")
        _as_type(self.expected_bad_count, int, "data.expected_bad_count")
        _as_type(self.timeout_seconds, float, "data.timeout_seconds")
        _as_type(self.retries, int, "data.retries")
        _as_type(self.backoff_seconds, float, "data.backoff_seconds")
        _as_type(self.min_interval_seconds, float, "data.min_interval_seconds")


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    n_splits: int

    def __post_init__(self) -> None:
        _as_type(self.test_size, float, "split.test_size")
        _as_type(self.n_splits, int, "split.n_splits")
        if not 0.0 < self.test_size < 1.0:
            raise ConfigError("split.test_size must be between 0 and 1")
        if self.n_splits < 2:
            raise ConfigError("split.n_splits must be at least 2")


@dataclass(frozen=True)
class CostConfig:
    cost_bad_approved: int
    cost_good_rejected: int
    threshold_min: float
    threshold_max: float
    tolerance: float

    def __post_init__(self) -> None:
        _as_type(self.cost_bad_approved, int, "cost.cost_bad_approved")
        _as_type(self.cost_good_rejected, int, "cost.cost_good_rejected")
        _as_type(self.threshold_min, float, "cost.threshold_min")
        _as_type(self.threshold_max, float, "cost.threshold_max")
        _as_type(self.tolerance, float, "cost.tolerance")
        if self.cost_bad_approved <= 0 or self.cost_good_rejected <= 0:
            raise ConfigError("cost values must be positive")
        if not 0.0 < self.threshold_min < self.threshold_max < 1.0:
            raise ConfigError("cost.threshold_min must be below cost.threshold_max inside (0, 1)")


@dataclass(frozen=True)
class CalibrationConfig:
    method: str
    inner_splits: int

    def __post_init__(self) -> None:
        _as_type(self.inner_splits, int, "calibration.inner_splits")
        if self.method != "isotonic":
            raise ConfigError("calibration.method must be 'isotonic'")
        if self.inner_splits < 2:
            raise ConfigError("calibration.inner_splits must be at least 2")


@dataclass(frozen=True)
class TrainConfig:
    deployed_model: str
    label: str
    lightgbm: dict[str, Any] = field(default_factory=dict)
    logistic: dict[str, Any] = field(default_factory=dict)
    permutation_repeats: int = 10

    def __post_init__(self) -> None:
        _as_type(self.permutation_repeats, int, "train.permutation_repeats")
        if self.deployed_model not in {"lightgbm", "logistic"}:
            raise ConfigError("train.deployed_model must be 'lightgbm' or 'logistic'")
        if not isinstance(self.lightgbm, dict) or not isinstance(self.logistic, dict):
            raise ConfigError("train.lightgbm and train.logistic must be mappings")


@dataclass(frozen=True)
class DriftConfig:
    bins: int
    warn: float
    alert: float

    def __post_init__(self) -> None:
        _as_type(self.bins, int, "drift.bins")
        _as_type(self.warn, float, "drift.warn")
        _as_type(self.alert, float, "drift.alert")
        if self.bins < 2:
            raise ConfigError("drift.bins must be at least 2")
        if not 0.0 < self.warn < self.alert:
            raise ConfigError("drift thresholds must satisfy 0 < warn < alert")


@dataclass(frozen=True)
class ServingConfig:
    host: str
    port: int

    def __post_init__(self) -> None:
        _as_type(self.port, int, "serving.port")


@dataclass(frozen=True)
class Config:
    seed: int
    root: Path
    paths: Paths
    data: DataConfig
    split: SplitConfig
    cost: CostConfig
    calibration: CalibrationConfig
    train: TrainConfig
    drift: DriftConfig
    serving: ServingConfig
    source_file: Path | None = None

    # Path helpers -----------------------------------------------------------------
    def _dir(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.root / path)

    @property
    def raw_dir(self) -> Path:
        return self._dir(self.paths.raw)

    @property
    def processed_dir(self) -> Path:
        return self._dir(self.paths.processed)

    @property
    def reports_dir(self) -> Path:
        return self._dir(self.paths.reports)

    @property
    def figures_dir(self) -> Path:
        return self._dir(self.paths.figures)

    @property
    def models_dir(self) -> Path:
        return self._dir(self.paths.models)

    @property
    def raw_file(self) -> Path:
        return self.raw_dir / self.data.filename

    @property
    def manifest_file(self) -> Path:
        return self.raw_dir / "manifest.json"

    @property
    def description_file(self) -> Path:
        return self.raw_dir / self.data.description_filename

    @property
    def processed_file(self) -> Path:
        return self.processed_dir / "credit_features.csv"

    @property
    def train_summary_file(self) -> Path:
        return self.reports_dir / "train_summary.json"

    @property
    def metrics_file(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def drift_file(self) -> Path:
        return self.reports_dir / "drift.json"

    # Metadata helpers -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        as_dict = dataclasses.asdict(self)
        as_dict.pop("root", None)
        as_dict.pop("source_file", None)
        return as_dict

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_types(cfg: Config) -> None:
    _as_type(cfg.seed, int, "seed")


def load_config(
    path: str | Path | None = None,
    root: str | Path | None = None,
) -> Config:
    """Load the YAML config, resolving relative run directories against ``root``."""
    config_path = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a mapping at the top level")

    required = {"seed", "paths", "data", "split", "cost", "calibration", "train", "drift"}
    unknown = sorted(set(raw) - required - {"serving"})
    if unknown:
        raise ConfigError(f"{config_path}: unknown sections {unknown}")
    if "serving" not in raw:
        raise ConfigError(f"{config_path}: missing section 'serving'")

    base_root = Path(root) if root else Path.cwd()
    base_root = base_root if base_root.is_absolute() else base_root.resolve()

    cfg = Config(
        seed=_as_type(raw["seed"], int, "seed"),
        root=base_root,
        paths=_build(Paths, raw["paths"], "paths"),
        data=_build(DataConfig, raw["data"], "data"),
        split=_build(SplitConfig, raw["split"], "split"),
        cost=_build(CostConfig, raw["cost"], "cost"),
        calibration=_build(CalibrationConfig, raw["calibration"], "calibration"),
        train=_build(TrainConfig, raw["train"], "train"),
        drift=_build(DriftConfig, raw["drift"], "drift"),
        serving=_build(ServingConfig, raw["serving"], "serving"),
        source_file=config_path,
    )
    _validate_types(cfg)
    return cfg
