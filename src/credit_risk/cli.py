"""Command line interface: fetch, validate, features, tune, train, evaluate, drift, serve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from credit_risk.config import Config, ConfigError, load_config
from credit_risk.data.fetch import FetchError, fetch, load_raw
from credit_risk.data.validate import validate_file
from credit_risk.features.build import build_dataset, split_frames, write_processed
from credit_risk.models import registry


def _common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--root", default=None, help="directory that run paths resolve against")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="credit_risk",
        description="Credit default model on the Statlog German Credit data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", parents=[common], help="download raw data")
    fetch_parser.add_argument("--force", action="store_true", help="ignore the cached copy")

    subparsers.add_parser("validate", parents=[common], help="check the data contract")
    subparsers.add_parser("features", parents=[common], help="build processed features and split")

    train_parser = subparsers.add_parser("train", parents=[common], help="fit and register models")
    train_parser.add_argument("--version", default=None, help="artifact version label")
    train_parser.add_argument(
        "--track",
        action="store_true",
        help="log every candidate to MLflow (MLFLOW_TRACKING_URI, default sqlite:///mlflow.db)",
    )

    subparsers.add_parser("evaluate", parents=[common], help="write reports and figures")

    tune_parser = subparsers.add_parser(
        "tune", parents=[common], help="optuna search on the development folds"
    )
    tune_parser.add_argument("--trials", type=int, default=None, help="trials per model")
    tune_parser.add_argument(
        "--timeout", type=int, default=None, help="seconds per study, 0 or unset means no limit"
    )
    tune_parser.add_argument(
        "--models", nargs="+", default=None, help="subset of candidate names to tune"
    )
    tune_parser.add_argument(
        "--output", default=None, help="write the tuned config here (default configs/tuned.yaml)"
    )

    drift_parser = subparsers.add_parser(
        "drift", parents=[common], help="PSI/Evidently drift report"
    )
    drift_parser.add_argument("--input", default=None, help="CSV to score, defaults to processed")
    drift_parser.add_argument(
        "--split", default="both", choices=["all", "holdout", "both"], help="which slice to test"
    )
    drift_parser.add_argument(
        "--evidently", action="store_true", help="also run the Evidently statistical tests"
    )

    serve_parser = subparsers.add_parser("serve", parents=[common], help="run the HTTP API")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, root=args.root)


def _ensure_directories(cfg: Config) -> None:
    for path in (
        cfg.raw_dir,
        cfg.processed_dir,
        cfg.reports_dir,
        cfg.figures_dir,
        cfg.models_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def cmd_fetch(cfg: Config, args: argparse.Namespace) -> int:
    _ensure_directories(cfg)
    manifest = fetch(cfg, force=args.force)
    print(f"source {manifest['source_url_used']}")
    print(
        f"cache {manifest.get('cache', 'n/a')}  rows {manifest['rows']}"
        f"  sha256 {manifest['sha256']}"
    )
    print(f"written {cfg.raw_file}")
    print(f"manifest {cfg.manifest_file}")
    return 0


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    report = validate_file(cfg)
    print(report.summary())
    return 0 if report.ok else 1


def cmd_features(cfg: Config, args: argparse.Namespace) -> int:
    from credit_risk.data import contracts

    report = validate_file(cfg)
    if not report.ok:
        print(report.summary(), file=sys.stderr)
        return 1
    frame = build_dataset(load_raw(cfg), cfg)
    path = write_processed(frame, cfg)
    contract = contracts.check(frame, "processed")
    print(f"contract {contracts.format_check(contract)}")
    parts = split_frames(frame)
    print(f"processed {len(frame)} rows x {frame.shape[1]} columns -> {path}")
    print(
        f"development {len(parts['development'])} rows, holdout {len(parts['holdout'])} rows,"
        f" folds {sorted(frame.loc[frame['is_heldout'] == 0, 'fold'].unique().tolist())}"
    )
    print(f"default rate {frame['default'].mean():.3f}")
    return 0


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from credit_risk.models.train import format_summary, run_training

    _ensure_directories(cfg)
    summary = run_training(cfg, version=args.version, track=args.track)
    print(format_summary(summary))
    print(f"summary {cfg.train_summary_file}")
    for model_type, record in summary.get("tracking", {}).items():
        if record.get("run_id"):
            print(f"mlflow {model_type}: run {record['run_id']}"
                  f" registered {record.get('registered_model')}"
                  f" version {record.get('version')}")
        if record.get("fallback"):
            print(f"mlflow {model_type}: {record['fallback']}")
    champion = summary.get("champion", {}).get("alias")
    if champion:
        print(f"mlflow alias champion: {champion}")
    return 0


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    from credit_risk.models.evaluate import EvaluationError, format_evaluation, run_evaluation

    _ensure_directories(cfg)
    try:
        payload = run_evaluation(cfg)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_evaluation(payload))
    print(f"metrics {cfg.metrics_file}")
    return 0


def cmd_drift(cfg: Config, args: argparse.Namespace) -> int:
    from credit_risk.monitor.drift import format_drift, run_drift

    try:
        payload = run_drift(cfg, input_path=args.input, split=args.split)
    except registry.RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_drift(payload))
    print(f"report {cfg.drift_file}")
    if args.evidently:
        from credit_risk.monitor.evidently_report import format_evidently, run_evidently

        try:
            evidently = run_evidently(cfg, input_path=args.input)
        except RuntimeError as exc:
            print(f"evidently skipped: {exc}", file=sys.stderr)
            return 1
        print()
        print(format_evidently(evidently))
        print(f"report {cfg.evidently_json}")
    return 0


def cmd_tune(cfg: Config, args: argparse.Namespace) -> int:
    from credit_risk.tuning.search import (
        OptunaUnavailable,
        format_results,
        tune,
        write_tuned_config,
    )

    _ensure_directories(cfg)
    try:
        results = tune(cfg, models=args.models, trials=args.trials, timeout=args.timeout)
    except OptunaUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_results(results))
    source = cfg.source_file or (cfg.root / "configs" / "base.yaml")
    destination = (
        Path(args.output) if args.output else cfg.root / "configs" / "tuned.yaml"
    )
    path = write_tuned_config(cfg, results, source, destination)
    print(f"tuned config {path}")
    return 0


def cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from credit_risk.serving.app import create_app

    host = args.host or cfg.serving.host
    port = args.port or cfg.serving.port
    print(f"serving on http://{host}:{port}")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


COMMANDS = {
    "fetch": cmd_fetch,
    "validate": cmd_validate,
    "features": cmd_features,
    "tune": cmd_tune,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "drift": cmd_drift,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _config_from_args(args)
        return COMMANDS[args.command](cfg, args)
    except (ConfigError, FetchError, FileNotFoundError, ValueError, registry.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
