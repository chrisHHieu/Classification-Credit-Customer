"""Versioned artifact registry: ``models/<model_type>/<version>/``.

Each version directory holds ``model.joblib``, ``metadata.json`` and
``reference.json`` (the training distributions used by drift). A single text file
``models/latest`` names the deployed artifact as ``<model_type>/<version>``, which
is what the API loads at startup.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib

from credit_risk.config import Config

LATEST_FILE = "latest"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.json"


class RegistryError(RuntimeError):
    """Raised when an artifact is missing or malformed."""


def new_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_commit(root: Path) -> str:
    """Current commit hash, or ``unknown`` when git is unavailable or fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def artifact_dir(cfg: Config, model_type: str, version: str) -> Path:
    return cfg.models_dir / model_type / version


def save_artifact(
    cfg: Config,
    model_type: str,
    version: str,
    bundle,
    metadata: dict,
    reference: dict | None = None,
) -> Path:
    directory = artifact_dir(cfg, model_type, version)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / MODEL_FILE)
    payload = dict(metadata)
    payload["reference_file"] = REFERENCE_FILE
    (directory / METADATA_FILE).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if reference is not None:
        (directory / REFERENCE_FILE).write_text(
            json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return directory


def write_latest(cfg: Config, model_type: str, version: str) -> Path:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.models_dir / LATEST_FILE
    path.write_text(f"{model_type}/{version}\n", encoding="utf-8")
    return path


def read_latest(cfg: Config) -> tuple[str, str]:
    path = cfg.models_dir / LATEST_FILE
    if not path.exists():
        raise RegistryError(f"{path} not found, run 'credit_risk train' first")
    content = path.read_text(encoding="utf-8").strip()
    if "/" not in content:
        raise RegistryError(f"{path} should contain '<model_type>/<version>', found {content!r}")
    model_type, version = content.split("/", 1)
    return model_type, version


def load_metadata(cfg: Config, model_type: str | None = None, version: str | None = None) -> dict:
    if model_type is None or version is None:
        model_type, version = read_latest(cfg)
    path = artifact_dir(cfg, model_type, version) / METADATA_FILE
    if not path.exists():
        raise RegistryError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(cfg: Config, model_type: str | None = None, version: str | None = None):
    if model_type is None or version is None:
        model_type, version = read_latest(cfg)
    path = artifact_dir(cfg, model_type, version) / MODEL_FILE
    if not path.exists():
        raise RegistryError(f"{path} not found, run 'credit_risk train' first")
    return joblib.load(path), load_metadata(cfg, model_type, version)


def list_versions(cfg: Config, model_type: str) -> list[str]:
    directory = cfg.models_dir / model_type
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def load_reference(cfg: Config, model_type: str | None = None, version: str | None = None) -> dict:
    if model_type is None or version is None:
        model_type, version = read_latest(cfg)
    path = artifact_dir(cfg, model_type, version) / REFERENCE_FILE
    if not path.exists():
        raise RegistryError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))
