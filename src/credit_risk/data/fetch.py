"""Download the Statlog German Credit data and pin it with a manifest.

The file is a single space separated table with no header. It is not committed:
``fetch`` writes it to ``data/raw/`` next to a sidecar manifest carrying the
source URL, fetch time, SHA256 and basic shape. Re-running without ``--force`` is
a no-op as long as the cached payload still matches its manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_risk.config import Config
from credit_risk.data.schema import COLUMNS, NUMERIC_COLUMNS, TARGET

DATASET_NAME = "statlog-german-credit"

# Minimum spacing between any two outbound requests, in seconds.
MIN_INTERVAL = 1.0
_last_request_at: float = 0.0


class FetchError(RuntimeError):
    """Raised when the payload cannot be downloaded or does not parse."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _throttle(min_interval: float) -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_at = time.monotonic()


def _get(url: str, cfg: Config) -> bytes:
    """One GET with the project user agent, bounded retries and backoff."""
    last_error: Exception | None = None
    for attempt in range(cfg.data.retries):
        _throttle(max(MIN_INTERVAL, cfg.data.min_interval_seconds))
        request = urllib.request.Request(url, headers={"User-Agent": cfg.data.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=cfg.data.timeout_seconds) as response:
                return response.read()
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt + 1 < cfg.data.retries:
                time.sleep(cfg.data.backoff_seconds * (2**attempt))
    raise FetchError(f"could not download {url}: {last_error}")


def _extract_payload(url: str, payload: bytes, filename: str) -> bytes:
    """Return the data file, unzipping a release archive when that is what arrived."""
    if not url.endswith(".zip") and not payload.startswith(b"PK\x03\x04"):
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(filename)]
        if not candidates:
            raise FetchError(f"{url}: archive does not contain {filename}")
        return archive.read(candidates[0])


def _count_rows(payload: bytes) -> int:
    return sum(1 for line in payload.decode("utf-8").splitlines() if line.strip())


def _first_last(payload: bytes) -> tuple[str, str]:
    lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
    return lines[0].strip(), lines[-1].strip()


def _cached_manifest(cfg: Config) -> dict | None:
    manifest_path = cfg.manifest_file
    if not manifest_path.exists() or not cfg.raw_file.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != sha256_file(cfg.raw_file):
        return None
    if manifest.get("rows") != _count_rows(cfg.raw_file.read_bytes()):
        return None
    return manifest


def fetch(cfg: Config, force: bool = False) -> dict:
    """Download the dataset unless a matching cached copy exists."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        cached = _cached_manifest(cfg)
        if cached is not None:
            cached["cache"] = "hit"
            return cached

    requested_url = cfg.data.source_url
    used_url = requested_url
    fallback_used = False
    try:
        payload = _extract_payload(used_url, _get(used_url, cfg), cfg.data.filename)
    except FetchError:
        used_url = cfg.data.fallback_url
        fallback_used = True
        payload = _extract_payload(used_url, _get(used_url, cfg), cfg.data.filename)

    rows = _count_rows(payload)
    if rows == 0:
        raise FetchError(f"{used_url}: payload has no rows")
    first_row, last_row = _first_last(payload)

    cfg.raw_file.write_bytes(payload)
    manifest = {
        "dataset": DATASET_NAME,
        "source_url": requested_url,
        "source_url_used": used_url,
        "fallback_used": fallback_used,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "rows": rows,
        "columns": len(payload.splitlines()[0].split()),
        "first_observation": first_row,
        "last_observation": last_row,
        "file": cfg.data.filename,
        "user_agent": cfg.data.user_agent,
    }
    _fetch_description(cfg, manifest)
    cfg.manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["cache"] = "miss"
    return manifest


def _fetch_description(cfg: Config, manifest: dict) -> None:
    """Best effort attribute description. Parsing never depends on it."""
    try:
        if not cfg.description_file.exists():
            payload = _get(cfg.data.description_url, cfg)
            cfg.description_file.write_bytes(payload)
        manifest["description_file"] = cfg.data.description_filename
        manifest["description_sha256"] = sha256_file(cfg.description_file)
    except (FetchError, OSError):
        manifest["description_file"] = None


def read_raw(path: str | Path) -> pd.DataFrame:
    """Parse the Statlog file into readable columns with explicit dtypes."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=list(COLUMNS), dtype=str)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise FetchError(f"{path}: missing columns {missing}")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").astype("Int64")
    return frame


def load_raw(cfg: Config) -> pd.DataFrame:
    if not cfg.raw_file.exists():
        raise FetchError(f"{cfg.raw_file} not found, run 'credit_risk fetch' first")
    return read_raw(cfg.raw_file)


def describe_frame(frame: pd.DataFrame) -> dict:
    """Small structural summary used by the CLI report."""
    target = frame[TARGET].dropna().astype(int)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "good": int((target == 1).sum()),
        "bad": int((target == 2).sum()),
        "bad_rate": float((target == 2).mean()) if len(target) else float("nan"),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Helper shared with tests: coerce numeric columns without raising."""
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

