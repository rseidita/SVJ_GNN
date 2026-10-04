"""Disk cache for signal_contamination bucket aggregates (cutflows and yields)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from utils.InputJson import normalize_input_path

CACHE_VERSION = "contamination-v2"


def cache_directory(config_dir: str) -> str:
    path = os.path.join(os.path.abspath(config_dir), "cache_signal_contamination")
    os.makedirs(path, exist_ok=True)
    return path


def bucket_cache_key(
    bucket: str,
    year: str,
    role: str,
    files: list,
    cuts_fingerprint: str,
    no_trigger_sf: bool,
) -> str:
    digest = hashlib.sha1()
    digest.update(CACHE_VERSION.encode())
    digest.update(bucket.encode())
    digest.update(str(year).encode())
    digest.update(role.encode())
    digest.update(cuts_fingerprint.encode())
    digest.update(repr(bool(no_trigger_sf)).encode())
    digest.update(repr(sorted(normalize_input_path(p) for p in files)).encode())
    return digest.hexdigest()[:16]


def stats_to_payload(stats) -> dict:
    return {
        "n_pass": stats.n_pass,
        "w_pass": stats.w_pass,
        "n_pre": stats.n_pre,
        "n_post": stats.n_post,
        "w_pre": stats.w_pre,
        "w_post": stats.w_post,
        "cutflow": dict(stats.cutflow),
        "root_cutflow": dict(stats.root_cutflow),
        "weighted_cutflow": dict(stats.weighted_cutflow),
    }


def save_bucket_cache(config_dir: str, cache_key: str, stats) -> None:
    directory = Path(cache_directory(config_dir))
    path = directory / f"{cache_key}.json"
    tmp = directory / f"{cache_key}.tmp.json"
    with open(tmp, "w") as handle:
        json.dump(stats_to_payload(stats), handle, indent=2)
    os.replace(tmp, path)


def load_bucket_cache(config_dir: str, cache_key: str):
    path = Path(cache_directory(config_dir)) / f"{cache_key}.json"
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Warning: bucket cache read failed ({exc}), re-reading source")
        return None
