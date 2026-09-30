"""YAML config loading, canonical serialisation and content hashes."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from graphrecon.paths import REPO_ROOT


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def canonical_json(obj) -> str:
    """Deterministic JSON (sorted keys, no whitespace) used for hashing and DB storage."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def param_hash(obj, n: int = 10) -> str:
    return hashlib.sha1(canonical_json(obj).encode()).hexdigest()[:n]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() or "nogit") + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"
