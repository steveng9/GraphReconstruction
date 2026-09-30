"""Filesystem locations. Everything generated lives under WORKSPACE (gitignored).

Override the root with the GRAPHRECON_WORKSPACE environment variable.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("GRAPHRECON_WORKSPACE", REPO_ROOT / "workspace"))

RAW_DATA = WORKSPACE / "datasets" / "raw"
PROCESSED_DATA = WORKSPACE / "datasets" / "processed"
RELEASES = WORKSPACE / "releases"          # releases/<kind>/<dataset>/<split>/s<k>/<method>-<hash>/
ATTACK_OUTPUTS = WORKSPACE / "attack_outputs"  # per-run predictions, keyed by run_id
RESULTS_DB = Path(os.environ.get("GRAPHRECON_DB", WORKSPACE / "results.db"))
CONFIGS = REPO_ROOT / "configs"
