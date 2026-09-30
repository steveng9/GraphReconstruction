"""Helpers for turning raw attribute columns into capped categorical codes."""
from __future__ import annotations

import numpy as np

from graphrecon.data.graph import MISSING

OTHER = "other"


def encode_column(raw: np.ndarray, missing_values=(), max_categories: int | None = None,
                  labels: dict | None = None):
    """Encode a raw column to codes 0..C-1 (MISSING for ``missing_values``).

    Categories are ordered by decreasing frequency. With ``max_categories=K`` the
    K-1 most frequent values are kept and the remainder pooled into ``"other"``
    (so the result has at most K categories).
    Returns (codes, category_labels).
    """
    raw = np.asarray(raw)
    miss = np.isin(raw, list(missing_values))
    vals, counts = np.unique(raw[~miss], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    vals = vals[order]
    if max_categories is not None and len(vals) > max_categories:
        keep, pooled = vals[: max_categories - 1], True
    else:
        keep, pooled = vals, False
    lookup = {v: i for i, v in enumerate(keep)}
    other = len(keep)
    codes = np.array([MISSING if m else lookup.get(v, other) for v, m in zip(raw, miss)], dtype=np.int64)
    names = [str((labels or {}).get(v, v)) for v in keep] + ([OTHER] if pooled else [])
    return codes, names
