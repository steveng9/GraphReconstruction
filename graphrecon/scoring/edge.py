"""Link reconstruction scores over the candidate pairs of an EdgeView."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def score_edges(truth: np.ndarray, scores: np.ndarray) -> dict:
    """AUC, average precision, and precision among the top-|positives| pairs."""
    out = {("edges", "n_pairs"): float(len(truth)), ("edges", "n_pos"): float(truth.sum())}
    if truth.min() == truth.max():
        return out
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    s = np.where(np.isinf(s), np.nanmin(s[np.isfinite(s)]) - 1 if np.isfinite(s).any() else 0.0, s)
    out[("edges", "auc")] = float(roc_auc_score(truth, s))
    out[("edges", "ap")] = float(average_precision_score(truth, s))
    k = int(truth.sum())
    top = np.argsort(-s, kind="stable")[:k]
    out[("edges", "precision_at_k")] = float(truth[top].mean())
    return out
