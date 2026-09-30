"""Node-attribute reconstruction scores.

R_adv (Golob et al., PoPETs 2027): rarity-weighted accuracy on a 0-100 scale. A
correct guess of value v earns weight n / count(v) over the scored targets, so a
constant (mode) guess scores exactly 100 / C and a perfect attack scores 100.

Targets whose true value is missing, or whose value the adversary was given
(``known`` labels), are excluded from scoring.
"""
from __future__ import annotations

import numpy as np

from graphrecon.data.graph import MISSING


def _scored_mask(truth: np.ndarray, known: np.ndarray | None) -> np.ndarray:
    mask = truth != MISSING
    if known is not None:
        mask &= known == MISSING
    return mask


def rarity_weights(truth: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(truth, return_inverse=True, return_counts=True)
    return len(truth) / counts[inv]


def rarity_weighted_score(truth: np.ndarray, pred: np.ndarray) -> float:
    if len(truth) == 0:
        return float("nan")
    w = rarity_weights(truth)
    return float(100.0 * (w * (truth == pred)).sum() / w.sum())


def score_nodes(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray],
                known: dict[str, np.ndarray] | None = None):
    """Returns ({(attr, metric): value}, row_scores (n,) rarity-weighted mean per target).

    Metrics per attribute: ``r_adv``, ``accuracy``, ``n_scored``; plus
    ``("_mean", "r_adv")`` and ``("_mean", "accuracy")`` averaged over attributes.
    """
    known = known or {}
    scores, rows, row_w = {}, None, None
    for a, t in truth.items():
        m = _scored_mask(t, known.get(a))
        t_s, p_s = t[m], preds[a][m]
        scores[(a, "r_adv")] = rarity_weighted_score(t_s, p_s)
        scores[(a, "accuracy")] = float(100.0 * (t_s == p_s).mean()) if m.any() else float("nan")
        scores[(a, "n_scored")] = float(m.sum())
        # Row-level: rarity-weighted correctness per target, normalised per attribute
        if rows is None:
            rows = np.zeros(len(t)); row_w = np.zeros(len(t))
        if m.any():
            w = rarity_weights(t_s)
            rows[m] += w * (t_s == p_s) / w.mean()
            row_w[m] += 1
    for metric in ("r_adv", "accuracy"):
        vals = [scores[(a, metric)] for a in truth]
        scores[("_mean", metric)] = float(np.nanmean(vals)) if vals else float("nan")
    row_scores = np.divide(rows, row_w, out=np.full_like(rows, np.nan), where=row_w > 0) if rows is not None else None
    return scores, row_scores
