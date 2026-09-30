"""Release-independent link baselines (registered for every release kind).

Structural heuristics use only the adversary's *observed* target edges (so they
are uninformative when ``observe_edges_frac = 0``); AttributeSimilarity uses only
the known attributes. They calibrate how much a release adds beyond what the
adversary already knew.
"""
from __future__ import annotations

import numpy as np

from graphrecon.attacks import RELEASE_KINDS, register_attack
from graphrecon.data.graph import MISSING


def _common_neighbors(A, pairs):
    return np.asarray(A[pairs[:, 0]].multiply(A[pairs[:, 1]]).sum(axis=1)).ravel()


def pair_heuristics(graph, pairs) -> dict[str, np.ndarray]:
    A = graph.adjacency().tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    cn = _common_neighbors(A, pairs)
    union = deg[pairs[:, 0]] + deg[pairs[:, 1]] - cn
    inv_log = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    aa = np.asarray(A[pairs[:, 0]].multiply(A[pairs[:, 1]]) @ inv_log).ravel()
    return {
        "CommonNeighbors": cn,
        "Jaccard": np.divide(cn, union, out=np.zeros_like(cn), where=union > 0),
        "AdamicAdar": aa,
        "PreferentialAttachment": np.log1p(deg[pairs[:, 0]] * deg[pairs[:, 1]]),
    }


def attribute_matches(graph, pairs) -> np.ndarray:
    """(P, k) indicator that both endpoints share each known attribute (missing = no match)."""
    cols = []
    for a in graph.attr_names:
        c = graph.col(a)
        u, v = c[pairs[:, 0]], c[pairs[:, 1]]
        cols.append(((u == v) & (u != MISSING)).astype(np.float32))
    return np.stack(cols, axis=1) if cols else np.zeros((len(pairs), 0), np.float32)


@register_attack(RELEASE_KINDS, "edge", "Random", defaults={})
def random_scores(release, view, rng):
    """Uniform random scores (AUC 0.5 reference)."""
    return rng.random(len(view.pairs))


def _heuristic(name):
    @register_attack(RELEASE_KINDS, "edge", name, defaults={})
    def attack(release, view, rng):
        return pair_heuristics(view.graph, view.pairs)[name]
    attack.__doc__ = f"{name} on the adversary's observed target edges."


for _n in ("CommonNeighbors", "Jaccard", "AdamicAdar", "PreferentialAttachment"):
    _heuristic(_n)


@register_attack(RELEASE_KINDS, "edge", "AttributeSimilarity", defaults={})
def attribute_similarity(release, view, rng):
    """Number of known attributes the two endpoints share."""
    return attribute_matches(view.graph, view.pairs).sum(axis=1) + 1e-3 * rng.random(len(view.pairs))
