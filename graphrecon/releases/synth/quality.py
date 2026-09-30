"""Fidelity summary of a synthetic graph relative to the real one (stored with each release)."""
from __future__ import annotations

import numpy as np

from graphrecon.data.graph import MISSING, AttributedGraph


def edge_homophily(g: AttributedGraph, attr: str) -> float:
    """Fraction of edges (both endpoints observed) whose endpoints share ``attr``."""
    c = g.col(attr)
    u, v = c[g.edges[:, 0]], c[g.edges[:, 1]]
    ok = (u != MISSING) & (v != MISSING)
    return float((u[ok] == v[ok]).mean()) if ok.any() else float("nan")


def _tvd(a: np.ndarray, b: np.ndarray, card: int) -> float:
    pa = np.bincount(a + 1, minlength=card + 1) / len(a)
    pb = np.bincount(b + 1, minlength=card + 1) / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def _clustering(g: AttributedGraph) -> float:
    A = g.adjacency()
    tri = (A @ A).multiply(A).sum() / 6.0
    d = g.degrees()
    wedges = (d * (d - 1) / 2).sum()
    return float(3 * tri / wedges) if wedges else 0.0


def quality_report(real: AttributedGraph, syn: AttributedGraph) -> dict:
    dr, ds = real.degrees(), syn.degrees()
    rep = {
        "n_nodes": [real.num_nodes, syn.num_nodes],
        "n_edges": [real.num_edges, syn.num_edges],
        "mean_degree": [float(dr.mean()), float(ds.mean())],
        "max_degree": [int(dr.max()), int(ds.max())],
        "transitivity": [_clustering(real), _clustering(syn)],
        "attributes": {},
    }
    for a in real.attr_names:
        rep["attributes"][a] = {
            "marginal_tvd": _tvd(real.col(a), syn.col(a), real.card(a)),
            "edge_homophily": [edge_homophily(real, a), edge_homophily(syn, a)],
        }
    return rep
