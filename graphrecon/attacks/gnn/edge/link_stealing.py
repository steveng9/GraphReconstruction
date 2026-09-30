"""Link stealing, unsupervised variant (He et al., USENIX Security 2021, Attack-0).

A GNN's posterior for a node mixes in its neighbours' features, so connected
nodes receive similar posteriors. With black-box access to the served model
(evaluated on the private graph), the adversary scores a pair by the negated
distance between the two nodes' posteriors. Only meaningful for member targets
(the server graph is the training graph).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import distance as ssd

from graphrecon.attacks import register_attack

DISTANCES = ("correlation", "cosine", "euclidean", "chebyshev", "braycurtis", "canberra",
             "cityblock", "sqeuclidean", "jensenshannon")


@register_attack("gnn", "edge", "LinkStealing", defaults={"distance": "correlation"})
def link_stealing(release, view, rng, distance):
    """Negated distance between the served posteriors of the two endpoints."""
    if distance not in DISTANCES:
        raise ValueError(f"unknown distance {distance!r}; valid: {DISTANCES}")
    P = release.model.query_nodes()
    if len(P) != view.graph.num_nodes:
        raise ValueError("LinkStealing targets must be the model's training graph (target split 'train')")
    fn = getattr(ssd, distance)
    u, v = view.pairs[:, 0], view.pairs[:, 1]
    d = np.array([fn(P[a], P[b]) for a, b in zip(u, v)])
    return -np.nan_to_num(d, nan=np.nanmax(d) if np.isfinite(d).any() else 0.0)
