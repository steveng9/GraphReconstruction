"""Node feature builders shared by classifier-style attacks.

A feature set is a list of named blocks:

* ``qi``          -- one-hot of the known attributes (with a missing indicator);
* ``structure``   -- log degree, local clustering, mean neighbour log degree, log triangles;
* ``neighbor_qi`` -- mean of neighbours' ``qi`` one-hots, for hops 1..``hops``
                     (row-normalised adjacency powers, as in SIGN / feature propagation);
* ``x``           -- the graph's dense features, if any.

All blocks are computed identically on the synthetic graph (training) and on the
adversary's view of the targets (prediction), so the two are comparable.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from graphrecon.data.graph import MISSING, AttributedGraph

FEATURE_BLOCKS = ("qi", "structure", "neighbor_qi", "x")


def one_hot(graph: AttributedGraph, cols: list[str], missing_indicator: bool = True) -> np.ndarray:
    blocks = []
    for c in cols:
        v = graph.col(c)
        C = graph.card(c)
        oh = np.zeros((graph.num_nodes, C + int(missing_indicator)), dtype=np.float32)
        obs = v != MISSING
        oh[np.flatnonzero(obs), v[obs]] = 1.0
        if missing_indicator:
            oh[~obs, C] = 1.0
        blocks.append(oh)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((graph.num_nodes, 0), np.float32)


def row_normalized_adjacency(graph: AttributedGraph) -> sp.csr_matrix:
    A = graph.adjacency()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv) @ A


def structural_features(graph: AttributedGraph) -> np.ndarray:
    A = graph.adjacency()
    deg = np.asarray(A.sum(axis=1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2.0
    pairs = deg * (deg - 1) / 2.0
    clustering = np.divide(tri, pairs, out=np.zeros_like(tri), where=pairs > 0)
    logdeg = np.log1p(deg)
    nbr_logdeg = row_normalized_adjacency(graph) @ logdeg
    return np.stack([logdeg, clustering, nbr_logdeg, np.log1p(tri)], axis=1).astype(np.float32)


def neighbor_aggregates(graph: AttributedGraph, X: np.ndarray, hops: int) -> np.ndarray:
    P = row_normalized_adjacency(graph)
    out, cur = [], X
    for _ in range(hops):
        cur = P @ cur
        out.append(np.asarray(cur, dtype=np.float32))
    return np.concatenate(out, axis=1) if out else np.zeros((graph.num_nodes, 0), np.float32)


def node_features(graph: AttributedGraph, qi: list[str], blocks: list[str], hops: int = 1) -> np.ndarray:
    unknown = set(blocks) - set(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature blocks {unknown}; valid: {FEATURE_BLOCKS}")
    qi_oh = one_hot(graph, qi)
    parts = []
    if "qi" in blocks:
        parts.append(qi_oh)
    if "structure" in blocks:
        parts.append(structural_features(graph))
    if "neighbor_qi" in blocks:
        parts.append(neighbor_aggregates(graph, qi_oh, hops))
    if "x" in blocks and graph.x is not None:
        parts.append(graph.x.astype(np.float32))
    if not parts:
        return np.zeros((graph.num_nodes, 1), np.float32)
    return np.concatenate(parts, axis=1)
