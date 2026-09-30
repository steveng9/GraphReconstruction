"""Synthetic attributed-graph generators.

Identity  -- the real graph with shuffled node ids ("naive anonymisation"); the
             reference ceiling, analogous to the de-identification releases of the
             tabular SoK.
ChungLu   -- attributes from a tabular model, structure from Chung-Lu on the real
             degree sequence, *independent* of attributes: the null release in which
             topology carries no attribute information.
AGM       -- Attributed Graph Model (Pfeiffer et al., WWW 2014): Chung-Lu proposals
             accepted with probability proportional to how over-represented the
             endpoints' attribute pair is among real edges. Factorised over
             attributes, with iterative refinement of the acceptance ratios.
DCSBM     -- degree-corrected SBM over Louvain communities of the real graph;
             attributes sampled jointly with the community, so the coupling
             between structure and attributes is mediated by communities.

Where degrees must stay correlated with attributes (AGM, DCSBM), a degree-bin
column is modelled jointly with the attributes and each synthetic node draws a
real degree from its sampled bin as its Chung-Lu weight.
"""
from __future__ import annotations

import numpy as np

from graphrecon.data.graph import MISSING, AttributedGraph
from graphrecon.releases.synth import GENERATORS
from graphrecon.releases.synth.attr_models import AttributeModel
from graphrecon.releases.synth.structure import chung_lu, dcsbm


def _degree_bins(deg: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(deg, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(qs, deg, side="right")


def _weights_from_bins(bins_syn, bins_real, deg_real, rng) -> np.ndarray:
    w = np.empty(len(bins_syn), dtype=float)
    for b in np.unique(bins_syn):
        pool = deg_real[bins_real == b]
        if len(pool) == 0:
            pool = deg_real
        idx = np.flatnonzero(bins_syn == b)
        w[idx] = rng.choice(pool, size=len(idx))
    return np.maximum(w, 0.5)


def _sample_rows(graph, extra_cols: list[np.ndarray], extra_cards: list[int], n, attr_model, alpha, rng):
    """Fit the attribute model on attrs (+ extra latent columns) and sample n rows."""
    codes = np.concatenate([graph.attrs] + [c[:, None] for c in extra_cols], axis=1)
    cards = [graph.card(a) for a in graph.attr_names] + extra_cards
    rows = AttributeModel(attr_model, alpha).fit(codes, cards).sample(n, rng)
    k = len(graph.attr_names)
    return rows[:, :k], [rows[:, k + i] for i in range(len(extra_cols))]


@GENERATORS.register("Identity", defaults={})
def identity(graph: AttributedGraph, rng) -> AttributedGraph:
    """Real graph with node ids shuffled (naive anonymisation)."""
    perm = rng.permutation(graph.num_nodes)
    inv = np.argsort(perm)
    return AttributedGraph(graph.num_nodes, inv[graph.edges], graph.attrs[perm], dict(graph.schema),
                           None if graph.x is None else graph.x[perm], name="Identity")


@GENERATORS.register("ChungLu", defaults={"attr_model": "chowliu", "alpha": 0.1})
def chung_lu_null(graph: AttributedGraph, rng, attr_model, alpha) -> AttributedGraph:
    """Attributes from a tabular model; Chung-Lu structure independent of attributes."""
    n = graph.num_nodes
    attrs, _ = _sample_rows(graph, [], [], n, attr_model, alpha, rng)
    w = rng.permutation(graph.degrees()).astype(float) + 0.5
    edges = chung_lu(w, graph.num_edges, rng)
    return AttributedGraph(n, edges, attrs, dict(graph.schema), name="ChungLu")


@GENERATORS.register("AGM", defaults={
    "attr_model": "chowliu", "alpha": 0.1,
    "edge_attrs": None,        # attributes whose edge correlations are preserved; None = all
    "degree_bins": 10,
    "refine_iters": 2,         # rounds of re-estimating acceptance ratios from the generated graph
    "smoothing": 1.0,          # pseudo-count on attribute-pair edge counts
    "accept_quantile": 0.99,   # acceptance = min(1, r / q-quantile of r over proposals)
})
def agm(graph: AttributedGraph, rng, attr_model, alpha, edge_attrs, degree_bins, refine_iters,
        smoothing, accept_quantile) -> AttributedGraph:
    """Attributed Graph Model: Chung-Lu proposals with attribute-pair acceptance."""
    n, m = graph.num_nodes, graph.num_edges
    deg = graph.degrees()
    bins = _degree_bins(deg, degree_bins)
    attrs, (bins_syn,) = _sample_rows(graph, [bins], [degree_bins], n, attr_model, alpha, rng)
    w = _weights_from_bins(bins_syn, bins, deg, rng)
    names = edge_attrs or graph.attr_names

    def pair_dist(codes: np.ndarray, card: int, edges: np.ndarray) -> np.ndarray:
        c = codes + 1                                      # MISSING -> 0
        a, b = np.minimum(c[edges[:, 0]], c[edges[:, 1]]), np.maximum(c[edges[:, 0]], c[edges[:, 1]])
        M = np.full((card + 1, card + 1), smoothing)
        np.add.at(M, (a, b), 1.0)
        M = np.triu(M)
        return M / M.sum() + np.tril(np.ones_like(M), -1)   # lower triangle unused; 1 avoids 0/0

    target = {a: pair_dist(graph.col(a), graph.card(a), graph.edges) for a in names}
    idx = {a: graph.attr_names.index(a) for a in names}
    proposal = chung_lu(w, m, rng)
    ratio = {a: target[a] / pair_dist(attrs[:, idx[a]], graph.card(a), proposal) for a in names}

    def log_ratio(u, v):
        logr = np.zeros(len(u))
        for a in names:
            c = attrs[:, idx[a]] + 1
            lo, hi = np.minimum(c[u], c[v]), np.maximum(c[u], c[v])
            logr += np.log(ratio[a][lo, hi])
        return logr

    def accept(u, v):
        return np.minimum(1.0, np.exp(log_ratio(u, v) - logr_ref))

    for it in range(refine_iters + 1):
        # Normalise by a high quantile of the ratio over fresh proposals; the product of
        # per-attribute maxima would make acceptance vanishingly rare.
        pu = rng.choice(n, size=20000, p=w / w.sum())
        pv = rng.choice(n, size=20000, p=w / w.sum())
        logr_ref = np.quantile(log_ratio(pu, pv), accept_quantile)
        edges = chung_lu(w, m, rng, accept=accept, max_rounds=2000)
        if it < refine_iters:
            for a in names:
                ratio[a] = ratio[a] * target[a] / pair_dist(attrs[:, idx[a]], graph.card(a), edges)
    return AttributedGraph(n, edges, attrs, dict(graph.schema), name="AGM")


@GENERATORS.register("DCSBM", defaults={
    "attr_model": "chowliu", "alpha": 0.1,
    "resolution": 1.0,         # Louvain resolution (higher = more, smaller communities)
    "max_blocks": 30,          # communities beyond this (smallest first) are merged into one
    "degree_bins": 10,
})
def dcsbm_gen(graph: AttributedGraph, rng, attr_model, alpha, resolution, max_blocks,
              degree_bins) -> AttributedGraph:
    """Degree-corrected SBM over Louvain communities, attributes sampled with the community."""
    import networkx as nx

    n = graph.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(graph.edges.tolist())
    comms = nx.community.louvain_communities(G, resolution=resolution, seed=int(rng.integers(2**31)))
    comms = sorted(comms, key=len, reverse=True)
    block = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(comms):
        block[list(c)] = min(i, max_blocks - 1)
    K = int(block.max()) + 1

    deg = graph.degrees()
    bins = _degree_bins(deg, degree_bins)
    attrs, (block_syn, bins_syn) = _sample_rows(graph, [block, bins], [K, degree_bins], n,
                                                attr_model, alpha, rng)
    block_syn = np.where(block_syn == MISSING, 0, block_syn)
    w = _weights_from_bins(bins_syn, bins, deg, rng)

    B = np.zeros((K, K), dtype=np.int64)
    a, b = np.minimum(block[graph.edges[:, 0]], block[graph.edges[:, 1]]), \
        np.maximum(block[graph.edges[:, 0]], block[graph.edges[:, 1]])
    np.add.at(B, (a, b), 1)
    edges = dcsbm(block_syn, w, B, rng)
    return AttributedGraph(n, edges, attrs, dict(graph.schema), name="DCSBM")
