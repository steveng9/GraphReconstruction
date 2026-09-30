"""Threat model: what the adversary knows about the targets, and what it must infer.

The target graph is the private graph whose nodes are attacked (the SDG / GNN
training graph for member targets, or a disjoint holdout graph for the
memorization test). A *view* is the adversary's knowledge of it; the *truth* is
kept apart and only ever reaches the scorer.

Node scope (attribute reconstruction)
    The adversary knows the ``qi`` attributes of every target, a fraction
    ``observe_edges_frac`` of the target graph's edges, and optionally the true
    hidden values of a fraction ``known_label_frac`` of targets (e.g. users with
    public profiles). It must infer each ``hidden`` attribute of every target.

Edge scope (link reconstruction)
    The adversary knows ``known_attrs`` of every target and a fraction
    ``observe_edges_frac`` of the edges. It must score candidate pairs: every
    unobserved true edge plus ``neg_ratio`` times as many uniformly sampled non-edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphrecon.data.graph import MISSING, AttributedGraph

NODE_KNOWLEDGE_DEFAULTS = {"observe_edges_frac": 1.0, "known_label_frac": 0.0}
EDGE_KNOWLEDGE_DEFAULTS = {"observe_edges_frac": 0.0, "neg_ratio": 1.0}


def _resolve(defaults: dict, params: dict | None) -> dict:
    params = dict(params or {})
    unknown = set(params) - set(defaults)
    if unknown:
        raise KeyError(f"unknown knowledge params {sorted(unknown)}; valid: {sorted(defaults)}")
    return {**defaults, **params}


def _observe_edges(graph: AttributedGraph, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split edges into (observed, unobserved)."""
    m = graph.num_edges
    k = int(round(frac * m))
    perm = rng.permutation(m)
    return graph.edges[np.sort(perm[:k])], graph.edges[np.sort(perm[k:])]


# ---------------------------------------------------------------------------
# Node scope
# ---------------------------------------------------------------------------

@dataclass
class NodeTask:
    qi: list[str]
    hidden: list[str]

    def __post_init__(self):
        overlap = set(self.qi) & set(self.hidden)
        if overlap:
            raise ValueError(f"attributes both known and hidden: {overlap}")


@dataclass
class NodeView:
    graph: AttributedGraph                 # targets with only the qi attributes; observed edges
    task: NodeTask
    hidden_schema: dict[str, list[str]]    # categories of each hidden attribute
    known: dict[str, np.ndarray] = field(default_factory=dict)  # attr -> (n,) codes, MISSING = unknown
    knowledge: dict = field(default_factory=dict)

    @property
    def num_targets(self) -> int:
        return self.graph.num_nodes

    def card(self, attr: str) -> int:
        return len(self.hidden_schema[attr])


def make_node_view(target: AttributedGraph, task: NodeTask, knowledge: dict | None, seed: int):
    """Returns (view, truth) where truth maps hidden attr -> (n,) true codes."""
    k = _resolve(NODE_KNOWLEDGE_DEFAULTS, knowledge)
    rng = np.random.default_rng(seed)
    observed, _ = _observe_edges(target, k["observe_edges_frac"], rng)
    g = target.with_attrs(task.qi).with_edges(observed)
    truth = {h: target.col(h).copy() for h in task.hidden}
    known = {}
    if k["known_label_frac"] > 0:
        for h in task.hidden:
            mask = rng.random(target.num_nodes) < k["known_label_frac"]
            known[h] = np.where(mask, truth[h], MISSING)
    view = NodeView(g, task, {h: target.schema[h] for h in task.hidden}, known, k)
    return view, truth


# ---------------------------------------------------------------------------
# Edge scope
# ---------------------------------------------------------------------------

@dataclass
class EdgeTask:
    known_attrs: list[str]


@dataclass
class EdgeView:
    graph: AttributedGraph         # targets with known_attrs; observed edges only
    pairs: np.ndarray              # (P, 2) candidate pairs to score
    task: EdgeTask
    knowledge: dict = field(default_factory=dict)


def _sample_non_edges(graph: AttributedGraph, k: int, rng) -> np.ndarray:
    n = graph.num_nodes
    existing = set(map(tuple, graph.edges.tolist()))
    out = set()
    while len(out) < k:
        u = rng.integers(n, size=2 * (k - len(out)) + 16)
        v = rng.integers(n, size=len(u))
        for a, b in zip(np.minimum(u, v), np.maximum(u, v)):
            if a != b and (a, b) not in existing:
                out.add((int(a), int(b)))
                if len(out) == k:
                    break
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def make_edge_view(target: AttributedGraph, task: EdgeTask, knowledge: dict | None, seed: int):
    """Returns (view, truth) where truth is a (P,) 0/1 array aligned with view.pairs."""
    k = _resolve(EDGE_KNOWLEDGE_DEFAULTS, knowledge)
    rng = np.random.default_rng(seed)
    observed, hidden = _observe_edges(target, k["observe_edges_frac"], rng)
    neg = _sample_non_edges(target, int(round(k["neg_ratio"] * len(hidden))), rng)
    pairs = np.concatenate([hidden, neg])
    truth = np.concatenate([np.ones(len(hidden)), np.zeros(len(neg))]).astype(np.int64)
    order = rng.permutation(len(pairs))
    g = target.with_attrs(task.known_attrs).with_edges(observed)
    return EdgeView(g, pairs[order], task, k), truth[order]
