"""Disjoint node samples: the graph analogue of disjoint training samples.

The node set is randomly partitioned into ``n_parts`` equal parts and each part
induces a subgraph. For sample k:

* ``train``   -- part k: the private graph the defender releases from (SDG input
                 or GNN training graph); its nodes are the default targets;
* ``holdout`` -- part k+1: same-distribution non-members, for the memorization
                 test (reconstruction of non-training targets);
* ``aux``     -- part k+2: auxiliary population data some threat models grant
                 the adversary (e.g. to fit priors when no synthetic data exists).

Induced subgraphs of a random partition keep each part distributionally alike,
at the cost of thinning degrees by roughly 1/n_parts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphrecon.data.graph import AttributedGraph

SPLIT_DEFAULTS = {"n_parts": 4, "seed": 0}


@dataclass
class Split:
    sample: int
    parts: dict[str, np.ndarray]           # role -> node ids in the full graph
    graphs: dict[str, AttributedGraph]     # role -> induced subgraph

    def __getitem__(self, role) -> AttributedGraph:
        return self.graphs[role]


def partition_nodes(num_nodes: int, n_parts: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(num_nodes)
    return [np.sort(p) for p in np.array_split(perm, n_parts)]


def make_split(graph: AttributedGraph, sample: int, n_parts: int = 4, seed: int = 0) -> Split:
    if n_parts < 3:
        raise ValueError("n_parts must be >= 3 (train, holdout, aux)")
    if not 0 <= sample < n_parts:
        raise ValueError(f"sample {sample} out of range for n_parts={n_parts}")
    parts = partition_nodes(graph.num_nodes, n_parts, seed)
    roles = {"train": parts[sample], "holdout": parts[(sample + 1) % n_parts],
             "aux": parts[(sample + 2) % n_parts]}
    graphs = {r: graph.subgraph(ids, name=f"{graph.name}/s{sample}/{r}") for r, ids in roles.items()}
    return Split(sample, roles, graphs)
