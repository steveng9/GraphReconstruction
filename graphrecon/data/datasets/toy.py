"""Planted-partition toy graph with controllable homophily, for tests and sanity checks.

Each node has a latent block b. Attribute ``label`` equals b; ``qi`` is a noisy
copy of b (correct with probability ``qi_signal``); ``noise`` is independent.
Edges follow an SBM with in-block probability ``p_in`` and out-block ``p_out``.
So the QI gives partial information and the graph gives more: an attack that
propagates beliefs along edges should beat one that does not.
"""
from __future__ import annotations

import numpy as np

from graphrecon.data.datasets import DATASETS
from graphrecon.data.graph import AttributedGraph


@DATASETS.register("toy_sbm", defaults={
    "num_nodes": 1200, "num_blocks": 4, "p_in": 0.02, "p_out": 0.002,
    "qi_signal": 0.5, "seed": 0,
})
def load_toy_sbm(num_nodes, num_blocks, p_in, p_out, qi_signal, seed):
    """Planted-partition SBM with a homophilous hidden label and a noisy QI."""
    rng = np.random.default_rng(seed)
    b = rng.integers(num_blocks, size=num_nodes)
    qi = np.where(rng.random(num_nodes) < qi_signal, b, rng.integers(num_blocks, size=num_nodes))
    noise = rng.integers(3, size=num_nodes)
    iu, ju = np.triu_indices(num_nodes, k=1)
    p = np.where(b[iu] == b[ju], p_in, p_out)
    keep = rng.random(len(p)) < p
    edges = np.stack([iu[keep], ju[keep]], axis=1)
    schema = {"label": [f"b{i}" for i in range(num_blocks)],
              "qi": [f"q{i}" for i in range(num_blocks)],
              "noise": ["n0", "n1", "n2"]}
    return AttributedGraph(num_nodes, edges, np.stack([b, qi, noise], axis=1), schema, name="toy_sbm")
