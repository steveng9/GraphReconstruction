"""The attributed-graph container shared by every pipeline stage.

An ``AttributedGraph`` is an undirected simple graph with

* ``edges``  -- (E, 2) int64 array, each undirected edge stored once with u < v;
* ``attrs``  -- (n, k) int64 array of categorical node attributes, coded
                0..C_a-1, with ``MISSING`` (-1) for unobserved values;
* ``schema`` -- ordered {attribute name: list of category labels};
* ``x``      -- optional dense (n, d) float features (e.g. bag-of-words) that are
                not treated as reconstruction targets.

It is deliberately framework-agnostic (numpy / scipy); ``to_pyg`` converts when a
GNN needs it. Saved as ``graph.npz`` + ``schema.json`` (no pickle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

MISSING = -1


def canonical_edges(edges: np.ndarray, num_nodes: int | None = None) -> np.ndarray:
    """Drop self loops, orient u < v, deduplicate, sort."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    e = np.sort(e, axis=1)
    if len(e) == 0:
        return e
    if num_nodes is not None and e.max() >= num_nodes:
        raise ValueError("edge endpoint out of range")
    return np.unique(e, axis=0)


@dataclass
class AttributedGraph:
    num_nodes: int
    edges: np.ndarray
    attrs: np.ndarray
    schema: dict[str, list[str]]
    x: np.ndarray | None = None
    name: str = ""
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.edges = canonical_edges(self.edges, self.num_nodes)
        self.attrs = np.asarray(self.attrs, dtype=np.int64).reshape(self.num_nodes, len(self.schema))
        for j, (a, cats) in enumerate(self.schema.items()):
            col = self.attrs[:, j]
            if col.size and (col.max() >= len(cats) or col.min() < MISSING):
                raise ValueError(f"attribute {a!r} has codes outside [-1, {len(cats) - 1}]")

    # -- attribute access ---------------------------------------------------
    @property
    def attr_names(self) -> list[str]:
        return list(self.schema)

    def col(self, attr: str) -> np.ndarray:
        return self.attrs[:, self.attr_names.index(attr)]

    def card(self, attr: str) -> int:
        return len(self.schema[attr])

    def attrs_df(self, cols: list[str] | None = None) -> pd.DataFrame:
        cols = cols or self.attr_names
        return pd.DataFrame({c: self.col(c) for c in cols})

    # -- structure ----------------------------------------------------------
    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def adjacency(self) -> sp.csr_matrix:
        n, e = self.num_nodes, self.edges
        data = np.ones(2 * len(e), dtype=np.float32)
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def directed_edges(self) -> np.ndarray:
        """(2, 2E) array with both orientations, the PyG ``edge_index`` layout."""
        e = self.edges
        return np.concatenate([e.T, e[:, ::-1].T], axis=1)

    def degrees(self) -> np.ndarray:
        d = np.zeros(self.num_nodes, dtype=np.int64)
        np.add.at(d, self.edges.ravel(), 1)
        return d

    # -- derived graphs -----------------------------------------------------
    def subgraph(self, nodes: np.ndarray, name: str | None = None) -> "AttributedGraph":
        """Induced subgraph on ``nodes``, relabelled 0..len(nodes)-1 in the given order."""
        nodes = np.asarray(nodes, dtype=np.int64)
        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        e = remap[self.edges]
        e = e[(e >= 0).all(axis=1)]
        return AttributedGraph(len(nodes), e, self.attrs[nodes].copy(), dict(self.schema),
                               None if self.x is None else self.x[nodes].copy(),
                               name or self.name, dict(self.info))

    def with_edges(self, edges: np.ndarray) -> "AttributedGraph":
        return AttributedGraph(self.num_nodes, edges, self.attrs.copy(), dict(self.schema),
                               self.x, self.name, dict(self.info))

    def with_attrs(self, cols: list[str]) -> "AttributedGraph":
        """Same graph restricted to the attribute columns ``cols``."""
        idx = [self.attr_names.index(c) for c in cols]
        return AttributedGraph(self.num_nodes, self.edges, self.attrs[:, idx].copy(),
                               {c: self.schema[c] for c in cols}, self.x, self.name, dict(self.info))

    def to_pyg(self):
        import torch
        from torch_geometric.data import Data
        return Data(edge_index=torch.as_tensor(self.directed_edges(), dtype=torch.long),
                    num_nodes=self.num_nodes)

    # -- persistence --------------------------------------------------------
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"edges": self.edges, "attrs": self.attrs}
        if self.x is not None:
            arrays["x"] = self.x
        np.savez_compressed(d / "graph.npz", **arrays)
        with open(d / "schema.json", "w") as f:
            json.dump({"num_nodes": self.num_nodes, "name": self.name, "schema": self.schema,
                       "info": self.info}, f, indent=1)

    @classmethod
    def load(cls, directory) -> "AttributedGraph":
        d = Path(directory)
        with open(d / "schema.json") as f:
            meta = json.load(f)
        z = np.load(d / "graph.npz")
        return cls(meta["num_nodes"], z["edges"], z["attrs"], meta["schema"],
                   z["x"] if "x" in z else None, meta.get("name", ""), meta.get("info", {}))

    @staticmethod
    def exists(directory) -> bool:
        d = Path(directory)
        return (d / "graph.npz").exists() and (d / "schema.json").exists()

    def summary(self) -> str:
        deg = self.degrees()
        cards = ", ".join(f"{a}:{len(c)}" for a, c in self.schema.items())
        return (f"{self.name or 'graph'}: n={self.num_nodes} m={self.num_edges} "
                f"mean_deg={deg.mean():.1f} attrs=[{cards}]")
