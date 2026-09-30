"""Public input specification of a released GNN: which node attributes it consumes
and how they are one-hot encoded. The adversary knows this spec (it is the API)."""
from __future__ import annotations

import numpy as np

from graphrecon.data.graph import MISSING, AttributedGraph


class InputEncoder:
    """One-hot (plus missing indicator) of ``input_attrs``, concatenated with dense ``x`` if used."""

    def __init__(self, schema: dict[str, list[str]], input_attrs: list[str], use_x: bool, x_dim: int = 0):
        self.schema = {a: list(schema[a]) for a in input_attrs}
        self.input_attrs = list(input_attrs)
        self.use_x, self.x_dim = use_x, x_dim if use_x else 0
        self.offsets, off = {}, 0
        for a in input_attrs:
            self.offsets[a] = off
            off += len(schema[a]) + 1
        self.dim = off + self.x_dim

    def encode(self, graph: AttributedGraph, soft: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Encode ``graph``'s attributes. ``soft[attr]`` (n, C) overrides an attribute with a
        probability vector (used by white-box / relaxed inversion)."""
        soft = soft or {}
        X = np.zeros((graph.num_nodes, self.dim), dtype=np.float32)
        for a in self.input_attrs:
            o, C = self.offsets[a], len(self.schema[a])
            if a in soft:
                X[:, o:o + C] = soft[a]
                continue
            v = graph.col(a)
            obs = v != MISSING
            X[np.flatnonzero(obs), o + v[obs]] = 1.0
            X[~obs, o + C] = 1.0
        if self.use_x:
            X[:, self.dim - self.x_dim:] = graph.x
        return X

    def to_dict(self) -> dict:
        return {"schema": self.schema, "input_attrs": self.input_attrs, "use_x": self.use_x,
                "x_dim": self.x_dim}

    @classmethod
    def from_dict(cls, d: dict) -> "InputEncoder":
        return cls(d["schema"], d["input_attrs"], d["use_x"], d["x_dim"])
