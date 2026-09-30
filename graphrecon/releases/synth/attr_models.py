"""Tabular models for the node-attribute table of a synthetic graph.

Each model fits on an (n, k) integer code matrix (MISSING = -1 is treated as one
more category, so missingness patterns are synthesised too) and samples new rows.

* ``empirical`` -- resample real rows with replacement (maximal memorisation);
* ``marginals`` -- independent per-column marginals (no dependence at all);
* ``chowliu``   -- Chow-Liu tree: maximum spanning tree of pairwise mutual
                   information, Laplace-smoothed conditionals, ancestral sampling.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

ATTR_MODELS = ("empirical", "marginals", "chowliu")


def _mutual_info(a: np.ndarray, b: np.ndarray, ca: int, cb: int) -> float:
    joint = np.zeros((ca, cb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


class AttributeModel:
    def __init__(self, kind: str = "chowliu", alpha: float = 0.1):
        if kind not in ATTR_MODELS:
            raise ValueError(f"unknown attribute model {kind!r}; valid: {ATTR_MODELS}")
        self.kind, self.alpha = kind, alpha

    def fit(self, codes: np.ndarray, cards: list[int]):
        X = np.asarray(codes, dtype=np.int64) + 1          # MISSING -> 0
        self.cards = [c + 1 for c in cards]
        self.X = X
        n, k = X.shape
        self.marginals = [np.bincount(X[:, j], minlength=self.cards[j]) + self.alpha for j in range(k)]
        if self.kind == "chowliu" and k > 1:
            W = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    W[i, j] = _mutual_info(X[:, i], X[:, j], self.cards[i], self.cards[j]) + 1e-9
            tree = minimum_spanning_tree(-W).toarray()
            tree = (tree != 0) | (tree.T != 0)
            order, pred = breadth_first_order(tree, 0, directed=False, return_predecessors=True)
            self.order, self.parent = order, pred
            self.cond = {}
            for j in order[1:]:
                p = pred[j]
                T = np.full((self.cards[p], self.cards[j]), self.alpha)
                np.add.at(T, (X[:, p], X[:, j]), 1.0)
                self.cond[j] = T / T.sum(1, keepdims=True)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.X.shape[1]
        if self.kind == "empirical":
            out = self.X[rng.integers(len(self.X), size=n)]
        elif self.kind == "marginals" or k == 1:
            out = np.stack([rng.choice(self.cards[j], size=n, p=m / m.sum())
                            for j, m in enumerate(self.marginals)], axis=1)
        else:
            out = np.zeros((n, k), dtype=np.int64)
            root = self.order[0]
            m = self.marginals[root]
            out[:, root] = rng.choice(self.cards[root], size=n, p=m / m.sum())
            for j in self.order[1:]:
                P = self.cond[j][out[:, self.parent[j]]]          # (n, C_j)
                u = rng.random((n, 1))
                out[:, j] = np.minimum((P.cumsum(1) < u).sum(1), self.cards[j] - 1)
        return out - 1                                             # back to MISSING = -1
