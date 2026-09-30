"""Shared attack input / output types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphrecon.data.graph import MISSING, AttributedGraph
from graphrecon.releases.gnn.release import GNNRelease
from graphrecon.threat.views import NodeView


@dataclass
class GNNAccess:
    """What an adversary attacking a released GNN holds: the model (black-box
    queries and, if the threat model allows, white-box parameters) and optionally
    auxiliary population data (a disjoint graph with all attributes)."""
    model: GNNRelease
    aux: AttributedGraph | None = None

    def require_aux(self, attack: str) -> AttributedGraph:
        if self.aux is None:
            raise ValueError(f"{attack} needs auxiliary data (set adversary.aux: true)")
        return self.aux


@dataclass
class NodePrediction:
    """Per hidden attribute, an (n_targets, C) probability matrix over its categories.
    ``info`` carries optional diagnostics (stored with the run)."""
    probas: dict[str, np.ndarray]
    info: dict = field(default_factory=dict)

    def predictions(self, view: NodeView | None = None) -> dict[str, np.ndarray]:
        """Argmax predictions; values the adversary already knows are passed through."""
        out = {}
        for a, P in self.probas.items():
            pred = P.argmax(axis=1)
            if view is not None and a in view.known:
                k = view.known[a]
                pred = np.where(k != MISSING, k, pred)
            out[a] = pred
        return out


def normalize_rows(P: np.ndarray, eps: float = 0.0) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64) + eps
    s = P.sum(axis=1, keepdims=True)
    return np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[1]), where=s > 0)


def marginal(codes: np.ndarray, card: int, alpha: float = 0.0) -> np.ndarray:
    """Empirical distribution over 0..card-1, ignoring MISSING."""
    c = codes[codes != MISSING]
    p = np.bincount(c, minlength=card).astype(float) + alpha
    return p / p.sum() if p.sum() > 0 else np.full(card, 1.0 / card)
