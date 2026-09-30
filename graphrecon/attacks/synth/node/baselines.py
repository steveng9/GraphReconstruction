"""Reference points (not attacks): they calibrate what an R_adv value means."""
from __future__ import annotations

import numpy as np

from graphrecon.attacks import register_attack
from graphrecon.attacks.base import NodePrediction, marginal
from graphrecon.data.graph import MISSING


@register_attack("synth", "node", "Mode", defaults={})
def mode(release, view, rng):
    """Predict each hidden attribute's most frequent synthetic value."""
    out = {}
    for h in view.task.hidden:
        p = marginal(release.col(h), view.card(h))
        P = np.full((view.num_targets, len(p)), 1e-6)
        P[:, p.argmax()] = 1.0
        out[h] = P / P.sum(1, keepdims=True)
    return NodePrediction(out)


@register_attack("synth", "node", "Random", defaults={})
def random_guess(release, view, rng):
    """Draw each prediction from the synthetic marginal."""
    out = {}
    for h in view.task.hidden:
        p = marginal(release.col(h), view.card(h))
        draws = rng.choice(len(p), size=view.num_targets, p=p)
        P = np.full((view.num_targets, len(p)), 1e-6)
        P[np.arange(view.num_targets), draws] = 1.0
        out[h] = P / P.sum(1, keepdims=True)
    return NodePrediction(out)


@register_attack("synth", "node", "NeighborVote", defaults={"prior_weight": 1.0})
def neighbor_vote(release, view, rng, prior_weight):
    """Weighted-vote relational neighbour (wvRN, Macskassy & Provost 2007): the distribution
    of *known* hidden values among a target's observed neighbours, smoothed toward the
    synthetic marginal. Needs ``known_label_frac > 0`` to be more than the marginal."""
    A = view.graph.adjacency()
    out = {}
    for h in view.task.hidden:
        C = view.card(h)
        prior = marginal(release.col(h), C)
        K = np.zeros((view.num_targets, C))
        if h in view.known:
            k = view.known[h]
            m = k != MISSING
            K[np.flatnonzero(m), k[m]] = 1.0
        counts = A @ K
        out[h] = (counts + prior_weight * prior) / (counts.sum(1, keepdims=True) + prior_weight)
    return NodePrediction(out)
