"""Node models: fit hidden | known on one graph, predict on another.

``NodeModel`` is the shared interface behind the classifier attacks and the
unary potentials of CoBP-G:

    model.fit(train_graph, qi, labels, card, mask)   # labels: (n,) codes, mask: which to learn from
    model.predict_proba(graph) -> (n, card)
"""
from __future__ import annotations

import numpy as np

from graphrecon.attacks.common.classifiers import CLASSIFIER_DEFAULTS, fit_predict_proba
from graphrecon.data.features import node_features
from graphrecon.data.graph import MISSING

FEATURE_DEFAULTS = {"features": ["qi", "structure", "neighbor_qi"], "hops": 2}


class FeatureClassifierModel:
    """Tabularise each node (see data/features.py), then a standard classifier."""

    def __init__(self, classifier: str, params: dict, seed: int):
        self.classifier = classifier
        self.clf_params = {k: params[k] for k in CLASSIFIER_DEFAULTS[classifier]}
        self.features, self.hops, self.seed = params["features"], params["hops"], seed

    def fit(self, graph, qi, labels, card, mask=None):
        mask = (labels != MISSING) if mask is None else (mask & (labels != MISSING))
        self.qi, self.card = qi, card
        self.X = node_features(graph, qi, self.features, self.hops)[mask]
        self.y = labels[mask]
        return self

    def predict_proba(self, graph):
        X = node_features(graph, self.qi, self.features, self.hops)
        return fit_predict_proba(self.classifier, self.clf_params, self.X, self.y, X, self.card, self.seed)


def cross_fit_proba(make_model, graph, qi, labels, card, folds: int, rng) -> np.ndarray:
    """Out-of-fold posteriors on ``graph`` itself: each node is predicted by a model that
    never saw its label (features still use the whole graph, as at attack time)."""
    fold = rng.integers(folds, size=graph.num_nodes)
    P = np.full((graph.num_nodes, card), 1.0 / card)
    for k in range(folds):
        test = fold == k
        m = make_model().fit(graph, qi, labels, card, mask=~test)
        P[test] = m.predict_proba(graph)[test]
    return P
