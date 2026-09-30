"""PairClassifier: a link classifier trained on the synthetic graph.

The adversary rehearses its own situation on the synthetic graph: it hides the
same fraction of synthetic edges it cannot see among the targets, samples the
same number of synthetic non-edges, and learns edge | pair features. Features:

* per known attribute, the endpoints-match indicator;
* per known attribute, the log edge-propensity of the endpoints' value pair,
  log P(pair | edge) / P(pair | random pair), estimated on synthetic edges;
* the structural heuristics on the observed part of the graph.
"""
from __future__ import annotations

import numpy as np

from graphrecon.attacks import register_attack
from graphrecon.attacks.common.classifiers import CLASSIFIER_DEFAULTS, fit_predict_proba
from graphrecon.attacks.synth.edge.heuristics import attribute_matches, pair_heuristics
from graphrecon.threat.views import EdgeTask, make_edge_view

PAIR_DEFAULTS = {"classifier": "LightGBM", "classifier_params": {}, "use_structure": True,
                 "use_propensity": True, "smoothing": 1.0}


def _propensity_tables(syn, attrs, rng, smoothing, n_random=200_000):
    tables = {}
    e = syn.edges
    u, v = rng.integers(syn.num_nodes, size=(2, n_random))
    for a in attrs:
        c = syn.col(a) + 1
        C = syn.card(a) + 1
        def dist(x, y):
            M = np.full((C, C), smoothing)
            np.add.at(M, (np.minimum(x, y), np.maximum(x, y)), 1.0)
            return M / M.sum()
        tables[a] = np.log(dist(c[e[:, 0]], c[e[:, 1]])) - np.log(dist(c[u], c[v]))
    return tables


def _features(graph, pairs, tables, use_structure):
    parts = [attribute_matches(graph, pairs)]
    for a, T in tables.items():
        c = graph.col(a) + 1
        x, y = c[pairs[:, 0]], c[pairs[:, 1]]
        parts.append(T[np.minimum(x, y), np.maximum(x, y)][:, None])
    if use_structure:
        parts.append(np.stack(list(pair_heuristics(graph, pairs).values()), axis=1))
    return np.concatenate(parts, axis=1)


@register_attack("synth", "edge", "PairClassifier", defaults=PAIR_DEFAULTS)
def pair_classifier(release, view, rng, classifier, classifier_params, use_structure,
                    use_propensity, smoothing):
    """Edge classifier trained on a rehearsal of the attack on the synthetic graph."""
    syn = release.with_attrs(view.task.known_attrs)
    clf_params = {**CLASSIFIER_DEFAULTS[classifier], **classifier_params}
    tables = _propensity_tables(syn, syn.attr_names, rng, smoothing) if use_propensity else {}
    rehearsal, y = make_edge_view(syn, EdgeTask(view.task.known_attrs),
                                  {k: view.knowledge[k] for k in ("observe_edges_frac", "neg_ratio")},
                                  seed=int(rng.integers(2**31)))
    X_tr = _features(rehearsal.graph, rehearsal.pairs, tables, use_structure)
    X_te = _features(view.graph, view.pairs, tables, use_structure)
    return fit_predict_proba(classifier, clf_params, X_tr, y, X_te, 2, int(rng.integers(2**31)))[:, 1]
