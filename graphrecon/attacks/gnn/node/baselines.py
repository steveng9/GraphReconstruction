"""Model-free references for the GNN pipeline: what the adversary can infer from
auxiliary population data alone. ``Imputation`` is the key control: an attack on
the model is only evidence of *model* leakage to the extent it beats imputation
(Jayaraman & Evans, "Are attribute inference attacks just imputation?", CCS 2022)."""
from __future__ import annotations

import numpy as np

from graphrecon.attacks import register_attack
from graphrecon.attacks.base import NodePrediction, marginal
from graphrecon.attacks.common.classifiers import CLASSIFIER_DEFAULTS
from graphrecon.attacks.synth.node.classifiers import run_node_model
from graphrecon.attacks.synth.node.models import FEATURE_DEFAULTS, FeatureClassifierModel

IMPUTATION_DEFAULTS = {"classifier": "RandomForest", "classifier_params": {}, **FEATURE_DEFAULTS}


def imputation_model(params: dict, seed: int):
    clf = params["classifier"]
    p = {**CLASSIFIER_DEFAULTS[clf], **params["classifier_params"],
         "features": params["features"], "hops": params["hops"]}
    return lambda: FeatureClassifierModel(clf, p, seed)


@register_attack("gnn", "node", "Mode", defaults={})
def mode(release, view, rng):
    """Most frequent value in the auxiliary data."""
    aux = release.require_aux("Mode")
    out = {}
    for h in view.task.hidden:
        p = marginal(aux.col(h), view.card(h))
        P = np.full((view.num_targets, len(p)), 1e-6)
        P[:, p.argmax()] = 1.0
        out[h] = P / P.sum(1, keepdims=True)
    return NodePrediction(out)


@register_attack("gnn", "node", "Imputation", defaults=IMPUTATION_DEFAULTS)
def imputation(release, view, rng, **params):
    """Classifier trained on auxiliary data (hidden | QI + structure); never queries the model."""
    aux = release.require_aux("Imputation")
    return run_node_model(imputation_model(params, int(rng.integers(2**31))), aux, view)
