"""Classifier attacks: the graph analogue of the tabular SoK's "synthetic data as
training data" branch. One classifier per hidden attribute, trained on synthetic
nodes and applied to the targets. The feature blocks decide how much of the graph
the attack sees: ``[qi]`` alone is the purely tabular attack; adding ``structure``
and ``neighbor_qi`` lets it exploit topology.
"""
from __future__ import annotations

from graphrecon.attacks import register_attack
from graphrecon.attacks.base import NodePrediction
from graphrecon.attacks.common.classifiers import CLASSIFIER_DEFAULTS
from graphrecon.attacks.synth.node.models import FEATURE_DEFAULTS, FeatureClassifierModel


def run_node_model(make_model, release, view) -> NodePrediction:
    out = {}
    for h in view.task.hidden:
        m = make_model().fit(release, view.task.qi, release.col(h), view.card(h))
        out[h] = m.predict_proba(view.graph)
    return NodePrediction(out)


def _register(name):
    @register_attack("synth", "node", name, defaults={**FEATURE_DEFAULTS, **CLASSIFIER_DEFAULTS[name]})
    def attack(release, view, rng, **params):
        seed = int(rng.integers(2**31))
        return run_node_model(lambda: FeatureClassifierModel(name, params, seed), release, view)
    attack.__doc__ = f"{name} on tabularised node features, trained on the synthetic graph."
    return attack


for _name in CLASSIFIER_DEFAULTS:
    _register(_name)
