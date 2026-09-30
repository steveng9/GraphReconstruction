"""Attack registry, keyed by (release kind, scope, name).

    attacks/<release kind>/<scope>/     release kind in {synth, gnn}; scope in {node, edge, graph}

Signatures (``release`` is the synthetic ``AttributedGraph`` for ``synth`` and a
``GNNAccess`` for ``gnn``; ``rng`` is a numpy Generator):

    node:  fn(release, view: NodeView, rng, **params) -> NodePrediction
    edge:  fn(release, view: EdgeView, rng, **params) -> np.ndarray of pair scores

Register with ``@register_attack(kinds, scope, name, defaults)``; an attack valid
for several release kinds (e.g. a structural baseline) registers once for all.
"""
from __future__ import annotations

from graphrecon.registry import Registry

ATTACKS = Registry("attack")
RELEASE_KINDS = ("synth", "gnn")
SCOPES = ("node", "edge", "graph")


def register_attack(kinds, scope: str, name: str, defaults: dict | None = None, **meta):
    kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    assert scope in SCOPES and all(k in RELEASE_KINDS for k in kinds)

    def deco(fn):
        for k in kinds:
            ATTACKS.register((k, scope, name), defaults, **meta)(fn)
        return fn
    return deco


def get_attack(kind: str, scope: str, name: str):
    return ATTACKS.get((kind, scope, name))


def list_attacks(kind: str | None = None, scope: str | None = None):
    return sorted(k for k in ATTACKS.keys() if (kind in (None, k[0])) and (scope in (None, k[1])))


# Register everything (import for side effects).
from graphrecon.attacks.synth.node import baselines, classifiers, gnn_classifier, cobp  # noqa: E402,F401
from graphrecon.attacks.synth.edge import heuristics, pair_classifier  # noqa: E402,F401
from graphrecon.attacks.gnn.node import baselines as _gb, model_inversion  # noqa: E402,F401
from graphrecon.attacks.gnn.edge import link_stealing  # noqa: E402,F401
