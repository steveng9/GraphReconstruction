"""Black-box model inversion against a GNN (Fredrikson et al. 2015, adapted to graphs).

For a target i with known model output label y_i, the adversary scores each
candidate value v of a hidden input attribute h by the model's likelihood of y_i
when s_i = v, combined with a prior:

    score_i(v) = log E_ctx[ p_model(y_i | h_i = v, ctx) ] + log prior_i(v)

where ctx is every input the adversary does not know (other hidden attributes of
i, and all hidden attributes of i's neighbours, which also feed i's prediction
through message passing). ctx is marginalised by Monte Carlo: in each of R
rounds every unknown input of every target is drawn uniformly at random and the
whole target graph is queried once; the rounds in which h_i happened to equal v
estimate the expectation for (i, v). This costs R queries in total rather than
one per (node, value), and treats neighbour uncertainty honestly.

Priors: ``uniform`` (pure inversion), ``aux_marginal``, or ``imputation`` (the
posterior of the Imputation classifier on auxiliary data -- the classifier-first,
model-corrects-second composition, in the spirit of CoBP).
"""
from __future__ import annotations

import numpy as np

from graphrecon.attacks import register_attack
from graphrecon.attacks.base import NodePrediction, marginal, normalize_rows
from graphrecon.attacks.gnn.node.baselines import IMPUTATION_DEFAULTS, imputation_model
from graphrecon.data.graph import MISSING, AttributedGraph

MI_DEFAULTS = {"rounds": 200, "prior": "imputation", "imputation": {}}


@register_attack("gnn", "node", "ModelInversion", defaults=MI_DEFAULTS)
def model_inversion(release, view, rng, rounds, prior, imputation):
    """Monte-Carlo black-box inversion: log E[p(y | s=v)] + log prior(v)."""
    model = release.model
    label = model.label_attr
    if label not in view.task.qi:
        raise ValueError(f"ModelInversion needs the model's label {label!r} among the known attributes")
    n = view.num_targets
    y = view.graph.col(label)
    inputs = model.input_attrs
    schema = model.encoder.schema
    unknown = [a for a in inputs if a not in view.task.qi]
    invertible = [h for h in view.task.hidden if h in inputs]

    # Priors
    if prior == "uniform":
        logprior = {h: np.zeros((n, view.card(h))) for h in view.task.hidden}
    elif prior == "aux_marginal":
        aux = release.require_aux("ModelInversion(prior=aux_marginal)")
        logprior = {h: np.tile(np.log(marginal(aux.col(h), view.card(h), alpha=1.0)), (n, 1))
                    for h in view.task.hidden}
    elif prior == "imputation":
        aux = release.require_aux("ModelInversion(prior=imputation)")
        make = imputation_model({**IMPUTATION_DEFAULTS, **imputation}, int(rng.integers(2**31)))
        logprior = {h: np.log(make().fit(aux, view.task.qi, aux.col(h), view.card(h)).predict_proba(view.graph))
                    for h in view.task.hidden}
    else:
        raise ValueError(f"unknown prior {prior!r}")

    # Monte-Carlo likelihood of the known label under each candidate value
    S = {h: np.zeros((n, view.card(h))) for h in invertible}
    N = {h: np.zeros((n, view.card(h))) for h in invertible}
    known_cols = {a: view.graph.col(a) for a in inputs if a in view.task.qi}
    has_y = y != MISSING
    rows = np.arange(n)
    for _ in range(rounds):
        draw = {a: rng.integers(len(schema[a]), size=n) for a in unknown}
        cols = {**known_cols, **draw}
        q = AttributedGraph(n, view.graph.edges, np.stack([cols[a] for a in inputs], axis=1),
                            {a: schema[a] for a in inputs})
        lik = model.query_graph(q)[rows, np.maximum(y, 0)]
        for h in invertible:
            np.add.at(S[h], (rows[has_y], draw[h][has_y]), lik[has_y])
            np.add.at(N[h], (rows[has_y], draw[h][has_y]), 1.0)

    out, coverage = {}, {}
    for h in view.task.hidden:
        score = logprior[h].copy()
        if h in S:
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = S[h] / N[h]
            # unsampled (i, v): fall back to the node's average likelihood
            fill = np.nanmean(np.where(N[h] > 0, mean, np.nan), axis=1, keepdims=True)
            mean = np.where(N[h] > 0, mean, fill)
            loglik = np.log(np.clip(np.nan_to_num(mean, nan=1.0), 1e-12, None))
            score[has_y] += loglik[has_y]
            coverage[h] = float((N[h][has_y] > 0).mean())
        score -= score.max(axis=1, keepdims=True)
        out[h] = normalize_rows(np.exp(score))
    return NodePrediction(out, info={"coverage": coverage, "not_model_inputs":
                                     [h for h in view.task.hidden if h not in inputs]})
