"""CoBP-G: classifier posteriors corrected by belief propagation over the graph.

The graph analogue of CoBP-RA (Golob et al., PoPETs 2027). CoBP-RA couples the
hidden attributes *within a row* with BP; with a graph, the natural coupling is
also *across edges*: friends share dorms, years, majors. CoBP-G builds one
pairwise MRF over all (target node, hidden attribute) variables:

    unary       log p_clf(h_i = v | features_i)   from any node classifier trained on
                                                  the synthetic graph (default RandomForest)
    graph factor  psi_h(h_i, h_j)   for every observed target edge (i, j), each hidden h
    row factor    psi_hg(h_i, g_i)  for hidden pairs (h, g) on a Chow-Liu tree of the
                                    synthetic hidden columns (as in CoBP-RA)

and runs loopy sum-product BP (attacks/common/bp.py).

Potentials are estimated on the synthetic graph. ``pmi`` uses the plain log-PMI
of the co-occurrence table over synthetic edges (or rows). ``residual`` (default)
corrects for double counting, the problem CoBP-RA solved with neighbour-local
PMI tables: the classifier already explains part of the co-occurrence (e.g. from
QI homophily), so the potential should carry only what is left over:

    psi(v, w) = log (O[v, w] + alpha) / (E[v, w] + alpha)
    O = observed co-occurrence counts over synthetic pairs
    E = sum over the same pairs of q_i(v) q_j(w), with q the classifier's
        *out-of-fold* posteriors on the synthetic graph.

If the classifier already predicts the co-occurrences, psi ~ 0 and BP leaves it
alone. For a single edge this is exactly the potential under which the MRF's
pair marginal matches O.

Self-calibration (``strength: auto``): the adversary holds a fully labelled
synthetic graph, so it can rehearse the whole attack on it -- out-of-fold unaries,
the same observed-edge fraction and anchor fraction as the targets, potentials
cross-fitted over node folds so the rehearsal cannot flatter itself -- and choose
the graph / row factor strengths that maximise R_adv there. The best coupling is
adopted only if its paired per-node gain over no coupling exceeds ``calib_z``
standard errors; otherwise CoBP-G returns its unary. Nothing about the targets
is used.
"""
from __future__ import annotations

import itertools

import numpy as np

from graphrecon.attacks import ATTACKS, register_attack
from graphrecon.attacks.base import NodePrediction
from graphrecon.attacks.common.bp import BP_DEFAULTS, FactorType, loopy_bp
from graphrecon.attacks.common.classifiers import CLASSIFIER_DEFAULTS
from graphrecon.attacks.synth.node.gnn_classifier import GNNNodeModel
from graphrecon.attacks.synth.node.models import FeatureClassifierModel, cross_fit_proba
from graphrecon.data.graph import MISSING
from graphrecon.releases.synth.attr_models import _mutual_info
from graphrecon.scoring.node import rarity_weights

COBP_DEFAULTS = {
    "unary": "RandomForest",       # any synth/node classifier attack, or GNNClassifier
    "unary_params": {},            # overrides for that attack's defaults
    "graph_factors": True,         # couple each hidden attribute across observed target edges
    "row_factors": True,           # couple hidden attributes within a node (Chow-Liu tree)
    "potential": "residual",       # "residual" | "pmi"
    "alpha": 1.0,                  # pseudo-count for potential tables
    "cv_folds": 3,                 # folds for out-of-fold synthetic posteriors
    "strength": "auto",            # "auto" (self-calibrate on synth) or [graph_strength, row_strength]
    "strength_grid": [0.0, 0.25, 0.5, 1.0],
    "calib_z": 1.0,                # required significance of the rehearsal gain over no coupling
    **BP_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _unary_factory(name: str, overrides: dict, seed: int):
    params = ATTACKS.resolve(("synth", "node", name), overrides)
    if name in CLASSIFIER_DEFAULTS:
        return lambda: FeatureClassifierModel(name, params, seed)
    if name == "GNNClassifier":
        return lambda: GNNNodeModel(params, seed)
    raise ValueError(f"CoBP-G unary must be a classifier attack or GNNClassifier, got {name!r}")


def _hidden_tree(graph, hidden: list[str]) -> list[tuple[str, str]]:
    """Chow-Liu (max-MI spanning) tree over the hidden attributes of the synthetic nodes."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(hidden)
    for a, b in itertools.combinations(hidden, 2):
        x, y = graph.col(a), graph.col(b)
        ok = (x != MISSING) & (y != MISSING)
        if ok.sum() > 1:
            G.add_edge(a, b, weight=_mutual_info(x[ok], y[ok], graph.card(a), graph.card(b)))
    return list(nx.maximum_spanning_tree(G).edges())


def _log_potential(la, lb, Ca, Cb, src, dst, kind, alpha, Qa=None, Qb=None) -> np.ndarray:
    """Potential table from the pairs (src[e], dst[e]) with labels la, lb (and posteriors Qa, Qb)."""
    ok = (la[src] != MISSING) & (lb[dst] != MISSING)
    s, d = src[ok], dst[ok]
    O = np.zeros((Ca, Cb))
    np.add.at(O, (la[s], lb[d]), 1.0)
    if kind == "pmi":
        J = (O + alpha) / (O + alpha).sum()
        return np.log(J) - np.log(J.sum(1, keepdims=True)) - np.log(J.sum(0, keepdims=True))
    if kind == "residual":
        E = Qa[s].T @ Qb[d]
        return np.log(O + alpha) - np.log(E + alpha)
    raise ValueError(f"unknown potential {kind!r}")


def _node_scores(labels: dict, beliefs: dict) -> np.ndarray:
    """Per-node rarity-weighted correctness, averaged over hidden attributes (the
    row-level R_adv); mean over nodes ~ R_adv. NaN for nodes with no observed label."""
    tot, cnt = 0.0, 0.0
    for h, lab in labels.items():
        ok = lab != MISSING
        s = np.zeros(len(lab))
        w = rarity_weights(lab[ok])
        s[ok] = 100.0 * w * (beliefs[h].argmax(1)[ok] == lab[ok]) / w.mean()
        tot, cnt = tot + s, cnt + ok
    return np.divide(tot, cnt, out=np.full(len(cnt), np.nan), where=cnt > 0)


def _clamp(logu: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Pin variables with a known value (anchors) to that value."""
    lu = logu.copy()
    m = known != MISSING
    lu[m] = -30.0
    lu[np.flatnonzero(m), known[m]] = 0.0
    return lu


def _calibrate(syn, hidden, labels, Q, tree, potentials, knowledge, p, bp_kw, rng, info):
    """Choose (graph, row) strengths by rehearsing the attack on the synthetic graph.

    The rehearsal mirrors the target situation: out-of-fold unaries, the same
    observed-edge fraction, and the same fraction of anchors (known labels). To keep
    it honest, potentials are cross-fitted over two node folds: node fold A's labels
    estimate the potentials, BP runs on the full synthetic graph, and only fold-B
    nodes (whose labels the potentials never saw) are scored; then the folds swap.
    """
    n = syn.num_nodes
    keep = rng.random(syn.num_edges) < knowledge.get("observe_edges_frac", 1.0)
    edges = syn.edges[keep]
    frac_known = knowledge.get("known_label_frac", 0.0)
    known = {h: np.where(rng.random(n) < frac_known, labels[h], MISSING) for h in hidden}
    logQ = {h: _clamp(np.log(Q[h]), known[h]) for h in hidden}
    scored = {h: np.where(known[h] == MISSING, labels[h], MISSING) for h in hidden}
    fold = rng.integers(2, size=n).astype(bool)
    fold_psi = [potentials(~fold), potentials(fold)]          # potentials from the *other* fold
    g_grid = p["strength_grid"] if p["graph_factors"] else [0.0]
    r_grid = p["strength_grid"] if tree else [0.0]
    per_node = {}
    for sg, sr in itertools.product(g_grid, r_grid):
        score = np.full(n, np.nan)
        for f, (pg, pr) in zip((fold, ~fold), fold_psi):
            F = _factors(edges, n, hidden, tree, pg, pr, sg, sr)
            B = loopy_bp(logQ, F, **bp_kw)[0] if F else logQ
            score[f] = _node_scores(scored, B)[f]
        per_node[(sg, sr)] = score
    calib = {k: float(np.nanmean(v)) for k, v in per_node.items()}
    # best score (ties -> weaker coupling), kept only if significantly better than none
    best = max(calib, key=lambda k: (round(calib[k], 6), -k[0] - k[1]))
    diff = per_node[best] - per_node[(0.0, 0.0)]
    diff = diff[~np.isnan(diff)]
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.inf
    gain_z = diff.mean() / se if se > 0 else (np.inf if diff.mean() > 0 else 0.0)
    info["calibration"] = {f"{k[0]}/{k[1]}": round(v, 3) for k, v in calib.items()}
    info["calibration_gain_z"] = round(float(gain_z), 2)
    return best if gain_z >= p["calib_z"] else (0.0, 0.0)


def _factors(edges, n, hidden, tree, psi_graph, psi_row, s_graph, s_row) -> list[FactorType]:
    F = []
    if s_graph > 0 and len(edges):
        for h in hidden:
            F.append(FactorType(h, h, edges[:, 0], edges[:, 1], s_graph * psi_graph[h]))
    if s_row > 0:
        idx = np.arange(n)
        for a, b in tree:
            F.append(FactorType(a, b, idx, idx, s_row * psi_row[(a, b)]))
    return F


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------

@register_attack("synth", "node", "CoBP-G", defaults=COBP_DEFAULTS)
def cobp_g(release, view, rng, **p):
    """Classifier unaries + loopy BP across target edges and across each node's hidden attributes."""
    hidden, qi = view.task.hidden, view.task.qi
    syn = release
    seed = int(rng.integers(2**31))
    make = _unary_factory(p["unary"], p["unary_params"], seed)
    cards = {h: view.card(h) for h in hidden}
    labels = {h: syn.col(h) for h in hidden}

    # 1. unaries on the targets (trained on the whole synthetic graph)
    U = {h: make().fit(syn, qi, labels[h], cards[h]).predict_proba(view.graph) for h in hidden}

    # 2. out-of-fold synthetic posteriors (residual potentials, self-calibration)
    need_oof = p["potential"] == "residual" or p["strength"] == "auto"
    Q = {h: cross_fit_proba(make, syn, qi, labels[h], cards[h], p["cv_folds"], rng) for h in hidden} \
        if need_oof else None

    # 3. potentials from the synthetic graph
    tree = _hidden_tree(syn, hidden) if p["row_factors"] and len(hidden) > 1 else []

    def potentials(node_mask):
        """Graph / row potentials using only the labels of nodes in ``node_mask``."""
        e = syn.edges[node_mask[syn.edges[:, 0]] & node_mask[syn.edges[:, 1]]]
        both = np.concatenate([e, e[:, ::-1]])                 # symmetric: count each edge both ways
        idx = np.flatnonzero(node_mask)
        pg = {h: _log_potential(labels[h], labels[h], cards[h], cards[h], both[:, 0], both[:, 1],
                                p["potential"], p["alpha"], Q and Q[h], Q and Q[h]) for h in hidden}
        pr = {(a, b): _log_potential(labels[a], labels[b], cards[a], cards[b], idx, idx,
                                     p["potential"], p["alpha"], Q and Q[a], Q and Q[b]) for a, b in tree}
        return pg, pr

    psi_graph, psi_row = potentials(np.ones(syn.num_nodes, dtype=bool))
    bp_kw = {k: p[k] for k in BP_DEFAULTS}

    # 4. factor strengths
    info = {"row_tree": tree}
    if p["strength"] == "auto":
        s_graph, s_row = _calibrate(syn, hidden, labels, Q, tree, potentials, view.knowledge, p, bp_kw, rng, info)
    else:
        s_graph, s_row = p["strength"]
        s_graph = s_graph if p["graph_factors"] else 0.0
        s_row = s_row if tree else 0.0
    info.update(graph_strength=s_graph, row_strength=s_row)

    # 5. BP on the targets (known labels clamped)
    logU = {h: _clamp(np.log(U[h]), view.known[h]) if h in view.known else np.log(U[h]) for h in hidden}
    F = _factors(view.graph.edges, view.num_targets, hidden, tree, psi_graph, psi_row, s_graph, s_row)
    B, bp_info = loopy_bp(logU, F, **bp_kw)
    info.update(bp_info)
    pred = NodePrediction({h: np.exp(B[h]) for h in hidden})
    pred.info = info
    return pred
