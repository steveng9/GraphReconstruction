"""Belief propagation must be exact on trees (checked against brute-force enumeration)."""
import itertools

import numpy as np

from graphrecon.attacks.common.bp import FactorType, loopy_bp


def brute_force(unaries, factors):
    """Exact marginals by enumerating every joint assignment."""
    groups = list(unaries)
    variables = [(g, i) for g in groups for i in range(len(unaries[g]))]
    cards = [unaries[g].shape[1] for g, _ in variables]
    index = {v: k for k, v in enumerate(variables)}
    marg = {g: np.zeros_like(unaries[g]) for g in groups}
    for assign in itertools.product(*[range(c) for c in cards]):
        lp = sum(unaries[g][i, assign[index[(g, i)]]] for g, i in variables)
        for f in factors:
            for s, d in zip(f.src, f.dst):
                lp += f.log_psi[assign[index[(f.a, s)]], assign[index[(f.b, d)]]]
        w = np.exp(lp)
        for g, i in variables:
            marg[g][i, assign[index[(g, i)]]] += w
    return {g: m / m.sum(1, keepdims=True) for g, m in marg.items()}


def test_tree_exact_mixed_groups():
    rng = np.random.default_rng(0)
    # group h: 4 nodes on a path (graph factors); group g: row factors to h
    U = {"h": np.log(rng.dirichlet(np.ones(3), size=4)), "g": np.log(rng.dirichlet(np.ones(2), size=4))}
    psi_hh = rng.normal(size=(3, 3)); psi_hh = (psi_hh + psi_hh.T) / 2
    factors = [FactorType("h", "h", np.array([0, 1, 2]), np.array([1, 2, 3]), psi_hh),
               FactorType("h", "g", np.arange(4), np.arange(4), rng.normal(size=(3, 2)))]
    bp, info = loopy_bp(U, factors, max_iter=100, damping=0.0, tol=1e-10, device="cpu")
    exact = brute_force(U, factors)
    for gname in U:
        np.testing.assert_allclose(np.exp(bp[gname]), exact[gname], atol=1e-6)


def test_no_factors_returns_unaries():
    U = {"h": np.log(np.array([[0.2, 0.8], [0.6, 0.4]]))}
    bp, _ = loopy_bp(U, [], device="cpu")
    np.testing.assert_allclose(np.exp(bp["h"]), np.exp(U["h"]))


def test_loopy_homophily_pulls_toward_neighbours():
    # triangle with two confident 'class 0' nodes and one uncertain node
    U = {"h": np.log(np.array([[0.9, 0.1], [0.9, 0.1], [0.45, 0.55]]))}
    psi = np.log(np.array([[3.0, 1.0], [1.0, 3.0]]))
    f = FactorType("h", "h", np.array([0, 1, 0]), np.array([1, 2, 2]), psi)
    bp, _ = loopy_bp(U, [f], device="cpu")
    assert np.exp(bp["h"][2, 0]) > 0.5
