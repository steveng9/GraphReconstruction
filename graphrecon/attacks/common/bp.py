"""Vectorised loopy sum-product belief propagation on a pairwise MRF (log space, torch).

Variables come in *groups* that share a cardinality, e.g. group ``"dorm"`` holds
the dorm variable of every target node. A *factor type* connects variables of
group ``a`` to variables of group ``b`` along an edge list with one shared
log-potential table ``log_psi[C_a, C_b]``:

* graph factors: a = b = h, edges = the observed target edges (homophily in h);
* row factors:   a = h, b = g, edges = (i, i) for every node (h-g dependence).

Messages are updated synchronously (flooding) with damping in log space. On a
tree the fixed point gives exact marginals (tested against brute force). Optional
entropy weighting scales each outgoing message by the sender's confidence
1 - H(cavity)/log C, so uncertain variables barely influence their neighbours
(the variant CoBP-RA found best on tabular data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

BP_DEFAULTS = {"max_iter": 30, "damping": 0.5, "tol": 1e-4, "entropy_weighted": False}


@dataclass
class FactorType:
    a: str
    b: str
    src: np.ndarray        # (E,) indices into group a
    dst: np.ndarray        # (E,) indices into group b
    log_psi: np.ndarray    # (C_a, C_b)


def _normalize(logm: torch.Tensor) -> torch.Tensor:
    return logm - torch.logsumexp(logm, dim=-1, keepdim=True)


def _confidence(log_p: torch.Tensor) -> torch.Tensor:
    C = log_p.shape[-1]
    if C < 2:
        return torch.ones(log_p.shape[:-1], dtype=log_p.dtype, device=log_p.device)
    p = log_p.exp()
    H = -(p * log_p.clamp_min(-1e30)).sum(-1)
    return (1.0 - H / math.log(C)).clamp(0.0, 1.0)


def loopy_bp(log_unaries: dict[str, np.ndarray], factors: list[FactorType], max_iter: int = 30,
             damping: float = 0.5, tol: float = 1e-4, entropy_weighted: bool = False,
             device: str | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Returns ({group: (n_g, C_g) normalised log beliefs}, info)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float64
    U = {g: _normalize(torch.as_tensor(u, dtype=dt, device=device)) for g, u in log_unaries.items()}
    F = []
    for f in factors:
        if len(f.src) == 0:
            continue
        F.append({
            "a": f.a, "b": f.b,
            "src": torch.as_tensor(f.src, dtype=torch.long, device=device),
            "dst": torch.as_tensor(f.dst, dtype=torch.long, device=device),
            "psi": torch.as_tensor(f.log_psi, dtype=dt, device=device),
        })
    # m_ab[e]: message from (a, src[e]) to (b, dst[e]); m_ba[e]: the reverse.
    for f in F:
        E = len(f["src"])
        f["m_ab"] = torch.full((E, f["psi"].shape[1]), -math.log(f["psi"].shape[1]), dtype=dt, device=device)
        f["m_ba"] = torch.full((E, f["psi"].shape[0]), -math.log(f["psi"].shape[0]), dtype=dt, device=device)

    def beliefs():
        B = {g: u.clone() for g, u in U.items()}
        for f in F:
            B[f["b"]].index_add_(0, f["dst"], f["m_ab"])
            B[f["a"]].index_add_(0, f["src"], f["m_ba"])
        return B

    delta, it = float("inf"), 0
    for it in range(1, max_iter + 1):
        B = beliefs()
        delta = 0.0
        for f in F:
            # cavity of the sender excludes the message the receiver sent it
            cav_a = B[f["a"]][f["src"]] - f["m_ba"]            # (E, C_a)
            cav_b = B[f["b"]][f["dst"]] - f["m_ab"]            # (E, C_b)
            new_ab = _normalize(torch.logsumexp(cav_a[:, :, None] + f["psi"][None], dim=1))
            new_ba = _normalize(torch.logsumexp(cav_b[:, None, :] + f["psi"][None], dim=2))
            if entropy_weighted:
                new_ab = new_ab * _confidence(_normalize(cav_a))[:, None]
                new_ba = new_ba * _confidence(_normalize(cav_b))[:, None]
            new_ab = (1 - damping) * new_ab + damping * f["m_ab"]
            new_ba = (1 - damping) * new_ba + damping * f["m_ba"]
            delta = max(delta, float((new_ab - f["m_ab"]).abs().max()), float((new_ba - f["m_ba"]).abs().max()))
            f["m_ab"], f["m_ba"] = new_ab, new_ba
        if delta < tol:
            break
    B = beliefs()
    out = {g: _normalize(b).cpu().numpy() for g, b in B.items()}
    return out, {"iterations": it, "final_delta": delta}
