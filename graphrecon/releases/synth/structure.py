"""Random-graph samplers used by the synthetic graph generators."""
from __future__ import annotations

from typing import Callable

import numpy as np

from graphrecon.data.graph import canonical_edges


def chung_lu(weights: np.ndarray, num_edges: int, rng: np.random.Generator,
             accept: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
             max_rounds: int = 200) -> np.ndarray:
    """Sample ``num_edges`` distinct edges with endpoints drawn proportional to ``weights``.

    If ``accept(u, v)`` is given it returns per-proposal acceptance probabilities
    (the AGM proposal/acceptance scheme). Duplicates and self loops are rejected.
    """
    n = len(weights)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    keys: set[int] = set()
    for _ in range(max_rounds):
        need = num_edges - len(keys)
        if need <= 0:
            break
        batch = max(2 * need, 1024)
        u = rng.choice(n, size=batch, p=p)
        v = rng.choice(n, size=batch, p=p)
        ok = u != v
        if accept is not None:
            ok &= rng.random(batch) < accept(u, v)
        a, b = np.minimum(u[ok], v[ok]), np.maximum(u[ok], v[ok])
        for key in (a * n + b).tolist():
            keys.add(key)
            if len(keys) >= num_edges:
                break
    k = np.fromiter(keys, dtype=np.int64, count=len(keys))
    return canonical_edges(np.stack([k // n, k % n], axis=1))


def dcsbm(blocks: np.ndarray, weights: np.ndarray, block_edges: np.ndarray,
          rng: np.random.Generator) -> np.ndarray:
    """Degree-corrected SBM: ``block_edges[r, s]`` distinct edges between blocks r <= s,
    endpoints within a block drawn proportional to ``weights``."""
    out = []
    members = {r: np.flatnonzero(blocks == r) for r in range(block_edges.shape[0])}
    for r in range(block_edges.shape[0]):
        for s in range(r, block_edges.shape[0]):
            m = int(block_edges[r, s])
            A, B = members[r], members[s]
            if m == 0 or len(A) == 0 or len(B) == 0 or (r == s and len(A) < 2):
                continue
            max_pairs = len(A) * (len(A) - 1) // 2 if r == s else len(A) * len(B)
            m = min(m, max_pairs)
            pa = weights[A] / weights[A].sum()
            pb = weights[B] / weights[B].sum()
            keys: set[int] = set()
            n = len(blocks)
            for _ in range(200):
                need = m - len(keys)
                if need <= 0:
                    break
                u = A[rng.choice(len(A), size=2 * need + 16, p=pa)]
                v = B[rng.choice(len(B), size=len(u), p=pb)]
                ok = u != v
                a, b = np.minimum(u[ok], v[ok]), np.maximum(u[ok], v[ok])
                for key in (a * n + b).tolist():
                    keys.add(key)
                    if len(keys) >= m:
                        break
            k = np.fromiter(keys, dtype=np.int64, count=len(keys))
            out.append(np.stack([k // n, k % n], axis=1))
    return canonical_edges(np.concatenate(out) if out else np.zeros((0, 2), np.int64))
