"""Dataset registry. ``load_dataset(name, params)`` returns a cached ``AttributedGraph``.

Loaders are registered with their parameter defaults; the processed graph is
cached under workspace/datasets/processed/<name>-<param hash>/.
"""
from __future__ import annotations

from graphrecon.config import param_hash
from graphrecon.data.graph import AttributedGraph
from graphrecon.paths import PROCESSED_DATA
from graphrecon.registry import Registry

DATASETS = Registry("dataset")


def dataset_key(name: str, params: dict | None = None) -> str:
    """Stable identifier of a dataset configuration, e.g. ``facebook100-3f2a...``."""
    return f"{name}-{param_hash(DATASETS.resolve(name, params), 8)}"


def load_dataset(name: str, params: dict | None = None, cache: bool = True) -> AttributedGraph:
    p = DATASETS.resolve(name, params)
    out = PROCESSED_DATA / dataset_key(name, params)
    if cache and AttributedGraph.exists(out):
        return AttributedGraph.load(out)
    g = DATASETS.get(name).fn(**p)
    g.info["dataset_params"] = p
    if cache:
        g.save(out)
    return g


# Register loaders (import for side effects).
from graphrecon.data.datasets import facebook100, planetoid, toy  # noqa: E402,F401
