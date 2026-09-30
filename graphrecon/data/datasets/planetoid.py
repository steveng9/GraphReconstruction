"""Planetoid citation graphs (Cora, CiteSeer, PubMed) via PyG.

The class label is the single categorical attribute; the bag-of-words vector is
kept as dense ``x`` (public features, not a reconstruction target). Used mainly
for the GNN pipeline (e.g. link stealing), where it is the standard benchmark.
"""
from __future__ import annotations

import numpy as np

from graphrecon.data.datasets import DATASETS
from graphrecon.data.graph import AttributedGraph
from graphrecon.paths import RAW_DATA


@DATASETS.register("planetoid", defaults={"name": "Cora"})
def load_planetoid(name):
    """Planetoid citation network: label as attribute, bag-of-words as dense features."""
    from torch_geometric.datasets import Planetoid

    d = Planetoid(str(RAW_DATA / "planetoid"), name)[0]
    y = d.y.numpy().astype(np.int64)
    schema = {"label": [f"class{i}" for i in range(int(y.max()) + 1)]}
    return AttributedGraph(d.num_nodes, d.edge_index.numpy().T, y[:, None], schema,
                           x=d.x.numpy().astype(np.float32), name=f"planetoid-{name}")
