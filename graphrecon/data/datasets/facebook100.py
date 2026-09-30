"""Facebook100 (Traud, Mucha & Porter 2012): one friendship network per US college,
with categorical node attributes. The closest graph analogue of a demographic
tabular dataset, so it is our main dataset for node-attribute reconstruction.

Raw files are fetched from the LINKX repository (CUAI/Non-Homophily-Large-Scale).
``local_info`` columns: status (student/faculty/...), gender, major, minor, dorm,
year, high_school; 0 denotes a missing value.
"""
from __future__ import annotations

import urllib.request

import numpy as np

from graphrecon.data.datasets import DATASETS
from graphrecon.data.datasets.categorical import encode_column
from graphrecon.data.graph import AttributedGraph
from graphrecon.paths import RAW_DATA

LOCAL_INFO_COLUMNS = ["status", "gender", "major", "minor", "dorm", "year", "high_school"]
_URL = "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/{}.mat"
_REMOTE_NAME = {"JohnsHopkins55": "Johns%20Hopkins55"}


def _raw_path(school: str):
    path = RAW_DATA / "facebook100" / f"{school}.mat"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URL.format(_REMOTE_NAME.get(school, school)), path)
    return path


@DATASETS.register("facebook100", defaults={
    "school": "Amherst41",            # Reed98 (962 nodes) | Amherst41 (2.2k) | JohnsHopkins55 (5.2k) | Cornell5 | Penn94
    "attributes": ["status", "gender", "major", "dorm", "year"],  # minor (61% missing) and high_school (~2k values) dropped
    "max_categories": 12,             # cap per attribute: top-11 values + "other"
    "largest_component": True,
})
def load_facebook100(school, attributes, max_categories, largest_component):
    """Facebook100 college friendship network with categorical demographics."""
    from scipy.io import loadmat
    import scipy.sparse as sp

    mat = loadmat(_raw_path(school))
    A = sp.triu(mat["A"].tocsr(), k=1).tocoo()
    info = mat["local_info"].astype(np.int64)
    n = info.shape[0]
    edges = np.stack([A.row, A.col], axis=1)

    nodes = np.arange(n)
    if largest_component:
        from scipy.sparse.csgraph import connected_components
        _, comp = connected_components(mat["A"], directed=False)
        nodes = np.flatnonzero(comp == np.bincount(comp).argmax())

    codes, schema = [], {}
    for a in attributes:
        raw = info[:, LOCAL_INFO_COLUMNS.index(a)]
        labels = {1: "female", 2: "male"} if a == "gender" else None
        c, names = encode_column(raw, missing_values=(0,), max_categories=max_categories, labels=labels)
        codes.append(c)
        schema[a] = names
    g = AttributedGraph(n, edges, np.stack(codes, axis=1), schema, name=f"fb100-{school}")
    return g.subgraph(nodes) if largest_component else g
