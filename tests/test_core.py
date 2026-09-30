"""Graph container, splits, views, registry and scoring."""
import numpy as np
import pytest

from graphrecon.data.datasets import load_dataset
from graphrecon.data.graph import MISSING, AttributedGraph
from graphrecon.data.splits import make_split
from graphrecon.registry import Registry
from graphrecon.scoring.edge import score_edges
from graphrecon.scoring.node import rarity_weighted_score, score_nodes
from graphrecon.threat.views import EdgeTask, NodeTask, make_edge_view, make_node_view


def tiny():
    return AttributedGraph(4, [[0, 1], [1, 0], [2, 2], [3, 1]], [[0, 1], [1, 0], [1, MISSING], [0, 0]],
                           {"a": ["x", "y"], "b": ["p", "q"]})


def test_graph_canonicalises_edges_and_roundtrips(tmp_path):
    g = tiny()
    assert g.edges.tolist() == [[0, 1], [1, 3]]          # dedup, no self loop, u < v
    assert g.degrees().tolist() == [1, 2, 0, 1]
    g.save(tmp_path)
    h = AttributedGraph.load(tmp_path)
    assert (h.edges == g.edges).all() and (h.attrs == g.attrs).all() and h.schema == g.schema


def test_subgraph_relabels():
    s = tiny().subgraph(np.array([3, 1]))
    assert s.edges.tolist() == [[0, 1]] and s.col("a").tolist() == [0, 1]


def test_attribute_codes_validated():
    with pytest.raises(ValueError):
        AttributedGraph(1, np.zeros((0, 2)), [[5]], {"a": ["x", "y"]})


def test_split_parts_are_disjoint_and_cover():
    g = load_dataset("toy_sbm")
    sp = make_split(g, sample=1, n_parts=4, seed=0)
    ids = [sp.parts[r] for r in ("train", "holdout", "aux")]
    assert sum(len(i) for i in ids) == len(np.unique(np.concatenate(ids)))


def test_node_view_hides_hidden_attributes():
    g = load_dataset("toy_sbm")
    task = NodeTask(qi=["qi"], hidden=["label"])
    view, truth = make_node_view(g, task, {"observe_edges_frac": 0.5, "known_label_frac": 0.2}, seed=0)
    assert view.graph.attr_names == ["qi"]
    assert abs(view.graph.num_edges - g.num_edges / 2) <= 1
    k = view.known["label"]
    assert ((k == MISSING) | (k == truth["label"])).all() and 0.1 < (k != MISSING).mean() < 0.3
    with pytest.raises(ValueError):
        NodeTask(qi=["label"], hidden=["label"])


def test_edge_view_pairs_are_hidden_edges_and_non_edges():
    g = load_dataset("toy_sbm")
    view, truth = make_edge_view(g, EdgeTask(["qi"]), {"observe_edges_frac": 0.5}, seed=0)
    edge_set = set(map(tuple, g.edges.tolist()))
    observed = set(map(tuple, view.graph.edges.tolist()))
    for (u, v), t in zip(view.pairs.tolist(), truth):
        assert ((u, v) in edge_set) == bool(t)
        assert (u, v) not in observed
    assert truth.sum() == len(truth) - truth.sum()


def test_registry_rejects_unknown_params():
    r = Registry("thing")
    r.register("x", defaults={"a": 1})(lambda a: a)
    assert r.resolve("x", {"a": 2}) == {"a": 2}
    with pytest.raises(KeyError):
        r.resolve("x", {"typo": 2})


def test_rarity_weighted_score_calibration():
    t = np.array([0, 0, 0, 1])
    assert rarity_weighted_score(t, np.zeros(4, int)) == pytest.approx(50.0)   # mode = 100 / C
    assert rarity_weighted_score(t, t) == pytest.approx(100.0)


def test_score_nodes_excludes_missing_and_known():
    truth = {"h": np.array([0, 1, MISSING, 1])}
    pred = {"h": np.array([0, 0, 1, 1])}
    s, _ = score_nodes(truth, pred, {"h": np.array([MISSING, 1, MISSING, MISSING])})
    assert s[("h", "n_scored")] == 2 and s[("h", "accuracy")] == 100.0


def test_edge_scores():
    s = score_edges(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.1, 0.2]))
    assert s[("edges", "auc")] == 1.0 and s[("edges", "precision_at_k")] == 1.0
