"""Generators keep the schema and edge budget; GNN releases train and reload."""
import numpy as np
import pytest

from graphrecon.data.datasets import load_dataset
from graphrecon.data.graph import AttributedGraph
from graphrecon.releases.synth import GENERATORS
from graphrecon.releases.synth.attr_models import AttributeModel


@pytest.fixture(scope="module")
def graph():
    return load_dataset("toy_sbm")


@pytest.mark.parametrize("name", ["Identity", "ChungLu", "AGM", "DCSBM"])
def test_generator_contract(graph, name):
    syn = GENERATORS.get(name).fn(graph, np.random.default_rng(0), **GENERATORS.resolve(name, {}))
    assert isinstance(syn, AttributedGraph)
    assert syn.schema == graph.schema and syn.num_nodes == graph.num_nodes
    assert abs(syn.num_edges - graph.num_edges) <= 0.05 * graph.num_edges


def test_identity_is_isomorphic(graph):
    syn = GENERATORS.get("Identity").fn(graph, np.random.default_rng(0))
    assert sorted(syn.degrees()) == sorted(graph.degrees())


@pytest.mark.parametrize("kind", ["empirical", "marginals", "chowliu"])
def test_attribute_models_respect_cardinality(graph, kind):
    rows = AttributeModel(kind).fit(graph.attrs, [graph.card(a) for a in graph.attr_names]) \
        .sample(500, np.random.default_rng(0))
    for j, a in enumerate(graph.attr_names):
        assert rows[:, j].min() >= -1 and rows[:, j].max() < graph.card(a)


def test_gnn_release_trains_and_reloads(graph, tmp_path):
    from graphrecon.releases.gnn import GNNRelease, train_gnn
    rel, metrics = train_gnn(graph, "SAGE", {"hidden": 16, "layers": 2, "dropout": 0.0, "aggr": "mean"},
                             {"label": "label", "epochs": 50}, seed=0, device="cpu")
    assert metrics["val_acc"] > 0.5          # homophilous toy graph is easy
    rel.save(tmp_path)
    again = GNNRelease.load(tmp_path, server_graph=graph, device="cpu")
    np.testing.assert_allclose(again.query_nodes(), rel.query_graph(graph), atol=1e-5)
