import random

import networkx as nx
import pytest

from gnn_generalization.graphs import generate_graph


@pytest.mark.parametrize(
    "family, settings, edge_count",
    [
        ("path", {}, 11),
        ("cycle", {}, 12),
        ("grid", {"rows": 3}, 17),
        ("erdos_renyi", {"p": 0.5}, None),
        ("regular", {"degree": 3}, 18),
    ],
)
def test_graph_properties_and_reproducibility(family, settings, edge_count):
    random_state = random.getstate()
    graph, source = generate_graph(family, 12, seed=42, **settings)
    duplicate, duplicate_source = generate_graph(family, 12, seed=42, **settings)

    assert random.getstate() == random_state
    assert set(graph) == set(range(12))
    assert source in graph
    assert nx.is_connected(graph)
    assert not graph.is_directed()
    assert not graph.is_multigraph()
    assert nx.number_of_selfloops(graph) == 0
    assert all(type(w) is int and 1 <= w <= 10 for _, _, w in graph.edges(data="weight"))
    assert nx.utils.graphs_equal(graph, duplicate)
    assert source == duplicate_source
    if edge_count is not None:
        assert graph.number_of_edges() == edge_count
    if family == "regular":
        assert {degree for _, degree in graph.degree} == {3}


@pytest.mark.parametrize(
    "family, n, settings",
    [
        ("unknown", 12, {}),
        ("path", 1, {}),
        ("cycle", 2, {}),
        ("grid", 11, {"rows": 3}),
        ("regular", 5, {"degree": 3}),
        ("erdos_renyi", 12, {"p": 0}),
        ("erdos_renyi", 12, {"max_attempts": 0}),
    ],
)
def test_invalid_generator_settings(family, n, settings):
    with pytest.raises(ValueError):
        generate_graph(family, n, seed=42, **settings)


def test_connectivity_budget():
    with pytest.raises(RuntimeError, match="no connected"):
        generate_graph("erdos_renyi", 12, seed=42, p=1e-9, max_attempts=2)
