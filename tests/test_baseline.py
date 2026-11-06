import networkx as nx
import pytest

from gnn_generalization.baseline import shortest_path_distances
from gnn_generalization.graphs import generate_graph


def test_weighted_detour_beats_direct_edge():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2), (1, 2, 3), (0, 2, 9)])
    assert shortest_path_distances(graph, 0) == {0: 0, 1: 2, 2: 5}


def test_generated_path_matches_cumulative_weights():
    graph, source = generate_graph("path", 12, seed=42)
    expected = {}
    for target in graph:
        start, end = sorted((source, target))
        expected[target] = sum(graph[i][i + 1]["weight"] for i in range(start, end))
    assert shortest_path_distances(graph, source) == expected


@pytest.mark.parametrize("weight", [None, 0, -1, 1.5, float("inf"), True])
def test_invalid_weights(weight):
    graph = nx.path_graph(2)
    if weight is not None:
        graph[0][1]["weight"] = weight
    with pytest.raises(ValueError, match="positive integer"):
        shortest_path_distances(graph, 0)


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])])
def test_rejects_nonsimple_or_directed_graphs(graph):
    with pytest.raises(ValueError, match="simple undirected"):
        shortest_path_distances(graph, 0)


def test_rejects_missing_source():
    with pytest.raises(ValueError, match="source"):
        shortest_path_distances(nx.path_graph(2), 5)


def test_rejects_disconnected_graph():
    with pytest.raises(ValueError, match="connected"):
        shortest_path_distances(nx.empty_graph(2), 0)
