"""Reference labels for connected graphs with positive integer weights."""

import networkx as nx


def shortest_path_distances(graph: nx.Graph, source: int) -> dict[int, int]:
    """Compute exact weighted distances without mutating the graph."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("expected a simple undirected graph")
    if source not in graph:
        raise ValueError("source must be a vertex of the graph")
    for _, _, attributes in graph.edges(data=True):
        weight = attributes.get("weight")
        if type(weight) is not int or weight <= 0:
            raise ValueError("each edge must have a positive integer weight")
    distances = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    if len(distances) != len(graph):
        raise ValueError("expected a connected graph")
    return distances
