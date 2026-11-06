"""Tensor encoding of a single graph-source query."""

from dataclasses import dataclass

import networkx as nx
import torch
from torch import Tensor

from .baseline import shortest_path_distances

DISTANCE_SCALE = 10.0


@dataclass(frozen=True)
class Query:
    source_mask: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    target: Tensor
    hops: Tensor


def encode_query(graph: nx.Graph, source: int) -> Query:
    """Encode reciprocal edges and distances in units of ten weight units."""
    distances = shortest_path_distances(graph, source)
    hops = nx.single_source_shortest_path_length(graph, source)
    nodes = sorted(graph)
    positions = {node: index for index, node in enumerate(nodes)}
    edges = sorted(
        (positions[u], positions[v], weight)
        for a, b, weight in graph.edges(data="weight")
        for u, v in ((a, b), (b, a))
    )
    return Query(
        source_mask=torch.tensor([node == source for node in nodes]),
        edge_index=torch.tensor([(u, v) for u, v, _ in edges], dtype=torch.long)
        .reshape(-1, 2).T.contiguous(),
        edge_weight=torch.tensor([[w / DISTANCE_SCALE] for _, _, w in edges])
        .reshape(-1, 1),
        target=torch.tensor([distances[node] / DISTANCE_SCALE for node in nodes]),
        hops=torch.tensor([hops[node] for node in nodes]),
    )
