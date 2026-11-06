"""Seeded graph construction with exact integer edge weights."""

from random import Random
from typing import Literal

import networkx as nx

Family = Literal["path", "cycle", "grid", "erdos_renyi", "regular"]


def generate_graph(
    family: Family,
    n: int,
    *,
    seed: int,
    p: float = 0.25,
    degree: int = 3,
    rows: int = 2,
    max_attempts: int = 100,
) -> tuple[nx.Graph, int]:
    """Return a connected weighted graph and a uniformly sampled source.

    Grid rows must divide n. Random families resample disconnected draws
    and raise RuntimeError if the attempt budget is exhausted.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = Random(seed)

    if family == "path":
        graph = nx.path_graph(n)
    elif family == "cycle":
        if n < 3:
            raise ValueError("a simple cycle requires at least 3 vertices")
        graph = nx.cycle_graph(n)
    elif family == "grid":
        if rows < 2 or n % rows or n // rows < 2:
            raise ValueError("grid requires at least two rows and columns and rows dividing n")
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, n // rows))
    elif family in ("erdos_renyi", "regular"):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if family == "erdos_renyi" and not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        if family == "regular":
            minimum_degree = 1 if n == 2 else 2
            if not minimum_degree <= degree < n or n * degree % 2:
                raise ValueError("degree and n must allow a connected regular graph")
        for _ in range(max_attempts):
            if family == "erdos_renyi":
                graph = nx.gnp_random_graph(n, p, seed=rng)
            else:
                graph = nx.random_regular_graph(degree, n, seed=rng)
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError(f"no connected {family} draw in {max_attempts} attempts")
    else:
        raise ValueError(f"unknown graph family {family!r}")

    for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges):
        graph[u][v]["weight"] = rng.randint(1, 10)
    return graph, rng.randrange(n)
