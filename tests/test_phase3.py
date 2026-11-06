import networkx as nx
import pytest

pytest.importorskip("torch")

from gnn_generalization.data import encode_query
from gnn_generalization.models import ShortestPathGNN
from gnn_generalization.phase3 import (
    evaluate_predictions, locality_pair, make_manifest, training_mean,
)


def test_manifest_is_reproducible_and_weighted_graphs_are_disjoint():
    manifest = make_manifest(1000, 4, 2)
    replay = make_manifest(1000, 4, 2)
    match = nx.algorithms.isomorphism.categorical_edge_match("weight", None)
    seen = []
    for split, group in manifest.items():
        for item, copy in zip(group, replay[split], strict=True):
            assert (item.seed, item.source) == (copy.seed, copy.source)
            assert nx.utils.graphs_equal(item.graph, copy.graph)
            assert not any(nx.is_isomorphic(item.graph, other, edge_match=match)
                           for other in seen if len(other) == len(item.graph))
            seen.append(item.graph)


def test_constant_fits_training_only_and_scores_in_weight_units():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 2, "weight")
    value = training_mean([encode_query(graph, 0)])
    nx.set_edge_attributes(graph, 4, "weight")
    scores = evaluate_predictions([encode_query(graph, 0)], value)
    assert scores["mae"] == pytest.approx(3)
    assert scores["rmse"] == pytest.approx(13**0.5)
    assert scores["hop"][2]["mae"] == pytest.approx(5)


@pytest.mark.parametrize("residual", [False, True])
def test_four_round_paired_sources(residual):
    model = ShortestPathGNN(width=8, depth=4, residual=residual, seed=11)
    result = locality_pair(model, 4)
    assert result["targets"] == (5, 6)
    assert result["predictions"][0] == result["predictions"][1]
