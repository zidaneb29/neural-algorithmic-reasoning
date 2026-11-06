import networkx as nx
import pytest

torch = pytest.importorskip("torch")

from gnn_generalization.data import encode_query
from gnn_generalization.models import ShortestPathGNN
from gnn_generalization.pilot import evaluate, train


def predict(model, query):
    return model(query.source_mask, query.edge_index, query.edge_weight)


def test_encode_weighted_detour_and_reciprocal_edges():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(3, 5, 2), (5, 9, 3), (3, 9, 9)])
    query = encode_query(graph, 3)
    assert query.source_mask.tolist() == [True, False, False]
    assert query.hops.tolist() == [0, 1, 1]
    torch.testing.assert_close(query.target, torch.tensor([0.0, 0.2, 0.5]))
    assert query.edge_index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    torch.testing.assert_close(
        query.edge_weight.flatten(), torch.tensor([0.2, 0.9, 0.2, 0.3, 0.9, 0.3])
    )


def test_initialization_is_paired_without_changing_global_rng():
    state = torch.random.get_rng_state()
    standard = ShortestPathGNN(width=8, depth=2, residual=False, seed=11)
    residual = ShortestPathGNN(width=8, depth=2, residual=True, seed=11)
    assert torch.equal(state, torch.random.get_rng_state())
    for a, b in zip(standard.parameters(), residual.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("residual", [False, True])
def test_locality_and_equivariance(residual):
    graph = nx.path_graph(7)
    nx.set_edge_attributes(graph, 2, "weight")
    model = ShortestPathGNN(width=8, depth=2, residual=residual, seed=11)
    prediction = predict(model, encode_query(graph, 0))
    moved = predict(model, encode_query(graph, 1))
    assert prediction[0].item() == 0
    torch.testing.assert_close(prediction[4:], moved[4:])
    reversed_labels = nx.relabel_nodes(graph, {node: 6 - node for node in graph})
    torch.testing.assert_close(
        prediction, predict(model, encode_query(reversed_labels, 6)).flip(0)
    )


@pytest.mark.parametrize("residual", [False, True])
def test_single_graph_fit(residual):
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 2, "weight")
    query = encode_query(graph, 0)
    model, history = train([query], [query], residual=residual, seed=11, epochs=100)
    mae = evaluate(model, [query])["mae"]
    assert mae == pytest.approx(min(row["validation_mae"] for row in history))
    assert mae < 0.2
