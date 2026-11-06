"""Matched depth and topology experiments with graph-level records."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
from random import Random
from statistics import mean

import networkx as nx
import torch

from .data import DISTANCE_SCALE, Query, encode_query
from .graphs import generate_graph
from .models import ShortestPathGNN
from .pilot import (GraphInstance, LEARNING_RATE, WIDTH, make_splits,
                    manifest_entry, train)

DEPTHS = (2, 4, 8)
SIZES = (8, 16, 32, 64)
FAMILIES = ("erdos_renyi", "path", "cycle", "grid")


def make_manifest(seed: int, training_count: int, evaluation_count: int) -> dict:
    rng = Random(seed)
    seen = defaultdict(list)
    match = nx.algorithms.isomorphism.categorical_edge_match("weight", None)
    groups = {}
    specs = [
        ("train", "erdos_renyi", (6, 8), training_count),
        ("validation", "erdos_renyi", (6, 8), evaluation_count),
    ] + [(f"{family}/{n}", family, (n,), evaluation_count)
         for family in FAMILIES for n in SIZES]
    for split, family, sizes, count in specs:
        groups[split] = []
        for index in range(count):
            n = sizes[index % len(sizes)]
            p = 3 / (n - 1)
            for _ in range(100):
                graph_seed = rng.randrange(2**32)
                graph, source = generate_graph(
                    family, n, seed=graph_seed, p=p, max_attempts=1000
                )
                key = (n, tuple(sorted(nx.get_edge_attributes(graph, "weight").values())))
                if any(nx.is_isomorphic(graph, other, edge_match=match)
                       for other in seen[key]):
                    continue
                seen[key].append(graph)
                groups[split].append(GraphInstance(graph, source, split, family, graph_seed, p))
                break
            else:
                raise RuntimeError(f"weighted duplicate retry budget exhausted for {split}")
    return groups


def predict(model: ShortestPathGNN, query: Query) -> torch.Tensor:
    return model(query.source_mask, query.edge_index, query.edge_weight)


def training_mean(queries: list[Query]) -> float:
    return torch.cat([q.target[~q.source_mask] for q in queries]).mean().item()


@torch.no_grad()
def evaluate_predictions(queries: list[Query], model: ShortestPathGNN | float) -> dict:
    graphs = []
    hops = defaultdict(list)
    for q in queries:
        mask = ~q.source_mask
        outputs = predict(model, q) if isinstance(model, ShortestPathGNN) else torch.full_like(q.target, model)
        prediction = outputs[mask] * DISTANCE_SCALE
        target = (q.target[mask] * DISTANCE_SCALE).round()
        error = (prediction - target).abs()
        graphs.append({
            "mae": error.mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "normalized_mae": (error.sum() / target.sum()).item(),
            "nodes": [
                {"hop": h, "target": y, "prediction": yhat}
                for h, y, yhat in zip(q.hops[mask].tolist(), target.tolist(),
                                     prediction.tolist(), strict=True)
            ],
        })
        for hop, value in zip(q.hops[mask].tolist(), error.tolist(), strict=True):
            hops[hop].append(value)
    return {
        **{key: mean(g[key] for g in graphs) for key in ("mae", "rmse", "normalized_mae")},
        "graphs": graphs,
        "hop": {key: {"nodes": len(values), "mae": mean(values),
                      "rmse": mean(value**2 for value in values)**0.5}
                for key, values in sorted(hops.items())},
    }


@torch.no_grad()
def locality_pair(model: ShortestPathGNN, depth: int) -> dict:
    graph = nx.path_graph(depth + 3)
    nx.set_edge_attributes(graph, 1, "weight")
    sources = (depth + 1, depth + 2)
    predictions = [predict(model, encode_query(graph, s))[0].item() * DISTANCE_SCALE
                   for s in sources]
    if predictions[0] != predictions[1]:
        raise AssertionError("source outside the receptive field affected vertex zero")
    return {"sources": sources, "target_node": 0, "targets": sources,
            "predictions": predictions}


def run(output: Path, seeds: list[int], epochs: int, train_count: int, eval_count: int) -> None:
    torch.set_num_threads(1)
    with output.open("x", encoding="utf-8") as stream:
        def record(row: dict) -> None:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        record({
            "kind": "config", "python": platform.python_version(), "torch": torch.__version__,
            "networkx": nx.__version__, "device": "cpu", "threads": 1, "seeds": seeds,
            "data_seed_offset": 1000, "epochs": epochs, "updates_per_epoch": train_count,
            "evaluation_graphs_per_cell": eval_count, "depths": DEPTHS, "width": WIDTH,
            "families": FAMILIES, "sizes": SIZES, "train_validation_sizes": [6, 8],
            "optimizer": "Adam", "lr": LEARNING_RATE, "loss": "scaled non-source MAE",
            "selection": "minimum graph-macro validation MAE",
            "weights": "uniform integers 1-10", "scale": DISTANCE_SCALE,
            "source": "uniform", "max_connectivity_attempts": 1000,
        })
        original = make_splits(17)
        reference = {k: [encode_query(x.graph, x.source) for x in v] for k, v in original.items()}
        for name, value in (("zero", 0.), ("training_mean", training_mean(reference["train"]))):
            record({"kind": "phase2_baseline", "model": name, "constant": value * DISTANCE_SCALE,
                    "metrics": {k: evaluate_predictions(v, value) for k, v in reference.items()
                                if k != "train"}})
        for seed in seeds:
            groups = make_manifest(seed + 1000, train_count, eval_count)
            record({
                "kind": "manifest", "seed": seed, "data_seed": seed + 1000,
                "groups": {k: [dict(manifest_entry(x), weights=sorted(
                    nx.get_edge_attributes(x.graph, "weight").values())) for x in v]
                    for k, v in groups.items()},
            })
            queries = {k: [encode_query(x.graph, x.source) for x in v] for k, v in groups.items()}
            training = queries.pop("train")
            for name, value in (("zero", 0.), ("training_mean", training_mean(training))):
                record({"kind": "baseline", "seed": seed, "model": name,
                        "constant": value * DISTANCE_SCALE,
                        "metrics": {k: evaluate_predictions(v, value) for k, v in queries.items()}})
            for depth in DEPTHS:
                for residual in (False, True):
                    model, history = train(training, queries["validation"], residual=residual,
                                           seed=seed, epochs=epochs, depth=depth)
                    model.eval()
                    name = "residual" if residual else "standard"
                    metrics = {k: evaluate_predictions(v, model) for k, v in queries.items()}
                    record({
                        "kind": "fit", "seed": seed, "model": name, "depth": depth,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "best_epoch": min(range(epochs), key=lambda i: history[i]["validation_mae"]) + 1,
                        "history": history, "metrics": metrics, "locality": locality_pair(model, depth),
                    })
                    print(f"{seed} {name} depth={depth} val={metrics['validation']['mae']:.3f}",
                          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-graphs", type=int, required=True)
    parser.add_argument("--eval-graphs", type=int, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.train_graphs, args.eval_graphs) < 1:
        parser.error("budgets must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    run(args.output, args.seeds, args.epochs, args.train_graphs, args.eval_graphs)


if __name__ == "__main__":
    main()
