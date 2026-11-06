"""Reproducible CPU pilot for matched shortest-path regressors."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
from random import Random

import networkx as nx
import torch

from .data import DISTANCE_SCALE, Query, encode_query
from .graphs import Family, generate_graph
from .models import ShortestPathGNN

WIDTH = 32
DEPTH = 4
LEARNING_RATE = 0.001
SPLITS = {"train": 64, "validation": 16, "test": 16, "larger": 16,
          "path": 4, "cycle": 4, "grid": 4}


@dataclass(frozen=True)
class GraphInstance:
    graph: nx.Graph
    source: int
    split: str
    family: Family
    seed: int
    p: float


def make_splits(seed: int) -> dict[str, list[GraphInstance]]:
    """Reject weighted isomorphic duplicates before encoding any queries."""
    rng = Random(seed)
    seen = defaultdict(list)
    splits = {}
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("weight", None)
    for split, count in SPLITS.items():
        family = split if split in ("path", "cycle", "grid") else "erdos_renyi"
        sizes = (12, 14) if split == "larger" else (6, 8)
        instances = []
        for _ in range(count):
            n = rng.choice(sizes)
            p = 3 / (n - 1)
            for _ in range(100):
                graph_seed = rng.randrange(2**32)
                graph, source = generate_graph(family, n, seed=graph_seed, p=p)
                key = nx.weisfeiler_lehman_graph_hash(graph, edge_attr="weight")
                if any(nx.is_isomorphic(graph, other, edge_match=edge_match)
                       for other in seen[key]):
                    continue
                seen[key].append(graph)
                instances.append(GraphInstance(graph, source, split, family, graph_seed, p))
                break
            else:
                raise RuntimeError("duplicate rejection budget exhausted")
        splits[split] = instances
    return splits


def manifest_entry(instance: GraphInstance) -> dict:
    graph = instance.graph
    return {
        "split": instance.split, "family": instance.family, "seed": instance.seed,
        "source": instance.source, "nodes": len(graph), "edges": graph.number_of_edges(),
        "diameter": nx.diameter(graph), "degrees": sorted(dict(graph.degree()).values()),
        "p": instance.p if instance.family == "erdos_renyi" else None,
        "rows": 2 if instance.family == "grid" else None,
    }


@torch.no_grad()
def evaluate(model: ShortestPathGNN, queries: list[Query]) -> dict:
    model.eval()
    graph_scores = []
    buckets = {"hop": defaultdict(list), "distance": defaultdict(list)}
    for query in queries:
        mask = ~query.source_mask
        prediction = model(query.source_mask, query.edge_index, query.edge_weight)
        error = (prediction[mask] - query.target[mask]).abs() * DISTANCE_SCALE
        distance = query.target[mask] * DISTANCE_SCALE
        graph_scores.append({
            "mae": error.mean().item(), "rmse": error.square().mean().sqrt().item(),
            "normalized_mae": (error.sum() / distance.sum()).item(),
        })
        for hop, target, absolute_error in zip(
            query.hops[mask].tolist(), distance.round().tolist(), error.tolist(), strict=True
        ):
            buckets["hop"][hop].append(absolute_error)
            buckets["distance"][int((target - 1) // 5)].append(absolute_error)
    return {
        **{name: sum(scores[name] for scores in graph_scores) / len(graph_scores)
           for name in graph_scores[0]},
        "graphs": len(graph_scores),
        **{name: {
            (str(key) if name == "hop" else f"{5 * key + 1}-{5 * key + 5}"):
            {"nodes": len(errors), "mae": sum(errors) / len(errors)}
            for key, errors in sorted(groups.items())
        } for name, groups in buckets.items()},
    }


def train(
    training: list[Query], validation: list[Query], *,
    residual: bool, seed: int, epochs: int, depth: int = DEPTH, width: int = WIDTH,
) -> tuple[ShortestPathGNN, list[dict]]:
    """Fit with one Adam step per graph and select by validation MAE."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model = ShortestPathGNN(width=width, depth=depth, residual=residual, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = Random(seed)
    best_mae = math.inf
    best_state = None
    history = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for query in rng.sample(training, len(training)):
            optimizer.zero_grad()
            prediction = model(query.source_mask, query.edge_index, query.edge_weight)
            mask = ~query.source_mask
            loss = (prediction[mask] - query.target[mask]).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        validation_mae = evaluate(model, validation)["mae"]
        if not math.isfinite(validation_mae):
            raise FloatingPointError("non-finite validation MAE")
        history.append({"train_mae": DISTANCE_SCALE * total_loss / len(training),
                        "validation_mae": validation_mae})
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(1)
    splits = make_splits(args.data_seed)
    queries = {split: [encode_query(item.graph, item.source) for item in instances]
               for split, instances in splits.items()}
    with args.output.open("x", encoding="utf-8") as stream:
        def record(entry: dict) -> None:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()

        record({
            "python": platform.python_version(), "torch": torch.__version__,
            "networkx": nx.__version__, "device": "cpu", "threads": 1,
            "data_seed": args.data_seed, "training_seeds": args.seeds, "epochs": args.epochs,
            "width": WIDTH, "depth": DEPTH, "learning_rate": LEARNING_RATE,
            "optimizer": "Adam", "loss": "non-source MAE on distances divided by ten",
            "updates_per_epoch": SPLITS["train"], "weight_law": "uniform integers 1-10",
            "max_connectivity_attempts": 100,
            "manifest": [manifest_entry(item) for group in splits.values() for item in group],
        })
        for seed in args.seeds:
            for residual in (False, True):
                model, history = train(queries["train"], queries["validation"],
                                       residual=residual, seed=seed, epochs=args.epochs)
                scores = {split: evaluate(model, group) for split, group in queries.items()
                          if split != "train"}
                record({"seed": seed, "residual": residual, "history": history,
                        "best_epoch": min(range(len(history)),
                                          key=lambda i: history[i]["validation_mae"]) + 1,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "metrics": scores})
                label = "residual" if residual else "standard"
                print(f"{seed} {label:8} val={scores['validation']['mae']:.3f} "
                      f"test={scores['test']['mae']:.3f} larger={scores['larger']['mae']:.3f}",
                      flush=True)


if __name__ == "__main__":
    main()
