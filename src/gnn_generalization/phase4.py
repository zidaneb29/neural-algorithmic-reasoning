"""Controlled depth and training source-location experiment on weighted paths."""

import argparse
import json
from pathlib import Path
import platform
from random import Random
from time import perf_counter

import networkx as nx
import torch

from .data import DISTANCE_SCALE, encode_query
from .graphs import generate_graph
from .phase3 import evaluate_predictions, locality_pair
from .pilot import LEARNING_RATE, train

DEPTH_WIDTH = ((4, 32), (8, 23))


def run(output: Path, pilot: bool) -> None:
    torch.set_num_threads(1)
    seeds = (90,) if pilot else (0, 1, 2, 3, 4)
    train_count, eval_count, epochs = (8, 4, 2) if pilot else (32, 16, 20)
    seen = set()
    with output.open("x", encoding="utf-8") as stream:
        def record(row: dict) -> None:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        record(dict(kind="config", seeds=seeds, depth_width=DEPTH_WIDTH, epochs=epochs,
                    train_count=train_count, eval_count=eval_count, vertices=24,
                    weights="uniform integers 1-10", sources=[11, 0], scale=DISTANCE_SCALE,
                    python=platform.python_version(), torch=torch.__version__,
                    networkx=nx.__version__, device="cpu", threads=1,
                    optimizer="Adam", lr=LEARNING_RATE))
        for seed in seeds:
            data_seed = 4000 + seed
            rng = Random(data_seed)
            groups = {}
            for split, count in (("train", train_count), ("validation", eval_count),
                                 ("test", eval_count)):
                entries = []
                while len(entries) < count:
                    graph_seed = rng.randrange(2**32)
                    graph, _ = generate_graph("path", 24, seed=graph_seed)
                    weights = tuple(graph[i][i + 1]["weight"] for i in range(23))
                    key = min(weights, weights[::-1])
                    if key not in seen:
                        seen.add(key)
                        entries.append((graph_seed, graph, weights))
                groups[split] = entries
            record(dict(kind="manifest", seed=seed, data_seed=data_seed,
                        groups={split: [(s, w) for s, _, w in entries]
                                for split, entries in groups.items()}))
            queries = {
                split: {name: [encode_query(graph, source) for _, graph, _ in entries]
                        for name, source in (("center", 11), ("endpoint", 0))}
                for split, entries in groups.items()
            }
            validation = queries["validation"]["center"] + queries["validation"]["endpoint"]
            for depth, width in DEPTH_WIDTH:
                for condition, training in queries["train"].items():
                    start = perf_counter()
                    model, history = train(training, validation, residual=False, seed=seed,
                                           epochs=epochs, depth=depth, width=width)
                    record(dict(kind="fit", seed=seed, depth=depth, width=width,
                                condition=condition, history=history,
                                parameters=sum(p.numel() for p in model.parameters()),
                                locality=locality_pair(model, depth),
                                metrics={name: evaluate_predictions(test, model)
                                         for name, test in queries["test"].items()},
                                seconds=perf_counter() - start))
                    print(f"{seed} depth={depth} train={condition} "
                          f"val={min(e['validation_mae'] for e in history):.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.output, args.pilot)
