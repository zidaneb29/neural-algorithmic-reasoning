"""Summarize a completed Phase 3 JSONL and render standalone SVG figures."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ("mae", "rmse", "normalized_mae")
COLORS = ("#1065a6", "#bb4027", "#208066", "#7856a3", "#ac6500", "#434b53")


def summarize(rows: list[dict], output: Path) -> tuple[dict, dict]:
    cells = defaultdict(dict)
    curves = defaultdict(list)
    for row in rows:
        if row["kind"] not in {"fit", "baseline"}:
            continue
        model = row["model"] + (f'-{row["depth"]}' if "depth" in row else "")
        seed = row["seed"]
        for split, scores in row["metrics"].items():
            for metric in METRICS:
                cells[split, "all", model, metric][seed] = scores[metric]
            for hop, bucket in scores["hop"].items():
                for metric in ("mae", "rmse"):
                    cells[split, hop, model, metric][seed] = bucket[metric]
            nodes = [node for graph in scores["graphs"] for node in graph["nodes"]]
            distance_errors = defaultdict(list)
            for node in nodes:
                bucket = (int(node["target"]) - 1) // 5
                distance_errors[bucket].append(abs(node["prediction"] - node["target"]))
            for bucket, errors in distance_errors.items():
                label = f"weight:{5 * bucket + 1}-{5 * bucket + 5}"
                cells[split, label, model, "mae"][seed] = mean(errors)
                cells[split, label, model, "rmse"][seed] = mean(e**2 for e in errors)**0.5
            for label, lower, upper in (("1-2", 1, 2), ("3-4", 3, 4),
                                        ("5-8", 5, 8), ("9+", 9, float("inf"))):
                errors = [abs(node["prediction"] - node["target"]) for node in nodes
                          if lower <= node["hop"] <= upper]
                if errors:
                    cells[split, label, model, "mae"][seed] = mean(errors)
                    cells[split, label, model, "rmse"][seed] = mean(e**2 for e in errors)**0.5
        for epoch, scores in enumerate(row.get("history", []), 1):
            for metric, value in scores.items():
                curves[model, epoch, metric].append(value)

    def stats(values: list[float]) -> tuple:
        return len(values), mean(values), stdev(values) if len(values) > 1 else ""

    with output.with_suffix(".scores.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("split", "stratum", "model", "metric", "seeds", "mean", "sample_sd"))
        for key, values in sorted(cells.items()):
            writer.writerow((*key, *stats(list(values.values()))))
    comparisons = [
        (f"residual-{d}", f"standard-{d}") for d in (2, 4, 8)
    ] + [(f"{arch}-{a}", f"{arch}-{b}") for arch in ("standard", "residual")
         for a, b in ((4, 2), (8, 4), (8, 2))]
    with output.with_suffix(".paired.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("split", "stratum", "first", "second", "metric", "seeds",
                         "mean_first_minus_second", "sample_sd", "seed_differences"))
        for (split, hop, model, metric), values in sorted(cells.items()):
            for first, second in comparisons:
                if model != first:
                    continue
                other = cells[split, hop, second, metric]
                diffs = {seed: value - other[seed] for seed, value in values.items() if seed in other}
                if diffs:
                    writer.writerow((split, hop, first, second, metric,
                                     *stats(list(diffs.values())), json.dumps(diffs)))
    with output.with_suffix(".curves.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("model", "epoch", "metric", "seeds", "mean", "sample_sd"))
        for key, values in sorted(curves.items()):
            writer.writerow((*key, *stats(values)))
    return cells, curves


def plot(path: Path, panels: list, labels: list[str], xlabel: str) -> None:
    width, height = 560, 315
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width*columns}" height="{height*rows+55}" '
        f'viewBox="0 0 {width*columns} {height*rows+55}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" font-size="13" fill="#19232e">',
    ]
    for index, (title, ticks, series) in enumerate(panels):
        svg.append(f'<g transform="translate({width*(index%columns)},{height*(index//columns)})">')
        left, right, top, bottom = 79, 533, 41, 251
        maximum = max(max(values) for values in series) * 1.10
        svg.append(f'<text x="79" y="25" font-size="17" font-weight="bold">{title}</text>')
        for i in range(5):
            y = bottom - i * (bottom - top) / 4
            svg.append(f'<path d="M {left} {y} H {right}" stroke="#d5dce2"/>')
            svg.append(f'<text x="70" y="{y+4}" text-anchor="end">{maximum*i/4:.1f}</text>')
        xs = [left + i * (right - left) / max(1, len(ticks) - 1) for i in range(len(ticks))]
        for x, label in zip(xs, ticks, strict=True):
            svg.append(f'<text x="{x}" y="272" text-anchor="middle">{label}</text>')
        svg.extend([
            f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#19232e"/>',
            f'<text x="306" y="296" text-anchor="middle">{xlabel}</text>',
            '<text transform="translate(23,146) rotate(-90)" text-anchor="middle">MAE (weight units)</text>',
        ])
        for values, color in zip(series, COLORS, strict=False):
            ys = [bottom - value / maximum * (bottom - top) for value in values]
            points = list(zip(xs, ys, strict=True))
            svg.append('<polyline fill="none" stroke-width="2.5" '
                       f'stroke="{color}" points="' + " ".join(f"{x},{y}" for x, y in points) + '"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        svg.append("</g>")
    for index, (label, color) in enumerate(zip(labels, COLORS, strict=True)):
        x = 82 + (index % 3) * 153
        y = rows * height + 13 + (index // 3) * 20
        svg.append(f'<circle cx="{x}" cy="{y-4}" r="4" fill="{color}"/>'
                   f'<text x="{x+10}" y="{y}">{label}</text>')
    svg.extend(["</g>", "</svg>"])
    with path.open("x", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    config = rows[0]
    assert sum(row["kind"] == "fit" for row in rows) == len(config["seeds"]) * 6
    cells, curves = summarize(rows, args.input)
    models = [f"{arch}-{depth}" for arch in ("standard", "residual") for depth in (2, 4, 8)]
    sizes = config["sizes"]
    families = {"erdos_renyi": "Erdos–Renyi", "path": "Path", "cycle": "Cycle", "grid": "Grid"}
    plot(args.input.with_suffix(".size.svg"), [
        (label, sizes, [[mean(cells[f"{family}/{n}", "all", model, "mae"].values())
                         for n in sizes] for model in models])
        for family, label in families.items()
    ], models, "Graph size (vertices, doubling scale)")
    bins = ("1-2", "3-4", "5-8", "9+")
    plot(args.input.with_suffix(".hop.svg"), [
        (f"{label}, 64 vertices", [b for b in bins if cells[f"{family}/64", b, models[0], "mae"]],
         [[mean(cells[f"{family}/64", b, model, "mae"].values())
           for b in bins if cells[f"{family}/64", b, model, "mae"]] for model in models])
        for family, label in families.items()
    ], models, "Unweighted hop-distance bin")
    epochs = range(1, config["epochs"] + 1)
    plot(args.input.with_suffix(".training.svg"), [
        (label, epochs, [[mean(curves[m, e, metric]) for e in epochs] for m in models])
        for label, metric in (("Online training loss", "train_mae"),
                              ("Validation error", "validation_mae"))
    ], models, "Epoch")


if __name__ == "__main__":
    main()
