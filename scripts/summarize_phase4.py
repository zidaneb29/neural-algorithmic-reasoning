"""Summarize paired Phase 4 test errors and draw the endpoint contrasts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

BANDS = ("all", "1-4", "5-8", "9-12", "13+")


def summarize(path: Path) -> None:
    fits = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] != "fit":
            continue
        for probe, metrics in row["metrics"].items():
            errors = defaultdict(list)
            for graph in metrics["graphs"]:
                for node in graph["nodes"]:
                    hop = node["hop"]
                    band = BANDS[1 + sum(hop > edge for edge in (4, 8, 12))]
                    errors[band].append(abs(node["prediction"] - node["target"]))
            fits[row["seed"], row["depth"], row["condition"], probe] = {
                band: mean(values) for band, values in errors.items()
            } | {"all": metrics["mae"]}
    seeds = sorted({key[0] for key in fits})
    panels = []
    for probe in ("center", "endpoint"):
        for title, first, second in (
            ("Depth, center training", (4, "center"), (8, "center")),
            ("Depth, endpoint training", (4, "endpoint"), (8, "endpoint")),
            ("Training source, depth 4", (4, "center"), (4, "endpoint")),
            ("Training source, depth 8", (8, "center"), (8, "endpoint")),
        ):
            print(probe, title)
            contrasts = {}
            for band in BANDS:
                if band not in fits[seeds[0], *first, probe]:
                    continue
                a = [fits[seed, *first, probe][band] for seed in seeds]
                b = [fits[seed, *second, probe][band] for seed in seeds]
                delta = [y - x for x, y in zip(a, b, strict=True)]
                spread = stdev(delta) if len(delta) > 1 else 0.0
                contrasts[band] = (mean(delta), spread)
                print(f"  {band:>4} {mean(a):6.2f} {mean(b):6.2f} "
                      f"{mean(delta):+6.2f} +/- {spread:.2f}")
            if probe == "endpoint":
                panels.append((title, contrasts))
    limit = max(abs(m) + sd for _, values in panels for m, sd in values.values())
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="530" '
           'viewBox="0 0 760 530" font-family="sans-serif">',
           '<rect width="760" height="530" fill="white"/>',
           '<text x="28" y="34" font-size="21">Endpoint test | Paired MAE changes</text>',
           '<text x="28" y="60" font-size="14">Weight units, mean ± sample SD across seeds</text>']
    for index, (title, values) in enumerate(panels):
        left = 28 + (index % 2) * 370
        top = 95 + (index // 2) * 215
        zero = left + 185
        scale = 125 / limit
        svg.extend([
            f'<text x="{left}" y="{top}" font-size="16">{title}</text>',
            f'<line x1="{zero}" x2="{zero}" y1="{top + 14}" y2="{top + 165}" stroke="#aaa"/>',
            f'<text x="{zero - 4}" y="{top + 183}" font-size="13">0</text>',
        ])
        for row, (band, (delta, sd)) in enumerate(values.items()):
            y = top + 28 + row * 29
            x = zero + scale * delta
            svg.extend([
                f'<text x="{left}" y="{y + 4}" font-size="13">{band}</text>',
                f'<line x1="{x-scale*sd}" x2="{x+scale*sd}" y1="{y}" y2="{y}" stroke="#245f90"/>',
                f'<circle cx="{x}" cy="{y}" r="4" fill="#245f90"/>',
            ])
    svg.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    summarize(parser.parse_args().path)
