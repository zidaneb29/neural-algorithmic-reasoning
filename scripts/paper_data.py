"""Recompute paper metrics and generate LaTeX data from saved predictions."""

import argparse
import hashlib
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev


def load(path: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    fits = [r for r in rows if r["kind"] == "fit"]
    assert len(fits) == count
    for row in fits:
        pair = row["locality"]
        a, b = pair["targets"]
        p, q = pair["predictions"]
        assert (a, b) == (row["depth"] + 1, row["depth"] + 2)
        assert p == q and max(abs(a - p), abs(b - q)) >= 0.5
    for row in rows:
        for cell in row.get("metrics", {}).values():
            for graph in cell["graphs"]:
                errors = [abs(n["prediction"] - n["target"]) for n in graph["nodes"]]
                total = sum(n["target"] for n in graph["nodes"])
                calculated = (mean(errors), sqrt(mean(e * e for e in errors)), sum(errors) / total)
                for key, value in zip(("mae", "rmse", "normalized_mae"), calculated, strict=True):
                    assert isclose(graph[key], value, rel_tol=1e-6, abs_tol=1e-6), (path, key)
            for key in ("mae", "rmse", "normalized_mae"):
                assert isclose(cell[key], mean(g[key] for g in cell["graphs"]), abs_tol=1e-10)
    return rows


def generate(p3: list[dict], p4: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, lines: list[str]) -> None:
        (out / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="ascii")

    def save_rows(name: str, command: str, rows: list[str]) -> None:
        # Suppress end-of-line space tokens between rows and before the rule.
        save(name, [rf"\newcommand{{\{command}}}{{%", *(row + "%" for row in rows), "}"])

    end = r" \\"
    cells = {(r["seed"], r["model"], r.get("depth"), split): cell
             for r in p3 if r["kind"] in ("fit", "baseline")
             for split, cell in r["metrics"].items()}

    def scores(split: str, model: str, depth: int | None, metric: str = "mae") -> list[float]:
        return [cells[s, model, depth, split][metric] for s in p3[0]["seeds"]]

    sizes = p3[0]["sizes"]
    table = []
    for size in sizes:
        split = f"erdos_renyi/{size}"
        nodes = [n for s in p3[0]["seeds"] for g in cells[s, "standard", 4, split]["graphs"]
                 for n in g["nodes"]]
        mae, rmse, normalized = (mean(scores(split, "standard", 4, key))
                                 for key in ("mae", "rmse", "normalized_mae"))
        table.append(f"{size} & {sum(n['hop'] > 4 for n in nodes):,}/{len(nodes):,} & "
                     f"{mae:.2f} & {rmse:.2f} & {normalized:.3f}" + end)
    save_rows("size-rows", "SizeRows", table)

    table = []
    for family, label in (("erdos_renyi", r"Erd\H{o}s--R\'enyi"), ("grid", "Grid"),
                          ("cycle", "Cycle"), ("path", "Path")):
        split = f"{family}/64"
        standard, residual = (scores(split, model, 8) for model in ("standard", "residual"))
        delta = [b - a for a, b in zip(standard, residual, strict=True)]
        baseline = mean(scores(split, "training_mean", None))
        table.append(f"{label} & {mean(standard):.2f} & {mean(residual):.2f} & "
                     f"{baseline:.2f} & " + "$" + rf"{mean(delta):+.2f}\pm{stdev(delta):.2f}$" + end)
    save_rows("family-rows", "FamilyRows", table)

    plot = []
    for model, style in (("standard", "solid"), ("residual", "dashed")):
        for depth, mark in ((2, "*"), (4, "square*"), (8, "triangle*")):
            coords = " ".join(f"({n},{mean(scores(f'erdos_renyi/{n}', model, depth)):.6f})"
                              for n in sizes)
            plot.extend([rf"\addplot+[{style},mark={mark}] coordinates {{{coords}}};",
                         rf"\addlegendentry{{{model.capitalize()} {depth}}}"])
    save("size-plot", plot)

    paths = {(r["seed"], r["condition"], r["depth"], probe): cell
             for r in p4 if r["kind"] == "fit" for probe, cell in r["metrics"].items()}

    def path_scores(condition: str, depth: int, probe: str, low: int = 0,
                    high: int = 23) -> list[float]:
        cells_for_probe = [paths[s, condition, depth, probe] for s in p4[0]["seeds"]]
        if low == 0:
            return [cell["mae"] for cell in cells_for_probe]
        return [mean(abs(n["prediction"] - n["target"]) for g in cell["graphs"]
                     for n in g["nodes"] if low <= n["hop"] <= high)
                for cell in cells_for_probe]

    settings = (("center", 4), ("center", 8), ("endpoint", 4), ("endpoint", 8))
    save_rows("path-rows", "PathRows", [" & ".join([probe.capitalize()] +
         [f"{mean(path_scores(*setting, probe)):.2f}" for setting in settings]) + end
         for probe in ("center", "endpoint")])
    plot = []
    for depth, mark in ((4, "*"), (8, "square*")):
        coords = []
        for i, (low, high) in enumerate(((1, 4), (5, 8), (9, 12), (13, 23)), 1):
            a, b = (path_scores(c, depth, "endpoint", low, high) for c in ("center", "endpoint"))
            delta = [y - x for x, y in zip(a, b, strict=True)]
            x = i + (-0.1 if depth == 4 else 0.1)
            coords.append(f"({x},{mean(delta):.6f}) +- (0,{stdev(delta):.6f})")
        plot.extend([rf"\addplot+[only marks,mark={mark},error bars/.cd,y dir=both,y explicit]"
                     + " coordinates {" + " ".join(coords) + "};",
                     rf"\addlegendentry{{Depth {depth}}}"])
    save("path-plot", plot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase3", type=Path)
    parser.add_argument("phase4", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive-manifest", type=Path)
    args = parser.parse_args()
    if args.archive_manifest:
        for line in args.archive_manifest.read_text(encoding="ascii").splitlines():
            expected, name = line.split("  ", 1)
            with Path(name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Archive checksum mismatch: {name}")
        print("All archived file hashes match.")
    generate(load(args.phase3, 30), load(args.phase4, 20), args.output)
    print("Verified per-graph and macro metrics and 50 main-run locality pairs.")


if __name__ == "__main__":
    main()
