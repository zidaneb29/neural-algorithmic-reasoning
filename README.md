# Neural Algorithmic Reasoning

This undergraduate research project studies where graph neural networks succeed and fail at predicting weighted shortest-path distances. It combines controlled experiments on synthetic graphs with a proof of an architectural limit.

## Research question and approach

How do graph size, topology, and message-passing depth affect predictions on unseen graphs? NetworkX Dijkstra computes exact distances on graphs with positive integer edge weights. Standard and residual PyTorch models learn from final distances. The target minimizes the sum of edge weights, while hop distance counts edges and governs how many rounds a source signal needs to reach a node. Splits are made by graph before source queries are added.

[Read the paper (PDF)](paper/neural-algorithmic-reasoning.pdf) · [Paper source](paper/main.tex) · [Build instructions](paper/README.md) · [Scientific audit](paper/AUDIT.md) · [Final report](paper/FINAL_REPORT.md) · [Publication readiness](paper/PUBLICATION_READINESS.md)

## Main findings

- On connected Erdős–Rényi graphs, raw MAE rose across sizes for all six models. Standard depth-four normalized MAE rose from 0.323 at 8 vertices to 0.407 at 64. Diameter, hop coverage and target scale changed together.
- On 24-vertex paths, depth eight lowered overall mean error on both probes under both training regimes, with close parameter and dense-operation counts and higher measured runtime. Endpoint training improved endpoint MAE and worsened center MAE.
- Residual updates gave no consistent benefit under the tested protocol.

See [Phase 3 results](docs/size-and-topology-results.md), [Phase 4 theory](docs/message-passing-locality.md) and [Phase 4 results](docs/path-validation-results.md).

## Theoretical locality result

A synchronous $k$-round message-passing model with a nodewise readout and a source indicator as its only source-dependent input gives the same root prediction for two sources outside the root's $k$-hop neighborhood on a fixed graph. On a unit-weight path, choosing sources at hops $k+1$ and $k+2$ makes their exact distances differ by one. The shared prediction must incur at least 0.5 weight units of absolute error on one query, for any model parameters. See the [proof and assumptions](docs/message-passing-locality.md).

## Visual results

![Paired changes in endpoint-test MAE by hop band on 24-vertex paths](paper/figures/phase4-main.svg)

The plot shows mean paired MAE changes and sample standard deviations across five replicates. Negative values mean lower error for depth eight versus depth four, or for endpoint versus center training. It comes from the archived Phase 4 main run via [the Phase 4 summarizer](scripts/summarize_phase4.py).

## Repository structure

| Path | Contents |
| --- | --- |
| `src/gnn_generalization/` | Graph generation, exact baselines, models, and experiment entry points |
| `tests/` | Automated graph, baseline, model, and experiment checks |
| `scripts/` | Experiment summarizers and paper data generator |
| `paper/` | Reviewed PDF, LaTeX source, generated tables and plots, and archive checksums |
| `docs/` | Pilot audit, detailed experiment results, and locality proof |

## Installation and tests

Use Python 3.11 or later. The archived main runs used Python 3.12.14, PyTorch 2.15.0+cpu and NetworkX 3.4.2 with one CPU thread. Verification used these versions and pytest 8.3.5. The training extra permits compatible PyTorch builds. Exact numerical replay across operating systems, hardware and package versions is not guaranteed, even with the same seeds.

Create a virtual environment with **python -m venv .venv** and activate it using **.venv\Scripts\Activate.ps1** in PowerShell or **source .venv/bin/activate** in a POSIX shell.

~~~sh
python -m pip install -e ".[test,train]"
python -m pytest -q
~~~

The test extra alone suffices for the graph and exact-baseline code.

## Reproduce the experiments

From the repository root, create a fresh output directory with **mkdir replay**, then run:

~~~sh
python -m gnn_generalization.phase3 --output replay/phase3-main.jsonl --seeds 0 1 2 3 4 --epochs 12 --train-graphs 64 --eval-graphs 16
python scripts/summarize_phase3.py replay/phase3-main.jsonl
python -m gnn_generalization.phase4 replay/phase4-main.jsonl
python scripts/summarize_phase4.py replay/phase4-main.jsonl
python scripts/paper_data.py replay/phase3-main.jsonl replay/phase4-main.jsonl replay/paper-data
~~~

Phase 3 runs 30 fits and Phase 4 runs 20. The original pilot protocols can also be replayed:

~~~sh
python -m gnn_generalization.pilot --output replay/phase2-pilot.jsonl
python -m gnn_generalization.phase3 --output replay/phase3-pilot.jsonl --seeds 90 91 --epochs 2 --train-graphs 16 --eval-graphs 4
python -m gnn_generalization.phase4 replay/phase4-pilot.jsonl --pilot
~~~

Use fresh filenames for repeated runs. JSONL records preserve graph manifests, seeds, versions, curves, and, for the main studies, per-node predictions and locality pairs. Summarizers write scores and plots beside their input. The original runs directory is local and Git-ignored. Historical report paths into that directory require the archive, while the commands above regenerate equivalent artifact types. Compact paper data fragments are included with the source. See [paper verification](paper/README.md) for archive hashes.

## Limitations and future work

The five paired replicates support descriptive comparisons on synthetic graphs. The depth comparison also varies width and runtime, and changing the source policy shifts both hop distances and weighted targets. Final-distance supervision leaves internal execution of Dijkstra or Bellman–Ford untested. A follow-up could add algorithm-trace supervision or test graph families that vary hop coverage independently of size and distance scale.

## Licensing

The original Python source, tests, and scripts are available under the [MIT License](LICENSE). The research paper, LaTeX source, figures, and research documentation retain their separate copyright terms. Any future distribution of experiment data will have its own terms. Third-party materials retain their respective terms.
