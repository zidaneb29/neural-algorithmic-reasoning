# Project status

## Phase and objective

Phases 1–4 are complete. The project studies how graph size, topology, depth, and residual connections affect prediction of weighted shortest-path distances on unseen graphs.

## Completed work and key files

- [graphs.py](src/gnn_generalization/graphs.py) generates seeded paths, cycles, grids, Erdős–Rényi graphs, and random regular graphs.
- [baseline.py](src/gnn_generalization/baseline.py) computes exact single-source distances with NetworkX Dijkstra.
- [tests/](tests/) checks graph invariants, reproducibility, and label correctness. The Phase 1 suite passed all 26 tests on Python 3.12.
- [RESEARCH_PLAN.md](RESEARCH_PLAN.md) records the locality proof, hypotheses, and proposed evaluation protocol. [README.md](README.md) covers setup and usage.

## Decisions and limitations

- Use simple, connected, undirected graphs, a uniformly sampled source, and integer edge weights from 1 to 10 for exact labels. Bounded rejection sampling conditions random graph families on connectivity and can exhaust its attempt budget.
- Hop distance governs source information propagation under local message passing. Weighted distance is the prediction target. Final labels provide supervision for prediction, while algorithm execution requires separate evidence.
- Split by weighted graph instance and keep all its source queries and augmentations together. Repeated paths, cycles, and fixed-shape grids at a given size share a topology.

## Known issues and next steps

- No known baseline correctness issues. Phase 2 produces graph manifests and checks weighted isomorphism across all splits before encoding queries.
-

## Phase 2 implementation

- `data.py` encodes one source flag per node, reciprocal edge weights divided by ten, and Dijkstra distances divided by ten. Hop counts are evaluation metadata.
- `models.py` implements two width-32, depth-four pilot variants with identical trainable components and paired initialization. The residual variant adds the incoming node state at every layer.
- `pilot.py` uses one Adam step per graph with non-source MAE and selects the epoch by graph-macro validation MAE. The default budget is 768 updates per model and seed.
- The pilot uses 64/16/16 connected Erdős–Rényi graphs for train/validation/test at 6 or 8 nodes, 16 larger draws at 12 or 14 nodes, and four graphs each for the path, cycle and grid probes. The Erdős–Rényi parameter gives expected degree three before conditioning on connectivity.

## Initial pilot results

The five paired training seeds (0–4) shared one graph manifest (data seed 17). Each model had 21,217 parameters. All 32 tests passed. The run used Python 3.12.14, PyTorch 2.15.0+cpu and NetworkX 3.4.2. See `runs/phase2-pilot.jsonl` for the manifest, per-epoch curves and bucket counts, and `runs/phase2-pilot.log` for console output. PyTorch emitted a NumPy interoperability warning. The pilot uses PyTorch tensors directly.

| Split | Standard MAE | Residual MAE | Paired difference, mean ± sample SD |
| --- | ---: | ---: | ---: |
| Validation | 2.782 | 2.896 | +0.115 ± 0.365 |
| In-distribution test | 2.428 | 2.553 | +0.125 ± 0.521 |
| Larger graphs | 2.633 | 2.791 | +0.159 ± 0.252 |
| Path probe | 8.985 | 9.334 | +0.349 ± 1.275 |
| Cycle probe | 2.342 | 2.533 | +0.191 ± 0.475 |
| Grid probe | 3.013 | 2.998 | −0.014 ± 0.320 |

Differences are residual minus standard. Mean normalized MAE was 0.305 versus 0.327 on the in-distribution test and 0.320 versus 0.338 on larger graphs. The seed variability reflects training on one fixed data draw. Only one in-distribution test node, zero larger-graph nodes and six path-probe nodes were beyond four hops. The size probe therefore provides little evidence about source reach. These measurements characterize final-distance regression under the pilot protocol.

## Phase 3 audit

Phase 3 is authorized. The [Phase 2 audit](PHASE3_AUDIT.md) verified the archived manifest, split independence and encoding. The original suite passed all 32 tests. The new runs use separate output paths.

## Phase 3 results

Phase 3 is complete. The two-seed pilot verified the protocol, then five matched replicates compared 2, 4 and 8 message-passing rounds across 8–64 vertices and four families. The [results report](PHASE3_RESULTS.md) links numerical outputs, paired summaries, figures and limitations. The exact locality pairs passed, and all 36 tests passed. The Phase 2 reference JSONL retained its original checksum.

## Phase 4 theory and validation

[Phase 4 theory](PHASE4_THEORY.md) proves the attributed-neighborhood locality result for the implemented standard and residual models and derives the shortest-path error bound. [Phase 4 results](PHASE4_RESULTS.md) records a four-fit pilot and 20 controlled fits on 24-vertex weighted paths. All 24 locality checks and all 36 tests passed. Depth 8 lowered average error under both training regimes with near-matched parameter counts and dense-operation budgets. Endpoint training improved the longest-distance endpoint band while worsening overall center-query error. Scope is limited to one path topology and the tested optimizer budget. Phase 4 is complete.

## Phase 5 paper and finalization

The [scientific audit](paper/AUDIT.md), [LaTeX manuscript](paper/main.tex), archive-derived tables and plots, and [final report](paper/FINAL_REPORT.md) are complete. All 36 tests passed. The paper data script checked metrics from saved predictions, verified 50 main-run locality pairs, and confirmed the original archive checksums. Three Phase 3 table entries were corrected by 0.01 weight units. The README documents reproduction. PDF compilation awaits a LaTeX toolchain, and licensing remains a rights-holder decision.

The [publication readiness review](paper/PUBLICATION_READINESS.md) moved generated table row definitions outside the LaTeX alignments and clarified the scope of the research claims and replay instructions. All 36 tests, 34 archive checksums, 50 locality pairs and five generated data payloads were reverified. The final PDF build and visual inspection remain pending.

The publication-readiness revision subsequently compiled in Overleaf and passed independent visual review. The follow-up formatting revision displays the evaluation formulas and adjusts float placement around the methodology and proofs. This revision awaits an Overleaf layout check.

## GitHub publication preparation

The final formatting revision compiled and passed visual review in Overleaf, as confirmed by the project owner. The repository includes the newest local Overleaf export and a figure from the archived Phase 4 results. Publication preparation verified 36 tests, dependency consistency, 34 archive hashes, 50 locality pairs, and exact regeneration of all five paper fragments apart from line endings. The initial Git commit and publication remain pending.

## Initial release setup

The 37-file release candidate matches the previously proposed path list. The earlier summary counted it as 36. A targeted privacy scan found no matches in the proposed text files, and the paper PDF still matches the selected Overleaf export. The copyright holder approved the scoped MIT license for the original Python source, tests, and scripts and confirmed the public Git identity. The 38-file release includes the license. Public release remains pending approval.
