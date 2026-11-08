# Phase 3 generalization experiments

[Phase 2 audit](phase2-pilot-audit.md)

## Hypotheses and design

The size hypothesis predicts increasing error within the connected Erdős–Rényi family as evaluation size increases. The topology hypothesis predicts larger errors on sparse, long-hop families at matched size. The depth hypothesis predicts larger improvements at newly reachable hops. The residual hypothesis predicts better validation and OOD error under a fixed number of updates.

Train and select models on separate connected Erdős–Rényi graphs at 6 and 8 vertices in equal proportions. Evaluate Erdős–Rényi, path, cycle and two-row grid families at 8, 16, 32 and 64 vertices. The 8-vertex Erdős–Rényi cell is the in-distribution size anchor. All graphs use the same integer weight law (uniform 1–10) and one uniform source. Erdős–Rényi uses p = 3/(n−1) before conditioning on connectivity. Record realized degrees, edge counts, unweighted diameter, edge weights, exact targets and source-to-node hops. Reject weighted-isomorphic duplicates within each replicate before query encoding. Fixed-family shapes repeat with different weights and sources.

Each replicate pairs standard and residual models at depths 2, 4 and 8. At each depth the architecture pair shares the initial parameters and graph order. Across depths the encoder and shared prefix layers start identically for a given seed, while readout initialization depends on how many layers precede it. Each fit uses width 32, Adam at 0.001, one update per graph, and scaled non-source MAE. Parameter counts are 10,657, 21,217 and 42,337. The number of updates is matched, while message-passing operations and capacity grow with depth. Use the earliest minimum graph-macro validation MAE to select the epoch. Baselines predict zero or the non-source node-pooled mean of training labels.

For overall scores, average non-source node errors within each graph and weight all graphs equally, including when averaging per-graph RMSE and per-graph normalized MAE. Within a hop or target-distance bin, pool nodes inside a replicate. Average replicate-level summaries equally across available replicates. Differences are paired within replicate and sample SD includes both graph-draw and training variation. Nodes within the same graph are dependent. Rare bins may have fewer than five contributing replicates. The training curve averages loss at each update before the optimizer step.

## Planned and realized protocol

### Original hypotheses and comparisons

| Question | Testable hypothesis | Comparison |
| --- | --- | --- |
| RQ1 Size | Graph-level normalized MAE increases on larger graphs. | Train on small graphs and compare ID test error to size-only OOD error within family. |
| RQ2 Structure | At matched sizes and similar edge counts, families with a larger fraction of nodes beyond the receptive field have higher normalized MAE. | Train on one family and test on held-out families. Report residual differences in density, degree, and diameter. |
| RQ3 Depth | Increasing depth reduces error more for nodes newly within reach than for nodes already within reach. | Compare the same graph-source pairs at several depths and stratify by hop distance. |
| RQ4 Residuals | Residual connections reduce validation loss at a fixed update budget and reduce OOD normalized MAE. | Pair data and initialization seeds across models. Compare training curves and held-out error separately. |

### Original protocol before training

Start with small connected random graphs and deterministic path, cycle, and grid probes. Fix the weight law across splits. Rejection sampling changes the topology distribution to connected draws. For size shifts, preselect a density policy such as approximately fixed expected degree and record the realized degree distribution, since conditioning can change it. Compare a size-only shift, a family-only shift at matched sizes, and optionally a combined shift. Family comparisons are observational with respect to diameter and degree.

Create disjoint graph-level train, validation, and test manifests with recorded seeds and settings. Keep all sources, relabelings, and augmentations of a weighted instance in one split. Audit duplicate weighted instances, including relabelings where practical, before training. Paths, cycles, and fixed-shape grids have a single topology for a given size and shape. Repeated samples at those settings test new weights and sources on the same topology. Reserve shapes or families when testing transfer to unseen topologies.

Report per-graph non-source MAE and RMSE and macro-average across graphs. Also report per-graph normalized MAE, defined as the sum of absolute errors divided by the sum of true non-source distances. Stratify by unweighted hops and weighted target distance and report bucket counts. Record graph size, edge count, unweighted diameter, degree distribution, weight law, depth, parameter count, and data and training seeds. Use validation alone for model selection. Report paired seed-level differences over at least five seeds and their variability. Treat graphs and training runs as experimental units for uncertainty, since nodes from the same graph are dependent.

The plan called for fixing sample sizes, split manifests, size bands, density policy, depth and width settings, aggregation and edge encoding, loss, optimizer, and compute budget and comparing equal-width architectures with identical trainable components and a parameter-free residual addition. Spectral analysis can follow the initial comparisons. Oversmoothing and oversquashing would require separate diagnostics.

### Planned versus realized comparisons

- **Size and topology.** The run tested within-family size shifts and matched-size family shifts, with larger structured cells combining both shifts relative to training. Edge counts were measured and differ across families. Diameter, degree, hop coverage, and target scale also vary, so the family comparisons are observational.
- **Depth.** The comparison stratified the same graph-source observations by hops and matched update counts. Increasing depth also increased parameter count and work per update.
- **Residuals.** Standard and residual models were paired by data and initialization at each depth. The report separates validation and held-out errors and retains paired variability.
- **Replication.** The Phase 2 pilot shared one graph draw across five training seeds. The Phase 3 main study used five paired data and training replicates. Spectral analysis, oversmoothing, and oversquashing were outside the realized comparisons.

## Pilot and locality check

Two matched seeds (90 and 91; data seeds 1090 and 1091) used 16 training graphs, four graphs per validation or evaluation cell and two epochs. All 12 fits completed. The two 64-vertex Erdős–Rényi cells contained 25 and 55 non-source nodes beyond four hops. Each 64-vertex path cell contained 220. The pilot verified coverage and scoring, and its short training budget was for protocol validation. The main run uses five fresh matched seeds (0–4; data seeds 1000–1004), 64 training graphs, 16 graphs per validation or evaluation cell and 12 epochs (768 updates).

On the same unit-weight path, place the source at vertex 5 or 6 and predict at vertex 0. For both choices the source flag vanishes throughout the four-hop neighborhood of vertex 0. The weighted edges are unchanged and the synchronous four-round updates propagate source information by at most one hop per round. The nodewise readout and the source-zero override at vertex 0 are identical in both cases. Both models therefore predict the same value for targets of 5 and 6, so at least one absolute error is at least 0.5 weight units. This holds for any learned weights under the implemented assumptions. The executable check uses sources at depth + 1 and depth + 2 for each tested depth, and compares the outputs exactly on CPU. The generalization evaluation uses random weights and sources independently of the constructed check.

## Main results

All 30 fits completed. A weighted-isomorphism audit across all 1,972 Phase 2, Phase 3 pilot, and Phase 3 main graphs found no duplicates. The following are five-replicate averages. Errors are in original edge-weight units unless normalized.

### Size and family

| Erdős–Rényi vertices | Mean diameter | Mean degree | Mean target | Nodes beyond 4 hops | Standard depth-4 MAE / RMSE | Standard depth-4 normalized MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3.25 | 3.04 | 8.66 | 4 / 560 | 2.83 / 3.30 | 0.323 |
| 16 | 4.99 | 3.20 | 11.58 | 50 / 1,200 | 4.01 / 4.79 | 0.326 |
| 32 | 6.49 | 3.26 | 13.97 | 241 / 2,480 | 5.02 / 6.09 | 0.342 |
| 64 | 8.26 | 3.26 | 17.10 | 1,134 / 5,040 | 7.35 / 8.58 | 0.407 |

Mean absolute error rose at every size for all six models in connected Erdős–Rényi graphs. Normalized error was higher at 64 than at 8 for all six, with some intermediate decreases. Graph diameter and target magnitudes increased alongside size, despite fairly stable realized degree.

At 8 vertices, standard depth-4 MAE was 2.83 on Erdős–Rényi, 2.76 on grids, 3.62 on cycles and 6.64 on paths. All cycle vertices were within four hops, while 120 of 560 path vertices exceeded four hops. The cycle had larger targets on average than Erdős–Rényi (11.61 versus 8.66). At 64 vertices, mean targets were 17.10 on Erdős–Rényi, 53.68 on grids, 90.14 on cycles and 120.13 on paths. Matched sizes thus exposed large differences in hop coverage, target magnitude and degree.

| 64-vertex family | Nodes beyond 4 hops | Standard depth-8 MAE / RMSE | Residual depth-8 MAE / RMSE | Training mean MAE / RMSE | Depth-8 residual minus standard MAE, mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Erdős–Rényi | 1,134 / 5,040 | 6.64 / 7.88 | 8.04 / 9.29 | 10.09 / 11.56 | +1.40 ± 1.69 |
| Grid | 3,910 / 5,040 | 41.62 / 52.54 | 43.91 / 54.69 | 46.52 / 57.01 | +2.28 ± 2.52 |
| Cycle | 4,400 / 5,040 | 76.78 / 91.58 | 78.88 / 93.53 | 82.87 / 97.05 | +2.10 ± 2.75 |
| Path | 4,422 / 5,040 | 106.78 / 130.84 | 108.83 / 132.67 | 112.86 / 136.14 | +2.05 ± 2.75 |

### Depth, hops and training

On 64-vertex Erdős–Rényi graphs, standard MAE at depths 2, 4 and 8 was 7.58, 7.35 and 6.64. Residual MAE was 7.86, 7.83 and 8.04. The residual comparison depended on depth and split. At depth 8 the residual-minus-standard validation difference was −0.05 ± 0.23, versus +1.40 ± 1.69 on 64-vertex Erdős–Rényi. Across all 64-vertex cells, mean depth-8 standard MAE was lower than depth-4 standard MAE, while depth-8 residual MAE was higher than depth-4 residual MAE.

| 64-vertex family | Standard depth 8 minus 4, hops 1–4 | Standard depth 8 minus 4, hops 5–8 | Standard depth 8 minus 4, hops 9+ |
| --- | ---: | ---: | ---: |
| Erdős–Rényi | −0.58 ± 2.41 | −1.21 ± 3.71 | +1.00 ± 1.15 (2 seeds) |
| Grid | −0.57 ± 1.36 | −1.40 ± 4.84 | −1.46 ± 4.81 |
| Cycle | −0.47 ± 1.06 | −1.31 ± 4.10 | −1.36 ± 4.18 |
| Path | −0.55 ± 1.14 | −1.38 ± 4.12 | −1.33 ± 4.14 |

These paired MAE differences are descriptive. For the three structured families, nodes beyond eight hops improved by about as much on average as nodes at five through eight hops even though the former still could not receive the source flag. Residual differences in the newly reachable band ranged from +0.27 to +0.54. The data do not isolate a receptive-field benefit. Local structural features, increased parameter count, and different optimization can all affect predictions beyond the source's reach. The fixed-graph locality pairs isolate the source's influence and passed for all 30 fits.

Mean online training MAE decreased from 3.73–4.37 in epoch 1 to 1.84–2.35 in epoch 12 across configurations. Mean validation MAE decreased from 2.95–3.78 to 2.04–2.50. Thirteen of 30 models selected the final epoch, so the budget may constrain some fits. Depth-8 standard and residual had the lowest selected validation MAE in their respective architecture families (1.82 and 1.77).

## Limits and Phase 4 questions

The results use five joint data and initialization replicates, a single optimizer budget, one source per graph and synthetic connected families. Repeated structured shapes assess transfer to new weights and sources on those shapes. Conditioning Erdős–Rényi graphs on connectivity changes their degree distribution. Diameter, distance scale, and degree differ across families, and the evaluation target ranges exceed training support. Width is fixed, so depth changes parameter count and work per update. Final-distance regression and a locality check cannot establish that a model executes Dijkstra.

Phase 4 questions include whether the depth effect persists with parameter- or compute-matched models, whether train graphs with longer hop and weighted-distance coverage improve extrapolation, and whether algorithm-trace supervision or a shared recurrent update transfers across more rounds. A factorial follow-up could separately vary data and initialization seeds and match target-distance bands and local neighborhoods across topology probes.

## Artifacts

- Pilot predictions, manifests and curves (`runs/phase3-pilot.jsonl`) and pilot log (`runs/phase3-pilot.log`)
- Main predictions, manifests and curves (`runs/phase3-main.jsonl`) and main log (`runs/phase3-main.log`)
- Size and stratum scores (`runs/phase3-main.scores.csv`), paired differences (`runs/phase3-main.paired.csv`), and training and validation summaries (`runs/phase3-main.curves.csv`)
- Size figure (`runs/phase3-main.size.svg`), hop figure (`runs/phase3-main.hop.svg`), and learning curves (`runs/phase3-main.training.svg`)
- Test log (`runs/phase3-suite.log`) and numerical cross-checks (`runs/phase3-inspect.log`)

Generated runs are stored locally under the existing ignored runs directory. The README records reproduction commands.
