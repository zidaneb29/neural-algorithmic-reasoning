# Research plan

For a graph \(G=(V,E)\), source \(s\), and edge weights \(w\), the target \(d_w(s,v)\) is the minimum sum of weights along a path from \(s\) to \(v\). Positive integer weights make the NetworkX Dijkstra labels exact in integer arithmetic. Hop distance \(d_H\) counts edges along an unweighted shortest path. A minimum-weight path may use more edges than \(d_H(s,v)\). Training will supervise final distances. Algorithm execution would require separate evidence, such as comparisons of intermediate states to an algorithm trace. [Veličković et al., Neural Execution of Graph Algorithms](https://arxiv.org/abs/1910.10593).

## Locality proposition

Assume synchronous message passing on the original edges, fixed local edge features, a source indicator as the only source-dependent input, and a nodewise readout. Each update uses the vertex's previous state, its neighbors' previous states, and incident edge features. Residual connections may add the previous state. Global pooling and source-dependent preprocessing are excluded.

**Proposition.** On a fixed weighted graph, two source assignments whose indicators agree within \(k\) hops of \(v\) give identical outputs at \(v\) after \(k\) rounds.

**Proof.** At round zero, the initial state at each vertex depends only on its own indicator and fixed local features. Suppose that after \(t\) rounds each state depends on source indicators within \(t\) hops. Every vertex within \(t\) hops of a neighbor of \(v\) lies within \(t+1\) hops of \(v\). The update at \(v\) therefore depends only on source indicators within \(t+1\) hops, including when it adds the previous state. Induction and the nodewise readout give the result.

On a unit-weight path with vertices \(0,\ldots,k+2\), choose sources \(k+1\) and \(k+2\). At vertex zero both runs produce the same output despite targets differing by one, so at least one absolute error is at least one-half. The indicator at a source cannot affect a vertex before their hop distance in rounds. Reaching that depth alone gives no guarantee of exact weighted prediction. This argument applies the established message-passing framework. [Gilmer et al., Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212).

## Hypotheses and comparisons

| Question | Testable hypothesis | Comparison |
| --- | --- | --- |
| RQ1 Size | Graph-level normalized MAE increases on larger graphs. | Train on small graphs and compare ID test error to size-only OOD error within family. |
| RQ2 Structure | At matched sizes and similar edge counts, families with a larger fraction of nodes beyond the receptive field have higher normalized MAE. | Train on one family and test on held-out families. Report residual differences in density, degree, and diameter. |
| RQ3 Depth | Increasing depth reduces error more for nodes newly within reach than for nodes already within reach. | Compare the same graph-source pairs at several depths and stratify by hop distance. |
| RQ4 Residuals | Residual connections reduce validation loss at a fixed update budget and reduce OOD normalized MAE. | Pair data and initialization seeds across models. Compare training curves and held-out error separately. |

## Protocol before training

Start with small connected random graphs and deterministic path, cycle, and grid probes. Fix the weight law across splits. Rejection sampling changes the topology distribution to connected draws. For size shifts, preselect a density policy such as approximately fixed expected degree and record the realized degree distribution, since conditioning can change it. Compare a size-only shift, a family-only shift at matched sizes, and optionally a combined shift. Family comparisons are observational with respect to diameter and degree.

Create disjoint graph-level train, validation, and test manifests with recorded seeds and settings. Keep all sources, relabelings, and augmentations of a weighted instance in one split. Audit duplicate weighted instances, including relabelings where practical, before training. Paths, cycles, and fixed-shape grids have a single topology for a given size and shape. Repeated samples at those settings test new weights and sources on the same topology. Reserve shapes or families when testing transfer to unseen topologies.

Report per-graph non-source MAE and RMSE and macro-average across graphs. Also report per-graph normalized MAE, defined as the sum of absolute errors divided by the sum of true non-source distances. Stratify by unweighted hops and weighted target distance and report bucket counts. Record graph size, edge count, unweighted diameter, degree distribution, weight law, depth, parameter count, and data and training seeds. Use validation alone for model selection. Report paired seed-level differences over at least five seeds and their variability. Treat graphs and training runs as experimental units for uncertainty, since nodes from the same graph are dependent.

Before Phase 2, fix sample sizes, split manifests, size bands, density policy, depth and width settings, aggregation and edge encoding, loss, optimizer, and compute budget. Compare equal-width architectures with identical trainable components and a parameter-free residual addition. Spectral analysis can follow the initial comparisons. Oversmoothing and oversquashing would require separate diagnostics.

## Phase 3 implementation and findings

[Phase 3 results](PHASE3_RESULTS.md) records the realized size, family, locality and depth experiments, including differences from the Phase 2 protocol. The [Phase 2 audit](PHASE3_AUDIT.md) describes the archived split and its constant baselines.
