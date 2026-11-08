# Phase 2 audit

The original JSONL has SHA-256 `fa5a193798cfbf5edca0216d03da3ad7f1a58d166b71cf5b7cac9375833acf60`. Regeneration with data seed 17 reproduced its manifest exactly. An independent weighted-isomorphism check grouped all 124 instances by size and weight multiset, then checked candidate pairs using exact edge-weight matching. No duplicates were found. The 32-test suite passed, including weighted detours, reciprocal encoding, paired initialization without global RNG changes, source masking, and relabeling equivariance within floating-point tolerance.

Both models received the binary source flag and one edge weight divided by ten in each direction. The target was Dijkstra distance divided by ten. Unweighted hop counts were evaluation metadata. The nodewise readout was unconstrained for non-source vertices and explicitly zeroed the source. Both architectures used 21,217 parameters, paired initialization and graph orders, and 768 Adam updates per run. Residual connections added one state tensor per layer. The training curve averages losses observed before updates, and checkpoint selection uses graph-macro non-source validation MAE. Test metrics use the selected model in original weight units. Graph-macro RMSE averages per-graph square roots.

| Split | Graphs | Non-source nodes | Target min / mean / max | Nodes beyond four hops | Standard MAE / RMSE | Residual MAE / RMSE |
| --- | ---: | ---: | --- | ---: | --- | --- |
| Train | 64 | 390 | 1 / 7.99 / 35 | 0 | | |
| Validation | 16 | 94 | 1 / 8.54 / 34 | 0 | 2.782 / 3.263 | 2.896 / 3.412 |
| Test | 16 | 98 | 1 / 8.86 / 32 | 1 | 2.428 / 2.912 | 2.553 / 3.128 |
| Larger | 16 | 192 | 1 / 8.83 / 30 | 0 | 2.633 / 3.252 | 2.791 / 3.439 |
| Path | 4 | 26 | 1 / 19.46 / 56 | 6 | 8.985 / 11.486 | 9.334 / 11.919 |
| Cycle | 4 | 22 | 3 / 11.45 / 25 | 0 | 2.342 / 2.858 | 2.533 / 3.029 |
| Grid | 4 | 26 | 1 / 9.12 / 17 | 0 | 3.012 / 3.348 | 2.998 / 3.359 |

Phase 2 stored aggregate errors, curves and manifests. Individual predictions and checkpoint weights were not saved. The five training seeds shared one data draw. The sparse long-hop observations and target shift on paths limit the interpretation of the pilot. The recorded NumPy warning did not prevent tensor computation. Phase 3 scores zero and a node-pooled training-mean constant on the original manifest as well as its new evaluation splits.

## Pilot settings

- `data.py` encodes one source flag per node, reciprocal edge weights divided by ten, and Dijkstra distances divided by ten. Hop counts are evaluation metadata.
- `models.py` implements two width-32, depth-four pilot variants with identical trainable components and paired initialization. The residual variant adds the incoming node state at every layer.
- `pilot.py` uses one Adam step per graph with non-source MAE and selects the epoch by graph-macro validation MAE. The default budget is 768 updates per model and seed.
- The pilot uses 64/16/16 connected Erdős–Rényi graphs for train/validation/test at 6 or 8 nodes, 16 larger draws at 12 or 14 nodes, and four graphs each for the path, cycle and grid probes. The Erdős–Rényi parameter gives expected degree three before conditioning on connectivity.

## Pilot results

The five paired training seeds (0–4) shared one graph manifest (data seed 17). Each model had 21,217 parameters. All 32 tests passed. The run used Python 3.12.14, PyTorch 2.15.0+cpu and NetworkX 3.4.2. The private archive includes `runs/phase2-pilot.jsonl` for the manifest, per-epoch curves and bucket counts, and `runs/phase2-pilot.log` for console output. PyTorch emitted a NumPy interoperability warning. The pilot uses PyTorch tensors directly.

| Split | Standard MAE | Residual MAE | Paired difference, mean ± sample SD |
| --- | ---: | ---: | ---: |
| Validation | 2.782 | 2.896 | +0.115 ± 0.365 |
| In-distribution test | 2.428 | 2.553 | +0.125 ± 0.521 |
| Larger graphs | 2.633 | 2.791 | +0.159 ± 0.252 |
| Path probe | 8.985 | 9.334 | +0.349 ± 1.275 |
| Cycle probe | 2.342 | 2.533 | +0.191 ± 0.475 |
| Grid probe | 3.013 | 2.998 | −0.014 ± 0.320 |

Differences are residual minus standard. Mean normalized MAE was 0.305 versus 0.327 on the in-distribution test and 0.320 versus 0.338 on larger graphs. The seed variability reflects training on one fixed data draw. Only one in-distribution test node, zero larger-graph nodes and six path-probe nodes were beyond four hops. The size probe therefore provides little evidence about source reach. These measurements characterize final-distance regression under the pilot protocol.

The pilot uses simple, connected, undirected graphs with a uniformly sampled source and integer edge weights from 1 to 10. Bounded rejection sampling conditions random graph families on connectivity and can exhaust its attempt budget. To replay from the repository root, follow the [pilot command in the README](../README.md#reproduce-the-experiments).

## Constant baselines on the archived split

The node-pooled mean of the 390 non-source training targets was 7.990 weight units. The reconstructed test split gave graph-macro MAE / RMSE of 3.836 / 4.579 for that constant and 8.604 / 9.204 for zero. The larger split gave 3.922 / 4.823 for the training mean and 8.830 / 9.752 for zero. The path probe gave 12.448 / 15.258 for the training mean and 19.043 / 21.504 for zero. The Phase 2 GNNs had lower mean MAE than both constants on every archived evaluation split. The Phase 3 JSONL records the full baseline predictions and scores for the archived splits.

After adding the optional depth argument to the shared trainer, a separate replay of Phase 2 seed 0 reproduced both architectures' saved histories, selected epochs and evaluation metrics exactly. The archived reference kept its original checksum.
