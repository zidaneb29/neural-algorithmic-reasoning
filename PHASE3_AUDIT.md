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

## Constant baselines on the archived split

The node-pooled mean of the 390 non-source training targets was 7.990 weight units. The reconstructed test split gave graph-macro MAE / RMSE of 3.836 / 4.579 for that constant and 8.604 / 9.204 for zero. The larger split gave 3.922 / 4.823 for the training mean and 8.830 / 9.752 for zero. The path probe gave 12.448 / 15.258 for the training mean and 19.043 / 21.504 for zero. The Phase 2 GNNs had lower mean MAE than both constants on every archived evaluation split. The Phase 3 JSONL records the full baseline predictions and scores for the archived splits.

After adding the optional depth argument to the shared trainer, a separate replay of Phase 2 seed 0 reproduced both architectures' saved histories, selected epochs and evaluation metrics exactly. The archived reference kept its original checksum.
