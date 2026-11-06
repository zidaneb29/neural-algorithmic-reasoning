# Phase 4 targeted validation

## Protocol

The factorial comparison uses the existing standard GNN with four or eight rounds and trains each depth with sources at either vertex 11 or vertex 0 of the same 24-vertex weighted paths. Every path draws independent integer weights uniformly from 1–10. The center regime has hops 1–12 and the endpoint regime has hops 1–23. The training intervention changes the hop distribution, target distribution, and source position relative to the graph boundary together. It holds topology, graph size, edge weights and the number of queries fixed.

Within each replicate, split weighted paths into train, validation and test before encoding sources. The canonical weight sequence up to reversal excludes weighted-isomorphic duplicates across the study's main replicates and splits. Validation and test both query each path at the center and endpoint. The two test probes share graph draws. The model selector uses the earliest minimum validation MAE across both source placements. At each depth the two training regimes share initialization and graph order. Across depths the seeds and order match, with different layer widths and initialization shapes.

Depth 4 uses width 32 and 21,217 parameters. Depth 8 uses width 23 and 22,150 parameters, a 4.4% increase. The forward dense multiply-accumulate proxy is 679,168 versus 703,616 per query, a 3.6% increase. It counts the two message linear maps on each directed edge, two update linear maps per node, and encoder and readout. This proxy omits bias operations, aggregation, activations, backward work, optimizer overhead and layer dispatch. Width and measured runtime can differ. One Adam update per graph uses learning rate 0.001 and non-source MAE on labels divided by ten. There are 640 updates per main fit.

The pilot used training seed 90 and data seed 4090, eight training paths, four paths per held-out split and two epochs. It completed 64 updates across four fits and all four locality pairs. Recorded fit times totaled 3.6 seconds, including first-fit overhead. Extrapolating the warm fit timings gave roughly two to four minutes for the 12,800-update main run. The pilot's initial import failed because the source tree was absent from the module search path. Setting PYTHONPATH resolved it. No data or model protocol changed. The main run used training seeds 0–4, data seeds 4000–4004, 32 training paths, 16 paths per held-out split and 20 epochs. Both runs used one CPU thread, Python 3.12.14, PyTorch 2.15.0+cpu and NetworkX 3.4.2. PyTorch emitted a NumPy initialization warning. The experiment uses native tensors.

All test MAEs below use original weight units. Overall MAE averages graphs equally. Hop-band MAE pools non-source nodes within each replicate. Each replicate has 368 nodes per probe. Counts for the bands 1–4, 5–8, 9–12, and 13+ are 128, 128, 112, and 0 at center and 64, 64, 64, and 176 at endpoint. For depth 4, hops 5+ are outside source reach. For depth 8, hops 9+ are outside. Shared bands compare identical test observations. Variability is the sample SD of paired seed differences and combines graph-draw and initialization variation.

## Reproduction and artifacts

Run from the project root in PowerShell with the training dependencies installed.

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m gnn_generalization.phase4 runs/phase4-pilot.jsonl --pilot
.venv/Scripts/python.exe -m gnn_generalization.phase4 runs/phase4-main.jsonl
.venv/Scripts/python.exe scripts/summarize_phase4.py runs/phase4-main.jsonl
.venv/Scripts/python.exe -m pytest -q
```

Use fresh output names when reproducing existing runs. The JSONL files contain versions, graph seeds and ordered weights, curves, per-node predictions, and locality pairs. The optional spectral investigation was skipped to keep the study focused on the source-placement and depth controls.

- Pilot records (`runs/phase4-pilot.jsonl`), pilot log (`runs/phase4-pilot.log`), and initial import failure (`runs/phase4-pilot-import-failure.log`)
- Main records (`runs/phase4-main.jsonl`), main log (`runs/phase4-main.log`), and paired numerical summary (`runs/phase4-main-summary.log`)
- Test log (`runs/phase4-suite.log`)

## Results

All 20 main fits completed and passed the exact fixed-graph locality check. Five selected the final epoch, so the training curves do not establish convergence. Median recorded fit-and-evaluation time was 5.83 seconds at depth 4 and 9.15 at depth 8. The dense-operation proxy and parameter count were close, while wall time was about 57% higher for depth 8.

| Test source | Center training, depth 4 | Center training, depth 8 | Endpoint training, depth 4 | Endpoint training, depth 8 |
| --- | ---: | ---: | ---: | ---: |
| Center | 7.97 | 6.71 | 13.94 | 10.52 |
| Endpoint | 29.75 | 26.92 | 26.18 | 19.68 |

| Endpoint test hop band | Depth 8 − 4, center training | Depth 8 − 4, endpoint training | Endpoint − center training, depth 4 | Endpoint − center training, depth 8 |
| --- | ---: | ---: | ---: | ---: |
| 1–4 | +1.05 ± 1.43 | −0.93 ± 2.25 | +1.12 ± 3.05 | −0.85 ± 2.04 |
| 5–8 | −2.83 ± 2.59 | −14.32 ± 9.18 | +13.03 ± 6.45 | +1.54 ± 2.21 |
| 9–12 | −3.42 ± 2.19 | +4.32 ± 1.07 | −4.25 ± 3.56 | +3.49 ± 2.14 |
| 13+ | −4.02 ± 1.34 | −9.64 ± 5.96 | −11.05 ± 6.05 | −16.66 ± 3.52 |
| All | −2.83 ± 0.52 | −6.51 ± 1.92 | −3.56 ± 2.26 | −7.24 ± 1.91 |

Local figure `runs/phase4-main.svg`. Paired MAE changes by hop band on endpoint queries. Whiskers show one sample SD.

Depth 8 had lower overall mean MAE under both training regimes and on both probes with the near-matched parameter and dense-operation budgets. The largest depth gain after endpoint training occurred in the newly reachable 5–8 band, but the 13+ band improved while the 9–12 band worsened. Visibility, changed structural context, width, and different fitted parameters all vary in the depth comparison. The experiment does not estimate a causal effect of source visibility on test MAE.

Endpoint training improved the endpoint probe overall, particularly at hops 13–23, which lie beyond both models' receptive fields and outside center training hop support. The distant-node improvement is compatible with a shift in predictions based on local features when the training labels are larger. The result does not identify the learned mechanism. On the center probe, endpoint training raised mean MAE by 5.96 ± 2.24 at depth 4 and 3.81 ± 0.55 at depth 8. Endpoint training also worsened some endpoint bands, including 5–8 at depth 4. The observed gain is transfer to fresh weights on a fixed topology with a matched endpoint source policy. It does not establish extrapolation beyond the endpoint training range or transfer across families.

Taken together, the exact intervention identifies an architectural limit that holds for every fitted model. The held-out tests show that training distribution and near-budget-matched depth still influence errors within that limit. Fixed optimizer settings and unequal measured runtime leave optimization and elapsed-compute effects unresolved. The five replicates support descriptive paired comparisons on this protocol. Verification log (`runs/phase4-verify.log`)
