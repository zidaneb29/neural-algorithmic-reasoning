# Phase 5 final report

- The LaTeX paper includes a verified bibliography, three archive-derived tables, two plots, and complete proofs of attributed-neighborhood locality and the shortest-path pair bound. The proofs specialize established message passing.
- The generalization study found increasing raw error with size and mixed evidence that depth gives a benefit specific to newly reachable nodes. The controlled path study found lower mean error at depth eight under both source policies, with distinct runtime and width. Endpoint training traded center accuracy for endpoint accuracy.
- Three values in the Phase 3 report differed from the saved results by 0.01 weight units. The corrected standard depth-eight grid and cycle MAEs are 41.62 and 76.78. The cycle training-mean baseline MAE is 82.87.
- All 36 tests passed. Per-graph and macro metrics were independently checked against node predictions, and all 50 main-run source-swap pairs satisfied exact equality and the error bound. All original archive hashes matched. No new fits were required.
- The README provides installation, main and pilot replay commands, output conventions and environment versions. The original archives remain local. Cross-platform replay was outside this verification. PDF compilation and visual review require a LaTeX toolchain. The rights holder still needs to choose a license.
- Optional extensions include independently varying training and data seeds, matching hop and weighted-distance support across families, and studying trace supervision and representation diagnostics.

## Résumé description

**Neural Algorithmic Reasoning** · Independent undergraduate research project

- Built seeded synthetic graph benchmarks and PyTorch message-passing regressors for weighted shortest-path prediction with exact NetworkX Dijkstra targets.
- Analyzed 50 main-study fits across paired depth, residual, topology and source-placement comparisons using graph-level and hop-stratified error metrics.
- Proved a fixed-depth locality result and path-based prediction error bound, and verified exact source-swap invariance across all main-study fits.
