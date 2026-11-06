# Publication readiness

## Corrections

- The generated table rows already ended in `\\`. The braced LaTeX `\input` wrapper does file bookkeeping after the fragment ends. Inside `tabular`, those tokens intervene between the last row terminator and the `\noalign` issued by `\bottomrule`. The generator now emits row macros that load in the preamble and expand directly inside each alignment. The two plot fragments retain their original format. [LaTeX kernel file-loading implementation](https://github.com/latex3/latex2e/blob/main/base/ltfilehook.dtx)
- The paper and README describe the theorem as a specialization of message-passing locality. They distinguish observed generalization and architectural limits from evidence of algorithm execution under trace supervision. The scope and confounders of the depth and training-source comparisons remain documented.
- The setup and build instructions record verified versions, explain replay variability, and regenerate into a scratch directory. The Phase 3 and 4 reports identify archive-only paths without broken checkout links. The generated PDF and LaTeX contents file are ignored.

## Verification

- The full Python suite passed all 36 tests in the existing environment with Python 3.12.14, PyTorch 2.15.0+cpu, NetworkX 3.4.2 and pytest 8.3.5. `pip check` found no broken requirements. PyTorch emitted an optional NumPy warning in a direct import. The checked workflow does not require NumPy. A clean dependency install was not exercised.
- All 34 archived SHA-256 hashes matched. The paper generator recomputed metrics from saved predictions and checked all 50 main-run locality pairs. Regeneration in a temporary directory reproduced all three table payloads and both plots. The three table files differ from the original fragments only by the row macro wrappers and line-ending comments. Archived files were read only and no training was rerun.
- Both summarizers ran on temporary copies of the main JSONL files, and the replay entry points and arguments were checked against their help output and source. Exact replay can vary with operating system, hardware and dependency builds.
- All five figure and table reference keys and all eight citation keys resolve in the LaTeX source. Proofs, equations, numerical results and bibliography entries were preserved.
- No LaTeX compiler or build driver was available. Compilation, compiler warning review and visual PDF inspection remain pending. Follow the [build commands and inspection checklist](README.md) on a machine with TeX Live or MiKTeX.

## GitHub review

- The local repository has no commits, reflog entries, or configured remote. The index is empty. Git's object check found two unreachable trees and two unreachable blobs, which were inspected. They contain early project files, including an earlier ignore file and project instructions. No history was rewritten.
- A targeted scan of the prospective source tree found no obvious credential patterns, personal contact details, machine-specific paths or AI/LLM usage disclosures. The review covered publishable source and documentation. The excluded local experiment bundle would need its own review if distributed. The largest prospective file was the manuscript at approximately 21 KB. The virtual environment, caches and local runs are ignored.
- The checkout can build the paper using the included fragments once a TeX toolchain is available. Original hash verification requires separate access to the archived evidence bundle. Fresh runs can regenerate the artifact types described in the README.

## Decisions for the rights holder

- Choose reuse terms if you want to offer them. The repository currently has no declared license.
- Decide whether to distribute the original evidence bundle separately. The source release can use the included fragments and replay instructions.
- Authorize any future push or publication explicitly. This review left the repository local.

This is a historical review. See the latest [project status](../PROJECT_STATUS.md) for the final Overleaf review and current publication preparation.
