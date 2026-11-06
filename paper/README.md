# Paper build and verification

[main.tex](main.tex) includes three tables and two figures from the generated [data fragments](data/). Table row definitions load in the preamble so the file-input hooks run outside the table alignments. Build from the paper directory with TeX Live or MiKTeX including AMS-LaTeX, geometry, booktabs, flafter, placeins, hyperref, TikZ and PGFPlots 1.18 or newer.

~~~sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

Alternatively, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice there to resolve cross-references. After the build, inspect `main.log` for undefined references or citations, rerun notices and overfull or underfull boxes. In the PDF, check the metric displays, the placement of Figures 1 and 2 and all three tables, the continuity of the proofs, and the reference links. The final formatting revision was compiled and visually reviewed in Overleaf. The [reviewed paper PDF](neural-algorithmic-reasoning.pdf) is included. For an Overleaf rebuild, upload main.tex and the data directory together at the project root, select main.tex as the main document, and compile with pdfLaTeX.

From the repository root, verify the original local archive and regenerate the fragments into a scratch directory for comparison with `paper/data`.

~~~sh
python scripts/paper_data.py runs/phase3-main.jsonl runs/phase4-main.jsonl replay/archive-paper-data --archive-manifest paper/archive.sha256
git diff --no-index --ignore-cr-at-eol paper/data replay/archive-paper-data
~~~

The archive command requires the original local `runs/` bundle. A source checkout includes the generated fragments for building the paper. For replays, pass the fresh JSONL files and a fresh destination. Omit the archive flag. Replays can differ in runtime fields and floating-point values.
