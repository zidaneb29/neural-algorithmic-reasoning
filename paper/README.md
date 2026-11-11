# Paper build and verification

Zidane Boussofi is the author. The checked-in PDF was rebuilt and visually reviewed in Overleaf. [Research paper (PDF)](neural-algorithmic-reasoning.pdf)

[main.tex](main.tex) includes three tables and two figures from the generated [data fragments](data/). Table row definitions load in the preamble so the file-input hooks run outside the table alignments. Build from the paper directory with TeX Live or MiKTeX including AMS-LaTeX, geometry, booktabs, flafter, placeins, hyperref, TikZ and PGFPlots 1.18 or newer.

~~~sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

Alternatively, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice to resolve cross-references. In Overleaf, upload `main.tex` and the `data/` directory together at the project root, select `main.tex` as the main document, and compile with pdfLaTeX.

Inspect the compiler log for unresolved references or citations, rerun notices and layout warnings. In the PDF, check the byline, metric displays, both figures, all three tables, proof continuity, pagination and reference links. After review, save the build as `neural-algorithmic-reasoning.pdf` in this directory.

From the repository root, verify the original local archive and regenerate the fragments into a scratch directory for comparison with `paper/data`.

~~~sh
python scripts/paper_data.py runs/phase3-main.jsonl runs/phase4-main.jsonl replay/archive-paper-data --archive-manifest paper/archive.sha256
git diff --no-index --ignore-cr-at-eol paper/data replay/archive-paper-data
~~~

The archive command requires the original local `runs/` bundle. A source checkout includes the generated fragments for building the paper. For replays, pass the fresh JSONL files and a fresh destination. Omit the archive flag. Replays can differ in runtime fields and floating-point values.
