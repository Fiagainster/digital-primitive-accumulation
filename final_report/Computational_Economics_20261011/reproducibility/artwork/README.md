# Figure reproduction

Run `python artwork/build_artwork.py` from the package root after installing
`requirements-verification.txt`. All 15 figures are generated from the accepted
Stage 4 CSV tables and diagram geometry. Output is written to `output/artwork`;
accepted files and raw records are never overwritten. The wrapper changes paths
and output locations only relative to the submission figure generator.

PDF, EPS and SVG are vector outputs; plots also have 300-dpi PNG previews.
Diagram PNG previews require Poppler's `pdftoppm` on PATH, or a `POPPLER_BIN`
environment variable. Missing Poppler skips only the diagram PNG preview.
The diagram geometry JSON is editable input. Reconstructed `.drawio` files
and SVG files are also editable. Embedded fonts use free DejaVu Sans.

The official submission figures have width 119 mm and lettering at least 8.5 pt.
Figure numbers and raw-data provenance are in
`audit/cross_document_references.json` and `audit/submitted_artwork_manifest.json`.
Date and metadata fields can change PDF byte hashes between builds. Compare
the figure's numerical labels, vector content and frozen input fingerprints;
do not treat a newly generated PDF hash as a new simulation result.
