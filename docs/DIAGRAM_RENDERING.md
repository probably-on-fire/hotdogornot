# Diagram Rendering

Graphviz `.dot` sources for every committed diagram are stored under `docs/`
so the SVG/PNG renders can be regenerated from a single tool. This file
keeps the long render command list out of the main README.

## Prerequisites

Install Graphviz:

- Windows: <https://graphviz.org/download/> (adds `dot` to PATH).
- macOS: `brew install graphviz`.
- Debian/Ubuntu: `sudo apt-get install graphviz`.

## Render Commands

```bash
dot -Tsvg docs/README_ARCHITECTURE.dot -o docs/README_ARCHITECTURE.svg
dot -Tpng docs/README_ARCHITECTURE.dot -o docs/README_ARCHITECTURE.png

dot -Tsvg docs/README_TECHNICAL_OVERVIEW.dot -o docs/README_TECHNICAL_OVERVIEW.svg
dot -Tpng docs/README_TECHNICAL_OVERVIEW.dot -o docs/README_TECHNICAL_OVERVIEW.png

dot -Tsvg docs/SYSTEM_ARCHITECTURE_POSTER.dot -o docs/SYSTEM_ARCHITECTURE_POSTER.svg
dot -Gdpi=600 -Tpng docs/SYSTEM_ARCHITECTURE_POSTER.dot -o docs/SYSTEM_ARCHITECTURE_POSTER_600dpi.png

dot -Tsvg docs/SOFTWARE_ARCHITECTURE.dot -o docs/SOFTWARE_ARCHITECTURE.svg
dot -Tpng docs/SOFTWARE_ARCHITECTURE.dot -o docs/SOFTWARE_ARCHITECTURE.png

dot -Tsvg docs/MULTI_ARCHITECTURE_TRANSITION.dot -o docs/MULTI_ARCHITECTURE_TRANSITION.svg
dot -Tpng docs/MULTI_ARCHITECTURE_TRANSITION.dot -o docs/MULTI_ARCHITECTURE_TRANSITION.png
```

## Notes

- Commit both the `.dot` source and the rendered SVG/PNG. Rendered images
  make the README readable without tooling; `.dot` sources allow edits.
- Keep PNG output at the resolution intended for its use:
  - `*_ARCHITECTURE.png` and `*_TRANSITION.png`: default DPI for in-line
    README rendering.
  - `SYSTEM_ARCHITECTURE_POSTER_600dpi.png`: high-DPI for poster/print use.
- If a diagram is renamed, update both the `.dot` filename and any README
  references.
