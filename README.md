
# cartoplanet

**A Python package for reproducible, publication-quality planetary mapping and spherically-aware geospatial processing.**

## Features

- Modular plotting architecture (`Canvas`, `Pane`) for flexible figure layouts
- Spherical geospatial processing utilities
- Publication-ready figure generation

## Installation

### Recommended: Conda/conda-forge
Most dependencies are pip-installable, but some (notably `pyinterp`, `xESMF`, and sometimes `fiona`/`rasterio`) are best installed via conda-forge for reliability and compatibility.

```bash
conda env create -f environment.yml
conda activate cartoplanet
```

### Alternative: pip + Homebrew
Most core dependencies are pip-safe. For advanced geospatial features, you may need:
- `GDAL`: install via Homebrew (`brew install gdal`) on macOS
- `pyinterp`, `xESMF`: install via conda-forge only
- `fiona`, `rasterio`: pip-installable, but may require system libraries for advanced usage

```bash
pip install cartoplanet
# For GDAL (macOS): brew install gdal
```

## Usage

### Python API

```python
from cartoplanet.canvas import Canvas
from cartoplanet.projections import projections
from cartoplanet.boundaries import boundaries

# Create a 2x2 canvas with colorbar axis
canvas = Canvas(nrows=2, ncols=2, with_cax=True)

# Plot demo data on the first pane
canvas[0].plot_demo()

# Update projections and boundaries for all panes
proj = [projections['LAEA_NS'], projections['LAEA_FS']] * 2
bound = [boundaries['limb_circle']] * 4
for i, pane in enumerate(canvas):
  pane.update(projection=proj[i], boundary=bound[i])

# Show or save the figure
canvas.show()
canvas.save("output.png")
```

### CLI Usage

Run from the command line:

```bash
python -m cartoplanet --nrows 2 --ncols 2 --out output1.png output2.png
```

Arguments:
- `--nrows`, `--ncols`: Grid size
- `--out`: Output file name(s), accepts multiple values

## API Overview

- `Canvas`: Main plotting engine. Supports flexible grid layouts, colorbar axes, and batch operations.
- `Pane`: Represents a single subplot. Attributes: `ax`, `projection`, `boundary`. Methods: `plot_demo()`, `update()`.

## Development and Testing

- The conda environment includes `pytest` for running unit tests.
- To install test dependencies with pip (if using pyproject.toml):
  ```bash
  pip install .[test]
  ```
- To run tests:
  ```bash
  pytest
  ```

## License

MIT
