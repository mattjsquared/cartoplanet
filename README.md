
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
import numpy as np
from cartoplanet.canvas import Canvas
from cartoplanet.layer import GridLayer
from cartoplanet.projections import PLATE_CARREE, projections
from cartoplanet.boundaries import boundaries

# Create a 2x2 canvas with colorbar axis
canvas = Canvas(
  nrows=1, ncols=2, 
  with_cax=True,
  projections=[projections['LAEA_NS'], projections['LAEA_FS']]
  boundaries=boundaries['limb_circle']
)

# Create demo grid data
lon = np.linspace(-180, 180, 30)
lat = np.linspace(-90, 90, 15)
Lon, Lat = np.meshgrid(lon, lat)
Z = np.sin(np.radians(Lat)) * np.cos(np.radians(Lon))

# Wrap data in a GridLayer
grid_layer = GridLayer(
  name="demo",
  data=Z,
  lat=lat,
  lon=lon,
  crs=PLATE_CARREE
)

# Plot using Pane.draw
canvas[0].draw(grid_layer, cmap="viridis")
canvas[0].ax.set_title("Low-resolution grid demo")

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

- `Canvas`: Main plotting engine. Manages figure, grid layout, colorbar axes, and batch operations. Use `canvas[i]` to access panes.
- `Pane`: Represents a single subplot. Attributes: `ax`, `projection`, `boundary`. Methods: `draw(layer, **style)`, `update()`, `plot_demo()`.
- `Layer`: Abstract data wrapper. Subclasses:
  - `GridLayer`: 2D gridded data (e.g., `xarray.DataArray`, `numpy.ndarray`).
- `Renderer`: Stateless drawing engine. Subclasses:
  - `GridRenderer`: Plots grids/images.

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
