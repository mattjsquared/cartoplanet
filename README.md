# cartoplanet

A Python package for reproducible, publication-quality planetary mapping and spherically-aware geospatial processing.

## Features
- Flexible plotting for arrays, points, and geometry
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

See the documentation for details on optional features and troubleshooting installation issues.

## License
MIT (or specify)
