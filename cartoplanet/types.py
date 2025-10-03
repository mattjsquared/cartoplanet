import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd

from typing import Optional, Union, List, Dict, Any

# Plotting
ProjectionType = Optional[Union[ccrs.Projection, List[Optional[ccrs.Projection]]]]
BoundaryType = Optional[Union[mpl.path.Path, List[Optional[mpl.path.Path]]]]
KWType = Optional[Dict[str, Any]]

# Data
GridType = Union[np.ndarray, xr.Dataset, xr.DataArray]
CoordType = Union[list, np.ndarray, str]
VectorType = Union[list, np.ndarray, pd.Series]
ScatterType = Union[list, dict, np.ndarray, pd.DataFrame]

