import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr

from typing import Optional, Union, List, Dict, Any

# Plotting
ProjectionType = Optional[Union[ccrs.Projection, List[Optional[ccrs.Projection]]]]
BoundaryType = Optional[Union[mpl.path.Path, List[Optional[mpl.path.Path]]]]
KWType = Optional[Dict[str, Any]]

# Data
GridType = Union[np.ndarray, xr.Dataset, xr.DataArray]
CoordType = Optional[Union[list, np.ndarray, xr.DataArray]]

