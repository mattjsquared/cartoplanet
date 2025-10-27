import matplotlib as mpl
import geopandas as gpd
import xarray as xr
import xesmf as xe

from shapely.geometry import Polygon
from cartoplanet.projections import PLATE_CARREE


def regrid_spherical(
    source: xr.DataArray,
    target: xr.DataArray,
    method: str = 'bilinear',
    periodic: bool = True,
    **kwargs
) -> xr.DataArray:
    """
    Regrid a source xarray DataArray to the resolution of a target xarray DataArray using xESMF.

    Parameters:
    - source (xr.DataArray): The source DataArray to be regridded.
    - target (xr.DataArray): The target DataArray whose resolution will be used.
    - method (str): The regridding method. Options include 'bilinear', 'nearest_s2d', 'nearest_d2s', 'conservative', etc. Default is 'bilinear'.
    - periodic (bool): Whether the longitude is periodic (i.e., using a global grid). Default is True.

    Returns:
    - xr.DataArray: The regridded DataArray.
    """
    # Create regridder
    source = source.rename(x='lon', y='lat')
    target = target.rename(x='lon', y='lat')
    regridder = xe.Regridder(source, target, method=method, periodic=periodic, **kwargs)

    # Perform regridding
    regridded = regridder(source)

    return regridded

def regrid_to_lowest_resolution(
    *arrays: xr.DataArray,
    method: str = 'bilinear',
    periodic: bool = True,
    **kwargs
) -> list[xr.DataArray]:
    """
    Regrid multiple xarray DataArrays to the lowest resolution among them using xESMF.

    Parameters:
    - arrays (xr.DataArray): The DataArrays to be regridded.
    - method (str): The regridding method. Options include 'bilinear', 'nearest_s2d', 'nearest_d2s', 'conservative', etc. Default is 'bilinear'.
    - periodic (bool): Whether the longitude is periodic. Default is True.

    Returns:
    - list[xr.DataArray]: A list of regridded DataArrays.
    """
    # Determine the target array with the lowest resolution
    sizes = [da.sizes[da.dims[0]] * da.sizes[da.dims[1]] for da in arrays]
    target = arrays[sizes.index(max(sizes))]

    # Regrid all arrays to the target resolution
    regridded_arrays = [regrid_spherical(arr, target, method=method, periodic=periodic, **kwargs) for arr in arrays]

    return regridded_arrays

def generate_limb_circle() -> mpl.path.Path:
  xmin = -90
  xmax = 90
  ymin = -90
  ymax = 90
  rect = mpl.path.Path([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                        [xmin, ymin]]
                        ).interpolated(100)
  return rect

def generate_rectangle(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs=PLATE_CARREE
) -> mpl.path.Path:
  rect = mpl.path.Path([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                        [xmin, ymin]]
                        ).interpolated(100)
  return rect

def generate_nearside_mask(
    crs=PLATE_CARREE
) -> gpd.GeoDataFrame:
  mask = gpd.GeoDataFrame(geometry=[Polygon(generate_limb_circle().vertices)], crs=PLATE_CARREE)
  return mask.to_crs(crs)

def clip_geometry(
    gdf: gpd.GeoDataFrame,
    mask: gpd.GeoDataFrame, 
    how: str='intersection'
) -> gpd.GeoDataFrame:
  if gdf.crs != mask.crs:
    raise ValueError("GeoDataFrames `gdf` and `mask` must have the same CRS")
  if how == 'intersection':
    clipped = gpd.overlay(gdf, mask, how='intersection')
  elif how == 'difference':
    clipped = gpd.overlay(gdf, mask, how='difference')
  else:
    raise ValueError("Invalid 'how' parameter. Use 'intersection' or 'difference'.")
  return clipped

def dissolve_geometry(
    gdf: gpd.GeoDataFrame,
    name: str = "dissolved"
) -> gpd.GeoDataFrame:
  dissolved = gpd.GeoDataFrame(
    {'name': [name]},
    geometry = [gdf.union_all()],
    crs = gdf.crs
  )
  return dissolved
