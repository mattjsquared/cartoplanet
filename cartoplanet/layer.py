import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from typing import Union, Any
from cartoplanet.projections import PLATE_CARREE
from cartoplanet.types import GridType, CoordType


class Layer:
  def __init__(
      self,
      name: str,
      kind: str,
      data: Any,
      crs: ccrs.Projection = None
    ):
    # Validate input
    if (kind == 'sh') and (crs is not None):
      raise ValueError("SHLayer (`kind` == 'sh') corresponds to the frequency domain, not the spatial domain, and does not support a CRS.")
    # Set attributes
    self.name = name
    self._kind = kind
    self.data = data
    self.crs = crs or PLATE_CARREE

  @property
  def kind(self):
    return self._kind
  

class GridLayer(Layer):
  def __init__(
      self,
      name: str,
      data: GridType,
      crs: ccrs.Projection = None,
      lat: CoordType = None,
      lon: CoordType = None
  ):
    super().__init__(name=name, kind='grid', data=data, crs=crs)
    self.data = self._validate_data(data, lat, lon)

  def _validate_data(self, data, lat, lon):
    # Check validity of inputs
    # is `data` a valid type?
    if not isinstance(data, (np.ndarray, xr.DataArray, xr.Dataset)):
      raise TypeError("Data must be a valid grid type (`numpy.ndarray`, `xarray.DataArray`, or `xarray.Dataset`).")
    # are `lat` and `lon` valid types?
    if any(not isinstance(coord, (list, np.ndarray, xr.DataArray)) for coord in (lat, lon)):
      raise TypeError("If specified, `lat` and `lon` must be `list`, `numpy.ndarray`, or `xarray.DataArray`.")
    if not isinstance(lon, type(lat)):
      raise TypeError("`lat` and `lon` must be of the same type.")
    # do we have `lat` and `lon`?
    if isinstance(data, np.ndarray):
      if any(coord is None for coord in (lat, lon)):
        raise ValueError("If `data` is a `numpy.ndarray`, both `lat` and `lon` must be provided.")
    if isinstance(data, (xr.DataArray, xr.Dataset)):
      if any(coord not in data.coords for coord in ('lat', 'lon')):
        raise ValueError("If `data` is an `xarray.DataArray` or `xarray.Dataset`, both `lat` and `lon` must be present as coordinates in `data`.")
    # Standardize data and coordinates
    if isinstance(lat, (np.ndarray)) and (lat.ndim == 2):
      lat = lat[:, 0]
    if isinstance(lon, (np.ndarray)) and (lon.ndim == 2):
      lon = lon[0, :]
    if isinstance(data, np.ndarray):
      data = xr.DataArray(data, coords={'lat': lat, 'lon': lon})
    return data

  @property
  def lat(self):
    return self.data.coords['lat']

  @property
  def lon(self):
    return self.data.coords['lon']
  
  def get_coord_grids(self, as_xarray: bool = False):
    if as_xarray:
      return xr.broadcast(self.lon, self.lat)
    else:
      return np.meshgrid(self.lon, self.lat)


class PointLayer(Layer):
  def __init__(self, name, data, crs=None):
    super().__init__(name=name, kind='point', data=data, crs=crs)


class GeometryLayer(Layer):
  def __init__(self, name, data, crs=None):
    super().__init__(name=name, kind='geometry', data=data, crs=crs)


class SHLayer(Layer):
  def __init__(self, name, data, crs=None):
    super().__init__(name=name, kind='sh', data=data, crs=crs)
    
  @property
  def crs(self):
    raise AttributeError("SHLayer does not use a CRS.")
