import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import pyshtools as pysh
import warnings, pickle

from typing import Union, Optional, Any
from cartoplanet import config
from cartoplanet.palette import Palette
from cartoplanet.projections import PLATE_CARREE
from cartoplanet.types import GridType, CoordType, VectorType, ScatterType


class Layer:
  def __init__(
      self,
      name: str,
      kind: str,
      data: Optional[Any] = None,
      crs: Optional[ccrs.Projection] = None,
      palette: Optional[Any] = None
    ):
    # Validate input
    if (kind == 'sh') and (crs is not None):
      raise ValueError("SHLayer (`kind` == 'sh') corresponds to the frequency domain, not the spatial domain, and does not support a CRS.")
    # Set attributes
    self.name = name
    self._kind = kind
    self.data = data
    self.crs = crs or PLATE_CARREE
    # self.palette = self.initialize_palette(palette)

  @property
  def kind(self):
    return self._kind
  
  # def initialize_palette(self, palette_spec):
  #   if isinstance(palette_spec, (Palette, dict)):
  #     palette = Palette(palette_spec)
  #   elif palette_spec is not None:
  #     raise TypeError("`palette_spec` must be able to initialize a `Palette` object.")
  #   else:
  #     try:
  #       vmin = self.values.min()
  #       vmax = self.values.max()
  #     except:
  #       vmin = None
  #       vmax = None
  #     palette = Palette(vmin=vmin, vmax=vmax)
  #   return palette

  def _raise_undefined_conversion(self, method_name: str):
    raise NotImplementedError(f"{method_name}() is not implemented for Layer of kind '{self._kind}'.")

for method in [
  'to_numpy', 
  'to_xarray', 
  'to_dataframe', 
  'to_geodataframe', 
  'to_dict', 
  'to_shcoeff', 
  'to_shapely', 
  'to_path', 
  'to_file', 
  'to_pickle', 
  'to_netcdf',
  'to_grid',
  'to_point',
  'to_geometry',
  'to_sh'
]:
  setattr(Layer, method, lambda self, m=method: self._raise_undefined_conversion(m))

class GridLayer(Layer):
  """
  A data wrapper for 2D gridded data (e.g., topography, images) with named coordinates.

  Supports input as a 2D numpy array or xarray DataArray/Dataset, and ensures robust coordinate handling.

  Parameters
  ----------
  name : str
    Name of the layer.
  data : GridType
    The grid data. Must be a 2D numpy.ndarray, xarray.DataArray, or xarray.Dataset.
  x : array-like or str
    For numpy input: 1D or 2D array of x coordinates (columns). For xarray input: name of the x dimension.
  y : array-like or str
    For numpy input: 1D or 2D array of y coordinates (rows). For xarray input: name of the y dimension.
  crs : cartopy.crs.Projection, optional
    The coordinate reference system. Defaults to PlateCarree if not specified.
  xname : str, default 'x'
    Name to use for the x coordinate in the resulting DataArray.
  yname : str, default 'y'
    Name to use for the y coordinate in the resulting DataArray.

  Notes
  -----
  - For numpy input, `x` and `y` must match the shape of `data` (x: columns, y: rows).
  - For xarray input, `x` and `y` must be the names of existing dimensions in `data`.
  - The resulting data is always stored as an xarray.DataArray with named coordinates.
  - Coordinate names can be changed after construction via the `.xname` and `.yname` properties.
  """
  def __init__(
      self,
      name: str,
      data: GridType,
      x: CoordType,
      y: CoordType,
      crs: ccrs.Projection = None,
      data_name: str = 'value',
      xname: str = 'x',
      yname: str = 'y'
  ):
    validated_data = self._validate_data(data, x, y, data_name, xname, yname)
    super().__init__(name=name, kind='grid', data=validated_data, crs=crs)
    self._dname = data_name
    self._xname = xname
    self._yname = yname

  def _validate_data(self, data, x, y, data_name, xname, yname):
    if isinstance(data, xr.Dataset):
      raise NotImplementedError("GridLayer does not yet fully support xarray.Dataset input.")
    # Check that `data` is a valid type
    if not isinstance(data, (np.ndarray, xr.DataArray, xr.Dataset)):
      raise TypeError("Data must be a valid grid type (`numpy.ndarray`, `xarray.DataArray`, or `xarray.Dataset`).")
    # Handle numpy input
    if isinstance(data, np.ndarray):
      # Check shape of `data`
      if data.ndim != 2:
        raise ValueError("If `data` is a `numpy.ndarray`, it must be 2-dimensional.")
      # Force coords to be ndarrays
      if isinstance(x, list): x = np.asarray(x)
      if isinstance(y, list): y = np.asarray(y)
      x = x[0, :] if x.ndim == 2 else x
      y = y[:, 0] if y.ndim == 2 else y
      data = xr.DataArray(data, coords={yname: y, xname: x})
    # Handle xarray input
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
      if not data.coords:
        raise ValueError("If `data` is an `xarray.DataArray` or `xarray.Dataset`, it must have coordinates.")
      if any(coord not in data.dims for coord in (x, y)):
        raise ValueError("Please provide valid `data.coord` names for `x` and `y`.")
      data = data.rename({x: xname, y: yname})
    data = data.rename(data_name)
    return data

  @property
  def dname(self):
    return self._dname
  
  @dname.setter
  def dname(self, value):
    self.data = self.data.rename(value)
    self._dname = value

  @property
  def xname(self):
    return self._xname
  
  @xname.setter
  def xname(self, value):
    self.data = self.data.rename({self._xname: value})
    self._xname = value

  @property
  def yname(self):
    return self._yname

  @yname.setter
  def yname(self, value):
    self.data = self.data.rename({self._yname: value})
    self._yname = value

  @property
  def values(self):
    return self.data.values
  
  @property
  def coords(self):
    return self.data.coords

  def get_coord_grids(self):
    return np.meshgrid(self.coords[self.xname], self.coords[self.yname])
  
  def to_numpy(self):
    return self.data.values
  
  def to_xarray(self):
    return self.data.copy()
  
  def to_point(self, new_name=None, kw_pointlayer=None):
    # Validate inputs
    name = new_name or self.name
    kw_pointlayer = kw_pointlayer or {}
    if kw_pointlayer.get('crs'):
      warnings.warn("The `crs` argument to `PointLayer()` is ignored in `to_point()`; the GridLayer CRS is used.")
      kw_pointlayer.pop('crs')
    for k in ['data_name', 'xname', 'yname']:
      if not kw_pointlayer.get(k):
        kw_pointlayer[k] = getattr(self, f"{k}")
    # Convert to PointLayer
    xgrid, ygrid = self.get_coord_grids()
    points = np.column_stack((self.data.values.ravel(), xgrid.ravel(), ygrid.ravel()))
    pointlayer = PointLayer(
      name=name, 
      data=points, 
      crs=self.crs, 
      **kw_pointlayer
    )
    return pointlayer
  
  def to_sh(self, new_name=None, kind=None, kw_grid=None, kw_expand=None):
    name = new_name or self.name
    kw_grid = kw_grid or {}
    kw_expand = kw_expand or {}
    if kind in ['grav', 'gravity', 'SHGravGrid', 'SHGravCoeffs']:
      grid = pysh.SHGravGrid.from_xarray(self.data, **kw_grid)
    elif kind in ['mag', 'magnetic', 'SHMagGrid', 'SHMagCoeffs']:
      grid = pysh.SHMagGrid.from_xarray(self.data, **kw_grid)
    else:
      grid = pysh.SHGrid.from_xarray(self.data, **kw_grid)
    coeffs = grid.expand(**kw_expand)
    shlayer = SHLayer(name=name, data=coeffs)
    return shlayer
  
  def to_pickle(self, filepath, kw_pickle=None):
    kw_pickle = kw_pickle or {}
    with open(filepath, 'wb') as f:
      pickle.dump(self, f, **kw_pickle)



class PointLayer(Layer):
  """
  A data wrapper for unstructured point data (e.g., observations, events) with named value and coordinate columns.

  Supports input as a pandas DataFrame, dict, list, or numpy array, and ensures robust column naming and access.

  Parameters
  ----------
  name : str
    Name of the layer.
  data : ScatterType
    The point data. Can be a pandas DataFrame, dict, list, or numpy.ndarray.
  data_key : str, optional
    For DataFrame or dict input: the key or column name for the data values.
  x : str or array-like, optional
    For DataFrame or dict input: the key or column name for x coordinates. For list/ndarray input: the x values.
  y : str or array-like, optional
    For DataFrame or dict input: the key or column name for y coordinates. For list/ndarray input: the y values.
  crs : cartopy.crs.Projection, optional
    The coordinate reference system. Defaults to PlateCarree if not specified.
  data_name : str, default 'value'
    Name to use for the value column in the resulting DataFrame.
  xname : str, default 'x'
    Name to use for the x coordinate column in the resulting DataFrame.
  yname : str, default 'y'
    Name to use for the y coordinate column in the resulting DataFrame.

  Notes
  -----
  - For DataFrame or dict input, `data_key`, `x`, and `y` must be strings identifying the relevant columns or keys.
  - For list or ndarray input, `x` and `y` must be provided as arrays if `data` is 1D, or the data must be shape (N,3) if 2D.
  - The resulting data is always stored as a pandas DataFrame with named columns.
  - Column names can be changed after construction via the `.dname`, `.xname`, and `.yname` properties.
  """
  def __init__(
      self, 
      name: str, 
      data: ScatterType,
      data_key: Optional[str] = None,
      x: Any = None,
      y: Any = None,
      crs: ccrs.Projection = None,
      data_name: str = 'value',
      xname: str = 'x',
      yname: str = 'y'
  ):
    validated_data = self._validate_data(data, x, y, data_key, data_name, xname, yname)
    super().__init__(name=name, kind='point', data=validated_data, crs=crs)
    self._dname = data_name
    self._xname = xname
    self._yname = yname

  def _validate_data(self, data, x, y, data_key, data_name, xname, yname):
    # Check that `data` is a valid type
    if not isinstance(data, (pd.DataFrame, np.ndarray, list, dict)):
      raise TypeError("Data must be a valid type (`pandas.DataFrame`, `numpy.ndarray`, `list`, or `dict`).")
    # Check for missing arguments
    if isinstance(data, (pd.DataFrame, dict)):
      if not any(isinstance(k, str) for k in (data_key, x, y)):
        raise ValueError("When `data` is a DataFrame or dict, `data_key`, `x`, and `y` must be provided as strings to identify the relevant vectors.")
    if isinstance(data, (list, np.ndarray)) and not all(isinstance(coord, (list, np.ndarray)) for coord in (x, y)):
      raise ValueError("When `data` is a list or numpy array, both `x` and `y` must be provided as lists or numpy arrays.")
    # Handle pandas input
    if isinstance(data, pd.DataFrame):
      df = data.loc[:, [data_key, x, y]]
      df.rename(columns={data_key: data_name, x: xname, y: yname}, inplace=True)
    # Handle dict input
    elif isinstance(data, dict):
      v_data = data.get(data_key)
      v_x = data.get(x)
      v_y = data.get(y)
      df = pd.DataFrame({data_name: v_data, xname: v_x, yname: v_y})
    # Handle vector or ndarray inputs
    elif isinstance(data, (list, np.ndarray)):
      data = np.asarray(data)
      if data.ndim == 1:
        df = pd.DataFrame({data_name: data, xname: x, yname: y})
      elif (data.ndim == 2) and (data.shape[1] == 3):
        df = pd.DataFrame(data, columns=[data_name, xname, yname])
      else:
        raise ValueError("PointLayer data must be a 1D array/list or a 2D array/list with shape (N,3).")
    return df
  
  @property
  def dname(self):
    return self._dname
  
  @dname.setter
  def dname(self, value):
    self.data.rename(columns={self._dname: value}, inplace=True)
    self._dname = value

  @property
  def xname(self):
    return self._xname
  
  @xname.setter
  def xname(self, value):
    self.data.rename(columns={self._xname: value}, inplace=True)
    self._xname = value

  @property
  def yname(self):
    return self._yname

  @yname.setter
  def yname(self, value):
    self.data.rename(columns={self._yname: value}, inplace=True)
    self._yname = value

  @property
  def values(self):
    return self.data[self._dname].to_numpy()

  @property
  def x(self):
    return self.data[self._xname].to_numpy()

  @property
  def y(self):
    return self.data[self._yname].to_numpy()

  @property
  def coords(self):
    return self.data[[self._xname, self._yname]]
  
  def get_coord_grids(self):
    return np.meshgrid(self.coords[self._xname], self.coords[self._yname])

  def to_numpy(self):
    return self.data.to_numpy()

  def to_dataframe(self):
    return self.data.copy()
  
  def to_dict(self):
    return self.data.to_dict(orient='list')
  
  def to_grid(self, new_name=None, kw_gridlayer=None):
    raise NotImplementedError("PointLayer.to_grid() is not yet implemented.")

  def to_geometry(self, new_name=None, kw_geometrylayer=None):
    raise NotImplementedError("PointLayer.to_geometry() is not yet implemented.")
  
  def to_sh(self, new_name=None, kw_expand=None, kw_shlayer=None):
    raise NotImplementedError("PointLayer.to_sh() is not yet implemented.")




class GeometryLayer(Layer):
  """
  A data wrapper for geometric data (e.g., polygons, lines, points) with named geometry and attribute columns.

  Supports input as a GeoDataFrame, DataFrame with a geometry column, or shapely geometry/list, and ensures robust geometry handling and access.

  Parameters
  ----------
  name : str
      Name of the layer.
  data : gpd.GeoDataFrame, pd.DataFrame, shapely geometry, or list
      The geometry data. Can be a GeoDataFrame, DataFrame with a 'geometry' column, a shapely geometry, or a list of shapely geometries.
  crs : cartopy.crs.Projection, optional
      The coordinate reference system. Defaults to PlateCarree if not specified.
  geometry : str or array-like, optional
      For DataFrame input: the column name or array of geometries to use as the geometry column.
  attrs : dict, optional
      Additional attribute columns to add to the GeoDataFrame.

  Notes
  -----
  - For DataFrame input, a 'geometry' column must be present or specified via the `geometry` argument.
  - For shapely input, a GeoDataFrame is constructed with a single geometry or a list of geometries.
  - The resulting data is always stored as a GeoDataFrame with a named geometry column.
  - The geometry column name can be changed after construction via the `.geometry_name` property.
  """
  def __init__(
      self,
      name: str,
      data: Any,
      crs: ccrs.Projection = None,
      geometry: Optional[Union[str, Any]] = None,
      attrs: Optional[dict] = None,
      geometry_name: str = 'geometry'
  ):
    gdf = self._validate_data(data, crs, geometry, attrs, geometry_name)
    super().__init__(name=name, kind='geometry', data=gdf, crs=crs)
    self._geometry_name = geometry_name

  def _validate_data(self, data, crs, geometry, attrs, geometry_name):
    # Accept GeoDataFrame
    if isinstance(data, gpd.GeoDataFrame):
      gdf = data.copy()
      if geometry_name != gdf.geometry.name:
        gdf = gdf.set_geometry(geometry_name)
      if crs is not None:
        gdf = gdf.set_crs(crs, allow_override=True)
    # Accept DataFrame with geometry column
    elif isinstance(data, pd.DataFrame) and (geometry is not None or 'geometry' in data.columns):
      geom_col = geometry if geometry is not None else 'geometry'
      gdf = gpd.GeoDataFrame(data.copy(), geometry=geom_col, crs=crs)
      if geometry_name != gdf.geometry.name:
        gdf = gdf.set_geometry(geometry_name)
    # Accept shapely geometry or list of geometries
    elif hasattr(data, '__geo_interface__') or hasattr(data, 'geom_type') or (
        isinstance(data, (list, tuple)) and all(hasattr(g, '__geo_interface__') or hasattr(g, 'geom_type') for g in data)):
      geoms = [data] if hasattr(data, 'geom_type') or hasattr(data, '__geo_interface__') else list(data)
      gdf = gpd.GeoDataFrame({geometry_name: geoms})
      gdf = gdf.set_geometry(geometry_name)
      if crs is not None:
        gdf = gdf.set_crs(crs)
      if attrs is not None:
        for k, v in attrs.items():
          gdf[k] = v
    else:
      raise TypeError("GeometryLayer data must be a GeoDataFrame, DataFrame with geometry column, or shapely geometry/list")
    return gdf

  @property
  def geometry_name(self):
    return self._geometry_name

  @geometry_name.setter
  def geometry_name(self, value):
    self.data = self.data.set_geometry(value)
    self._geometry_name = value

  @property
  def geometry(self):
    return self.data.geometry

  @property
  def values(self):
    return self.data[self._geometry_name].values

  def to_geodataframe(self):
    return self.data.copy()

  def to_dataframe(self):
    return pd.DataFrame(self.data)

  def to_dict(self):
    return self.data.to_dict(orient='list')

  # @property
  # def bounds(self):
  #   return self.data.total_bounds

  def to_grid(self, new_name=None, kw_gridlayer=None):
    raise NotImplementedError("GeometryLayer.to_grid() is not yet implemented.")
  
  def to_point(self, new_name=None, kw_pointlayer=None):
    raise NotImplementedError("GeometryLayer.to_point() is not yet implemented.")


class SHLayer(Layer):
  def __init__(self, name, data, crs=None):
    validated_data = self._validate_data(data)
    super().__init__(name=name, kind='sh', data=validated_data, crs=crs)
    if isinstance(validated_data, pysh.SHGravCoeffs):
      self._SHkind = 'grav'
    elif isinstance(validated_data, pysh.SHMagCoeffs):
      self._SHkind = 'mag'
    else:
      self._SHkind = None

  def _validate_data(self, data):
    if not isinstance(data, (pysh.SHCoeffs, pysh.SHGravCoeffs, pysh.SHMagCoeffs)):
      raise TypeError("SHLayer data must be an instance of pyshtools.SHCoeffs, pyshtools.SHGravCoeffs, or pyshtools.SHMagCoeffs.")
    return data
  
  @property
  def crs(self):
    raise AttributeError("SHLayer does not use a CRS.")
  
  @property
  def lmax(self):
    return self.data.lmax
  
  @property
  def wavelength(self, R0: float=config.getfloat(config["BODY"]["body"], "R0")):
    return 2*np.pi * R0 / np.sqrt(self.lmax*(self.lmax+1))
  
  # def __getattr__(self, attr):
  #   return getattr(self.data, attr)

  def to_numpy(self):
    return self.data.coeffs
  
  def to_grid(self, new_name=None, kw_expand=None, kw_gridlayer=None):
    name = new_name or self.name
    kw_expand = kw_expand or {}
    kw_gridlayer = kw_gridlayer or {}
    if kw_expand.get('grid') == 'GLQ':
      raise NotImplementedError("For now, `GridLayer`s support only regular grids, not Gauss-Legendre quadrature (GLQ) grids.")
    grid = self.data.expand(**kw_expand)
    grid_layer = GridLayer(
      name=name,
      data=grid.to_xarray(),
      x='lon',
      y='lat',
      crs=PLATE_CARREE,
      **kw_gridlayer
    )
    return grid_layer

  
